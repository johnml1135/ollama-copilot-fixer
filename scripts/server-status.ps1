<#
.SYNOPSIS
    Shared helpers for parsing and displaying llama-server runtime status.
#>

function Format-LlamaNumber {
    param([object]$Value)

    if ($null -eq $Value) { return 'unknown' }
    if ($Value -is [string] -and $Value -eq '') { return 'unknown' }
    try { return ('{0:N0}' -f [double]$Value) } catch { return [string]$Value }
}

function Format-LlamaMiB {
    param([object]$MiB)

    if ($null -eq $MiB) { return 'unknown' }
    if ($MiB -is [string] -and $MiB -eq '') { return 'unknown' }
    $value = [double]$MiB
    if ($value -ge 1024) {
        return ('{0:N2} GiB ({1:N0} MiB)' -f ($value / 1024), $value)
    }
    return ('{0:N2} MiB' -f $value)
}

function Get-LlamaUniqueLines {
    param([string[]]$Lines)

    $seen = @{}
    foreach ($line in @($Lines)) {
        if ($null -eq $line) { continue }
        if (-not $seen.ContainsKey($line)) {
            $seen[$line] = $true
            $line
        }
    }
}

function Read-LlamaLogLines {
    param(
        [string]$LogFile,
        [string]$ErrorLogFile,
        [int]$Head = 1200,
        [int]$Tail = 800
    )

    $signalPattern = 'common_params_fit_impl:|HTTP server listening|server is listening on|model loaded|error loading model|failed to load|failed to bind|address already in use|out of memory|cannot allocate|cuda error|exception|aborted|fatal'
    $lines = @()
    foreach ($path in @($LogFile, $ErrorLogFile)) {
        if (-not $path -or -not (Test-Path $path)) { continue }
        $lines += @(Get-Content -Path $path -TotalCount $Head -ErrorAction SilentlyContinue)
        $lines += @(Get-Content -Path $path -Tail $Tail -ErrorAction SilentlyContinue)
        $lines += @(Select-String -Path $path -Pattern $signalPattern -ErrorAction SilentlyContinue | ForEach-Object { $_.Line })
    }

    @(Get-LlamaUniqueLines -Lines $lines)
}

function Get-LlamaLastRegexMatch {
    param(
        [string[]]$Lines,
        [string]$Pattern
    )

    for ($index = @($Lines).Count - 1; $index -ge 0; $index--) {
        $match = [regex]::Match($Lines[$index], $Pattern)
        if ($match.Success) { return $match }
    }

    return $null
}

function Get-LlamaBufferMatches {
    param(
        [string[]]$Lines,
        [string]$Pattern
    )

    $buffers = @()
    foreach ($line in @($Lines)) {
        $match = [regex]::Match($line, $Pattern)
        if ($match.Success) {
            $buffers += [pscustomobject]@{
                Name = $match.Groups[1].Value.Trim()
                MiB  = [double]$match.Groups[2].Value
                Line = $line.Trim()
            }
        }
    }

    @($buffers)
}

function Get-LlamaObjectValue {
    param(
        [object]$Object,
        [string]$Name
    )

    if ($null -eq $Object -or [string]::IsNullOrWhiteSpace($Name)) {
        return $null
    }

    if ($Object -is [System.Collections.IDictionary]) {
        if ($Object.Contains($Name)) {
            return $Object[$Name]
        }

        foreach ($key in @($Object.Keys)) {
            if ([string]$key -eq $Name) {
                return $Object[$key]
            }
        }

        return $null
    }

    if ($Object.PSObject.Properties.Name -contains $Name) {
        return $Object.PSObject.Properties[$Name].Value
    }

    return $null
}

function Get-LlamaMetricValue {
    param(
        [string]$MetricsText,
        [string]$Name
    )

    if ([string]::IsNullOrWhiteSpace($MetricsText) -or [string]::IsNullOrWhiteSpace($Name)) {
        return $null
    }

    $pattern = '(?m)^' + [regex]::Escape($Name) + '(?:\{[^}]*\})?\s+([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$'
    $match = [regex]::Match($MetricsText, $pattern)
    if (-not $match.Success) {
        return $null
    }

    [double]$match.Groups[1].Value
}

function Get-LlamaSlotStateLabel {
    param([object]$State)

    if ($null -eq $State) { return 'unknown' }

    if ($State -is [string]) {
        $trimmed = $State.Trim()
        if ($trimmed -eq '') { return 'unknown' }
        return $trimmed.ToLowerInvariant()
    }

    try {
        $stateValue = [int]$State
    } catch {
        return [string]$State
    }

    switch ($stateValue) {
        0 { return 'idle' }
        default { return "state $stateValue" }
    }
}

function Get-LlamaEndpointTelemetrySnapshot {
    param(
        [string]$MetricsText,
        [object[]]$Slots
    )

    $requestsProcessing = Get-LlamaMetricValue -MetricsText $MetricsText -Name 'llamacpp:requests_processing'
    $requestsDeferred = Get-LlamaMetricValue -MetricsText $MetricsText -Name 'llamacpp:requests_deferred'
    $contextHighWatermark = Get-LlamaMetricValue -MetricsText $MetricsText -Name 'llamacpp:n_tokens_max'

    $normalizedSlots = @()
    foreach ($slot in @($Slots)) {
        if ($null -eq $slot) { continue }

        if ($slot -is [System.Array]) {
            $normalizedSlots += @($slot)
            continue
        }

        $normalizedSlots += $slot
    }

    $slotSnapshots = @()
    foreach ($slot in @($normalizedSlots)) {
        if ($null -eq $slot) { continue }

        $slotId = Get-LlamaObjectValue -Object $slot -Name 'id'
        $slotState = Get-LlamaObjectValue -Object $slot -Name 'state'
        $slotContext = Get-LlamaObjectValue -Object $slot -Name 'n_ctx'
        $slotPredict = Get-LlamaObjectValue -Object $slot -Name 'n_predict'
        $slotPast = Get-LlamaObjectValue -Object $slot -Name 'n_past'
        $slotDecoded = Get-LlamaObjectValue -Object $slot -Name 'n_decoded'
        $slotRemaining = Get-LlamaObjectValue -Object $slot -Name 'n_remaining'

        $stateLabel = Get-LlamaSlotStateLabel -State $slotState
        $slotSnapshots += [pscustomobject]@{
            Id         = $slotId
            State      = $slotState
            StateLabel = $stateLabel
            Context    = $slotContext
            Predict    = $slotPredict
            Past       = $slotPast
            Decoded    = $slotDecoded
            Remaining  = $slotRemaining
            IsBusy     = $stateLabel -ne 'idle'
        }
    }

    [pscustomobject]@{
        PromptTokensPerSecond    = Get-LlamaMetricValue -MetricsText $MetricsText -Name 'llamacpp:prompt_tokens_seconds'
        PredictedTokensPerSecond = Get-LlamaMetricValue -MetricsText $MetricsText -Name 'llamacpp:predicted_tokens_seconds'
        RequestsProcessing       = if ($null -ne $requestsProcessing) { [int]$requestsProcessing } else { $null }
        RequestsDeferred         = if ($null -ne $requestsDeferred) { [int]$requestsDeferred } else { $null }
        ContextHighWatermark     = if ($null -ne $contextHighWatermark) { [int]$contextHighWatermark } else { $null }
        TotalSlots               = @($slotSnapshots).Count
        BusySlots                = @($slotSnapshots | Where-Object { $_.IsBusy }).Count
        Slots                    = @($slotSnapshots)
    }
}

function Get-LlamaServerLogSnapshot {
    param(
        [string]$LogFile,
        [string]$ErrorLogFile,
        [string]$Text
    )

    if ($PSBoundParameters.ContainsKey('Text')) {
        $lines = @($Text -split "`r?`n")
    } else {
        $lines = Read-LlamaLogLines -LogFile $LogFile -ErrorLogFile $ErrorLogFile
    }

    $lines = @(Get-LlamaUniqueLines -Lines $lines | Where-Object { $_ -ne '' })
    $joined = $lines -join "`n"

    $cudaDeviceCount = $null
    $cudaTotalMiB = $null
    $cudaInit = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'ggml_cuda_init: found\s+(\d+)\s+CUDA devices \(Total VRAM:\s+([\d.]+)\s+MiB\)'
    if ($cudaInit) {
        $cudaDeviceCount = [int]$cudaInit.Groups[1].Value
        $cudaTotalMiB = [double]$cudaInit.Groups[2].Value
    }

    $cudaDevices = @()
    foreach ($line in $lines) {
        $device = [regex]::Match($line, '^\s*Device\s+(\d+):\s+(.+?),\s+compute capability\s+([^,]+).*VRAM:\s+([\d.]+)\s+MiB')
        if ($device.Success) {
            $cudaDevices += [pscustomobject]@{
                Index             = [int]$device.Groups[1].Value
                Name              = $device.Groups[2].Value.Trim()
                ComputeCapability = $device.Groups[3].Value.Trim()
                VramMiB           = [double]$device.Groups[4].Value
            }
        }
    }

    $backends = @()
    foreach ($line in $lines) {
        $backend = [regex]::Match($line, 'load_backend: loaded\s+(.+?)\s+backend from\s+(.+)$')
        if ($backend.Success) {
            $backends += [pscustomobject]@{
                Name = $backend.Groups[1].Value.Trim()
                Path = $backend.Groups[2].Value.Trim()
            }
        }
    }

    $offloadedLayers = $null
    $totalLayers = $null
    $offloaded = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'load_tensors: offloaded\s+(\d+)/(\d+)\s+layers to GPU'
    if ($offloaded) {
        $offloadedLayers = [int]$offloaded.Groups[1].Value
        $totalLayers = [int]$offloaded.Groups[2].Value
    }

    $repeatingGpuLayers = $null
    $repeating = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'load_tensors: offloading\s+(\d+)\s+repeating layers to GPU'
    if ($repeating) { $repeatingGpuLayers = [int]$repeating.Groups[1].Value }

    $cpuMappedModelMiB = $null
    $cpuModel = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'load_tensors:\s+CPU[^=]*model buffer size =\s+([\d.]+)\s+MiB'
    if ($cpuModel) { $cpuMappedModelMiB = [double]$cpuModel.Groups[1].Value }

    $gpuModelBuffers = Get-LlamaBufferMatches -Lines $lines -Pattern 'load_tensors:\s+(CUDA\d+) model buffer size =\s+([\d.]+)\s+MiB'
    $gpuKvBuffers = Get-LlamaBufferMatches -Lines $lines -Pattern 'llama_kv_cache:\s+(CUDA\d+) KV buffer size =\s+([\d.]+)\s+MiB'
    $gpuRecurrentBuffers = Get-LlamaBufferMatches -Lines $lines -Pattern 'llama_memory_recurrent:\s+(CUDA\d+) RS buffer size =\s+([\d.]+)\s+MiB'
    $computeBuffers = Get-LlamaBufferMatches -Lines $lines -Pattern '(?:sched_reserve|alloc_compute_meta):\s+(CUDA\d+|CUDA_Host|CPU)\s+compute buffer size =\s+([\d.]+)\s+MiB'

    $contextAllocated = $null
    $contextMatch = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'llama_context: n_ctx\s+=\s+(\d+)'
    if ($contextMatch) { $contextAllocated = [int]$contextMatch.Groups[1].Value }

    $contextSeq = $null
    $contextSeqMatch = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'llama_context: n_ctx_seq\s+=\s+(\d+)'
    if ($contextSeqMatch) { $contextSeq = [int]$contextSeqMatch.Groups[1].Value }

    $nativeContext = $null
    $nativeContextMatch = Get-LlamaLastRegexMatch -Lines $lines -Pattern '(?:print_info|llama_context): n_ctx_train\s+=\s+(\d+)'
    if (-not $nativeContextMatch) {
        $nativeContextMatch = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'llama_model_loader: - kv\s+\d+:\s+\S+\.context_length\s+\S+\s+=\s+(\d+)'
    }
    if ($nativeContextMatch) { $nativeContext = [int]$nativeContextMatch.Groups[1].Value }

    $kvCacheMiB = $null
    $kvCacheCells = $null
    $kvCacheLayers = $null
    $kvCacheSeqs = $null
    $kvCacheMaxSeqs = $null
    $kvCacheKType = $null
    $kvCacheVType = $null
    $kvCacheKMiB = $null
    $kvCacheVMiB = $null
    $kvCacheTotalMiB = $null
    $kvCacheDetails = @()
    foreach ($line in $lines) {
        $kvDetail = [regex]::Match($line, 'llama_kv_cache: size =\s+([\d.]+)\s+MiB \((\d+) cells,\s+(\d+) layers,\s+(\d+)/(\d+) seqs\), K \(([^)]+)\):\s+([\d.]+)\s+MiB, V \(([^)]+)\):\s+([\d.]+)\s+MiB')
        if ($kvDetail.Success) {
            $kvCacheDetails += [pscustomobject]@{
                MiB     = [double]$kvDetail.Groups[1].Value
                Cells   = [int]$kvDetail.Groups[2].Value
                Layers  = [int]$kvDetail.Groups[3].Value
                Seqs    = [int]$kvDetail.Groups[4].Value
                MaxSeqs = [int]$kvDetail.Groups[5].Value
                KType   = $kvDetail.Groups[6].Value.Trim()
                KMiB    = [double]$kvDetail.Groups[7].Value
                VType   = $kvDetail.Groups[8].Value.Trim()
                VMiB    = [double]$kvDetail.Groups[9].Value
                Line    = $line.Trim()
            }
        }
    }
    if (@($kvCacheDetails).Count -gt 0) {
        $primaryKvCache = @($kvCacheDetails | Sort-Object -Property MiB -Descending | Select-Object -First 1)[0]
        $kvCacheTotalMiB = (@($kvCacheDetails) | Measure-Object -Property MiB -Sum).Sum
        $kvCacheMiB = $primaryKvCache.MiB
        $kvCacheCells = $primaryKvCache.Cells
        $kvCacheLayers = $primaryKvCache.Layers
        $kvCacheSeqs = $primaryKvCache.Seqs
        $kvCacheMaxSeqs = $primaryKvCache.MaxSeqs
        $kvCacheKType = $primaryKvCache.KType
        $kvCacheKMiB = $primaryKvCache.KMiB
        $kvCacheVType = $primaryKvCache.VType
        $kvCacheVMiB = $primaryKvCache.VMiB
    } elseif (@($gpuKvBuffers).Count -gt 0) {
        $kvCacheMiB = (@($gpuKvBuffers) | Measure-Object -Property MiB -Sum).Sum
        $kvCacheTotalMiB = $kvCacheMiB
    }

    $batch = $null
    $batchMatch = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'llama_context: n_batch\s+=\s+(\d+)'
    if ($batchMatch) { $batch = [int]$batchMatch.Groups[1].Value }

    $ubatch = $null
    $ubatchMatch = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'llama_context: n_ubatch\s+=\s+(\d+)'
    if ($ubatchMatch) { $ubatch = [int]$ubatchMatch.Groups[1].Value }

    $flashAttention = $null
    $flashMatch = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'llama_context: flash_attn\s+=\s+(\S+)'
    if ($flashMatch) { $flashAttention = $flashMatch.Groups[1].Value.Trim() }

    $chatTemplateThinking = $null
    $thinkingMatch = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'chat template, thinking =\s+(\d+)'
    if ($thinkingMatch) { $chatTemplateThinking = [int]$thinkingMatch.Groups[1].Value }

    $projectedUseMiB = $null
    $projectedFreeMiB = $null
    $projected = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'projected to use\s+([\d.]+)\s+MiB of device memory vs\.\s+([\d.]+)\s+MiB of free device memory'
    if ($projected) {
        $projectedUseMiB = [double]$projected.Groups[1].Value
        $projectedFreeMiB = [double]$projected.Groups[2].Value
    }

    $projectedLeaveMiB = $null
    $leave = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'will leave\s+([\d.]+)\s+>=\s+([\d.]+)\s+MiB of free device memory'
    if ($leave) { $projectedLeaveMiB = [double]$leave.Groups[1].Value }

    $listenLine = $null
    $listenUrl = $null
    $listen = Get-LlamaLastRegexMatch -Lines $lines -Pattern '^\s*main: server is listening on\s+(.+)$'
    if ($listen) {
        $listenLine = $listen.Value.Trim()
        $listenUrl = $listen.Groups[1].Value.Trim()
    } else {
        $listen = Get-LlamaLastRegexMatch -Lines $lines -Pattern '^\s*(?:INFO|VERB|WARN|ERROR)\s+\[[^\]]+\]\s+HTTP server listening\s+\|.*hostname="([^"]+)"\s+port="?([^"\s]+)"?'
        if ($listen) {
            $listenLine = $listen.Value.Trim()
            $listenUrl = 'http://{0}:{1}' -f $listen.Groups[1].Value.Trim(), $listen.Groups[2].Value.Trim()
        }
    }

    $failureLine = $null
    $fatalPattern = '(?i)(error loading model|failed to load|failed to bind|address already in use|out of memory|cannot allocate|cuda error|exception|aborted|fatal)'
    $fatalLogPrefix = '^(?:\s*error loading model:|\s*main:|\s*srv\s+load_model:|\s*load_tensors:|\s*common_params_fit_impl:|\s*llama_[^:]+:|\s*(?:ERROR|WARN)\s+\[[^\]]+\])'
    for ($index = $lines.Count - 1; $index -ge 0; $index--) {
        $candidateLine = $lines[$index].Trim()
        if ($candidateLine -notmatch $fatalLogPrefix) { continue }
        if ($candidateLine -match $fatalPattern) {
            $failureLine = $candidateLine
            break
        }
    }

    $lastSignal = $null
    $lastSignalShort = $null
    foreach ($pattern in @(
            '^\s*(?:INFO|VERB|WARN|ERROR)\s+\[[^\]]+\]\s+HTTP server listening\s+\|.+$',
            '^\s*main: server is listening on\s+.+$',
            '^\s*(?:INFO|VERB|WARN|ERROR)\s+\[[^\]]+\]\s+model loaded\s+\|.+$',
            '^\s*main: model loaded\s*$',
            '^\s*srv\s+load_model: initializing slots.+$',
            '^\s*common_init_from_params: warming up the model.+$',
            '^\s*load_tensors: loading model tensors.+$',
            '^\s*main: loading model\s*$',
            '^\s*srv\s+load_model: loading model.+$')) {
        $signal = Get-LlamaLastRegexMatch -Lines $lines -Pattern $pattern
        if ($signal) {
            $lastSignal = $signal.Value.Trim()
            # Build a short, human-friendly version
            $short = $lastSignal
            if ($lastSignal -match 'HTTP server listening') {
                $hostMatch = [regex]::Match($lastSignal, 'hostname="([^"]+)"\s+port="([^"]+)"')
                if ($hostMatch.Success) {
                    $short = "HTTP server listening on $($hostMatch.Groups[1].Value):$($hostMatch.Groups[2].Value)"
                }
            }
            $lastSignalShort = $short
            break
        }
    }

    # -- Parse latest prompt eval tok/sec --------------------------------------
    $promptEvalTokPerSec = $null
    $promptEvalMsPerToken = $null
    $promptEvalTokens = $null
    $promptEvalTime = Get-LlamaLastRegexMatch -Lines $lines -Pattern 'prompt eval time\s*=\s+([\d.]+)\s+ms\s*/\s+(\d+)\s+tokens\s*\(\s+([\d.]+)\s+ms per token,\s+([\d.]+)\s+tokens per second\)'
    if ($promptEvalTime) {
        $promptEvalMsPerToken = [double]$promptEvalTime.Groups[3].Value
        $promptEvalTokPerSec = [double]$promptEvalTime.Groups[4].Value
        $promptEvalTokens = [int]$promptEvalTime.Groups[2].Value
    }

    # -- Parse latest generation eval tok/sec ----------------------------------
    $genEvalTokPerSec = $null
    $genEvalMsPerToken = $null
    $genEvalTokens = $null
    $genEvalTime = Get-LlamaLastRegexMatch -Lines $lines -Pattern '^\s*eval time\s*=\s+([\d.]+)\s+ms\s*/\s+(\d+)\s+tokens\s*\(\s+([\d.]+)\s+ms per token,\s+([\d.]+)\s+tokens per second\)'
    if ($genEvalTime) {
        $genEvalMsPerToken = [double]$genEvalTime.Groups[3].Value
        $genEvalTokPerSec = [double]$genEvalTime.Groups[4].Value
        $genEvalTokens = [int]$genEvalTime.Groups[2].Value
    }

    $loadState = 'unknown'
    if ($listenLine) {
        $loadState = 'ready'
    } elseif ($failureLine) {
        $loadState = 'failed'
    } elseif ($joined -match '(?m)(main: loading model|srv\s+load_model: loading model|load_tensors: loading model tensors|warming up the model|initializing slots)') {
        $loadState = 'loading'
    } elseif ($lines.Count -gt 0) {
        $loadState = 'starting'
    }

    if ($listenLine) {
        $failureLine = $null
    }

    [pscustomobject]@{
        LoadState            = $loadState
        LastSignal           = $lastSignal
        LastSignalShort      = $lastSignalShort
        FailureLine          = $failureLine
        ListenUrl            = $listenUrl
        PromptEvalTokPerSec  = $promptEvalTokPerSec
        PromptEvalMsPerToken = $promptEvalMsPerToken
        PromptEvalTokens     = $promptEvalTokens
        GenEvalTokPerSec     = $genEvalTokPerSec
        GenEvalMsPerToken    = $genEvalMsPerToken
        GenEvalTokens        = $genEvalTokens
        CudaDeviceCount      = $cudaDeviceCount
        CudaTotalMiB         = $cudaTotalMiB
        CudaDevices          = @($cudaDevices)
        Backends             = @($backends)
        OffloadedLayers      = $offloadedLayers
        TotalLayers          = $totalLayers
        RepeatingGpuLayers   = $repeatingGpuLayers
        CpuMappedModelMiB    = $cpuMappedModelMiB
        GpuModelBuffers      = @($gpuModelBuffers)
        GpuKvBuffers         = @($gpuKvBuffers)
        GpuRecurrentBuffers  = @($gpuRecurrentBuffers)
        ComputeBuffers       = @($computeBuffers)
        ContextAllocated     = $contextAllocated
        ContextSeq           = $contextSeq
        NativeContext        = $nativeContext
        KvCacheMiB           = $kvCacheMiB
        KvCacheCells         = $kvCacheCells
        KvCacheLayers        = $kvCacheLayers
        KvCacheSeqs          = $kvCacheSeqs
        KvCacheMaxSeqs       = $kvCacheMaxSeqs
        KvCacheKType         = $kvCacheKType
        KvCacheVType         = $kvCacheVType
        KvCacheKMiB          = $kvCacheKMiB
        KvCacheVMiB          = $kvCacheVMiB
        KvCacheTotalMiB      = $kvCacheTotalMiB
        KvCacheDetails       = @($kvCacheDetails)
        Batch                = $batch
        UBatch               = $ubatch
        FlashAttention       = $flashAttention
        ChatTemplateThinking = $chatTemplateThinking
        ProjectedUseMiB      = $projectedUseMiB
        ProjectedFreeMiB     = $projectedFreeMiB
        ProjectedLeaveMiB    = $projectedLeaveMiB
        LineCount            = $lines.Count
    }
}

function Format-LlamaServerBriefStatus {
    param([pscustomobject]$Snapshot)

    if (-not $Snapshot) { return 'status unavailable' }

    $parts = @($Snapshot.LoadState)

    if ($Snapshot.OffloadedLayers -and $Snapshot.TotalLayers) {
        $parts += ('{0}/{1} layers on GPU' -f $Snapshot.OffloadedLayers, $Snapshot.TotalLayers)
    }
    if ($Snapshot.ContextAllocated) {
        $parts += ('ctx {0}' -f (Format-LlamaNumber $Snapshot.ContextAllocated))
    }
    if ($Snapshot.KvCacheMiB) {
        $kvTypes = ''
        if ($Snapshot.KvCacheKType -and $Snapshot.KvCacheVType) {
            $kvTypes = (' {0}/{1}' -f $Snapshot.KvCacheKType, $Snapshot.KvCacheVType)
        }
        $briefKvMiB = if (@($Snapshot.KvCacheDetails).Count -gt 1 -and $Snapshot.KvCacheTotalMiB) { $Snapshot.KvCacheTotalMiB } else { $Snapshot.KvCacheMiB }
        $kvLabel = if (@($Snapshot.KvCacheDetails).Count -gt 1) { 'KV total' } else { 'KV' }
        $parts += ('{0}{1} {2}' -f $kvLabel, $kvTypes, (Format-LlamaMiB $briefKvMiB))
    }
    if (@($Snapshot.GpuModelBuffers).Count -gt 0) {
        $gpuModelMiB = (@($Snapshot.GpuModelBuffers) | Measure-Object -Property MiB -Sum).Sum
        $parts += ('GPU model {0}' -f (Format-LlamaMiB $gpuModelMiB))
    }
    if ($Snapshot.ListenUrl) {
        $parts += ('listening {0}' -f $Snapshot.ListenUrl)
    } elseif ($Snapshot.FailureLine) {
        $parts += $Snapshot.FailureLine
    } elseif ($Snapshot.LastSignal) {
        $parts += $Snapshot.LastSignal
    }

    $parts -join '; '
}

function Write-LlamaServerLogStatus {
    param(
        [pscustomobject]$Snapshot,
        [pscustomobject]$Info
    )

    if (-not $Snapshot -or $Snapshot.LoadState -eq 'unknown') { return }

    $stateColor = 'Yellow'
    if ($Snapshot.LoadState -eq 'ready') { $stateColor = 'Green' }
    if ($Snapshot.LoadState -eq 'failed') { $stateColor = 'Red' }

    Write-Host ("  Load state  : {0}" -f $Snapshot.LoadState) -ForegroundColor $stateColor
    if ($Snapshot.LastSignalShort) {
        Write-Host ("  Last signal : {0}" -f $Snapshot.LastSignalShort)
    }
    if ($Snapshot.FailureLine) {
        Write-Host ("  Failure     : {0}" -f $Snapshot.FailureLine) -ForegroundColor Red
    }
    if ($Snapshot.ListenUrl) {
        Write-Host ("  Listening   : {0}" -f $Snapshot.ListenUrl)
    }

    # -- Performance (tok/sec) ------------------------------------------------
    if ($Snapshot.PromptEvalTokPerSec -or $Snapshot.GenEvalTokPerSec) {
        Write-Host ""
        Write-Host "Performance:" -ForegroundColor Cyan
        if ($Snapshot.PromptEvalTokPerSec) {
            $line = "  Prompt eval : {0:N1} tok/s ({1:N2} ms/tok" -f $Snapshot.PromptEvalTokPerSec, $Snapshot.PromptEvalMsPerToken
            if ($Snapshot.PromptEvalTokens) { $line += ", $($Snapshot.PromptEvalTokens) tokens" }
            $line += ")"
            Write-Host $line
        }
        if ($Snapshot.GenEvalTokPerSec) {
            $line = "  Gen eval    : {0:N1} tok/s ({1:N2} ms/tok" -f $Snapshot.GenEvalTokPerSec, $Snapshot.GenEvalMsPerToken
            if ($Snapshot.GenEvalTokens) { $line += ", $($Snapshot.GenEvalTokens) tokens" }
            $line += ")"
            Write-Host $line
        }
    }

    if ($Info -or $Snapshot.ContextAllocated -or $Snapshot.NativeContext -or $Snapshot.KvCacheMiB) {
        Write-Host ""
        Write-Host "Context and cache:" -ForegroundColor Cyan
        if ($Info -and $Info.Context) {
            $line = "  Requested   : $(Format-LlamaNumber $Info.Context) tokens"
            if ($Info.PSObject.Properties.Name -contains 'MaxContext' -and $Info.MaxContext) {
                $line += " (profile native max $(Format-LlamaNumber $Info.MaxContext))"
            }
            Write-Host $line
        }
        if ($Snapshot.ContextAllocated) {
            $line = "  Allocated   : $(Format-LlamaNumber $Snapshot.ContextAllocated) tokens"
            if ($Snapshot.ContextSeq) { $line += " (seq $(Format-LlamaNumber $Snapshot.ContextSeq))" }
            if ($Snapshot.NativeContext) { $line += "; model train/native $(Format-LlamaNumber $Snapshot.NativeContext)" }
            Write-Host $line
        } elseif ($Snapshot.NativeContext) {
            Write-Host ("  Native max  : {0} tokens" -f (Format-LlamaNumber $Snapshot.NativeContext))
        }
        if ($Snapshot.KvCacheMiB) {
            $line = "  KV cache    : $(Format-LlamaMiB $Snapshot.KvCacheMiB) primary"
            if ($Snapshot.KvCacheCells) { $line += "; $(Format-LlamaNumber $Snapshot.KvCacheCells) cells" }
            if ($Snapshot.KvCacheLayers) { $line += "; $($Snapshot.KvCacheLayers) layers" }
            if ($Snapshot.KvCacheKType -and $Snapshot.KvCacheVType) {
                $line += "; K $($Snapshot.KvCacheKType) $(Format-LlamaMiB $Snapshot.KvCacheKMiB), V $($Snapshot.KvCacheVType) $(Format-LlamaMiB $Snapshot.KvCacheVMiB)"
            } elseif ($Info -and $Info.PSObject.Properties.Name -contains 'CacheTypeK') {
                $line += "; configured K $($Info.CacheTypeK), V $($Info.CacheTypeV)"
            }
            if (@($Snapshot.KvCacheDetails).Count -gt 1 -and $Snapshot.KvCacheTotalMiB) {
                $line += "; total $(Format-LlamaMiB $Snapshot.KvCacheTotalMiB) across $(@($Snapshot.KvCacheDetails).Count) caches"
            }
            Write-Host $line
            if (@($Snapshot.KvCacheDetails).Count -gt 1) {
                $detailIndex = 1
                foreach ($detail in @($Snapshot.KvCacheDetails | Sort-Object -Property MiB -Descending)) {
                    Write-Host ("  KV detail {0} : {1}; {2} cells; {3} layers; K {4}, V {5}" -f $detailIndex, (Format-LlamaMiB $detail.MiB), (Format-LlamaNumber $detail.Cells), $detail.Layers, $detail.KType, $detail.VType)
                    $detailIndex++
                }
            }
        } elseif ($Info -and $Info.PSObject.Properties.Name -contains 'CacheTypeK') {
            Write-Host ("  KV config   : K {0}, V {1}" -f $Info.CacheTypeK, $Info.CacheTypeV)
        }
        if ($Snapshot.Batch -or $Snapshot.UBatch) {
            Write-Host ("  Batch       : n_batch {0}, n_ubatch {1}" -f (Format-LlamaNumber $Snapshot.Batch), (Format-LlamaNumber $Snapshot.UBatch))
        }
    }

    if ($Snapshot.OffloadedLayers -or @($Snapshot.GpuModelBuffers).Count -gt 0 -or $Snapshot.CpuMappedModelMiB) {
        Write-Host ""
        Write-Host "GPU / CPU split:" -ForegroundColor Cyan
        if ($Snapshot.OffloadedLayers -and $Snapshot.TotalLayers) {
            $line = "  Layers      : $($Snapshot.OffloadedLayers)/$($Snapshot.TotalLayers) on GPU"
            if ($Snapshot.RepeatingGpuLayers) { $line += " ($($Snapshot.RepeatingGpuLayers) repeating + output)" }
            if ($Info -and $Info.PSObject.Properties.Name -contains 'GpuLayers') { $line += "; requested --gpu-layers $($Info.GpuLayers)" }
            Write-Host $line
        } elseif ($Info -and $Info.PSObject.Properties.Name -contains 'GpuLayers') {
            Write-Host ("  Requested   : --gpu-layers {0}" -f $Info.GpuLayers)
        }
        if (@($Snapshot.GpuModelBuffers).Count -gt 0) {
            foreach ($buffer in @($Snapshot.GpuModelBuffers)) {
                Write-Host ("  {0,-11}: model weights {1}" -f $buffer.Name, (Format-LlamaMiB $buffer.MiB))
            }
        }
        if ($Snapshot.CpuMappedModelMiB) {
            Write-Host ("  CPU mapped  : model weights {0}" -f (Format-LlamaMiB $Snapshot.CpuMappedModelMiB))
        }
        foreach ($buffer in @($Snapshot.GpuKvBuffers)) {
            Write-Host ("  {0,-11}: KV cache {1}" -f $buffer.Name, (Format-LlamaMiB $buffer.MiB))
        }
        foreach ($buffer in @($Snapshot.GpuRecurrentBuffers)) {
            Write-Host ("  {0,-11}: recurrent state {1}" -f $buffer.Name, (Format-LlamaMiB $buffer.MiB))
        }
        foreach ($buffer in @($Snapshot.ComputeBuffers)) {
            Write-Host ("  {0,-11}: compute {1}" -f $buffer.Name, (Format-LlamaMiB $buffer.MiB))
        }
        if ($Snapshot.ProjectedUseMiB -or $Snapshot.ProjectedFreeMiB -or $Snapshot.ProjectedLeaveMiB) {
            $line = "  Fit estimate:"
            if ($Snapshot.ProjectedUseMiB) { $line += " uses $(Format-LlamaMiB $Snapshot.ProjectedUseMiB)" }
            if ($Snapshot.ProjectedFreeMiB) { $line += " of $(Format-LlamaMiB $Snapshot.ProjectedFreeMiB) initially free" }
            if ($Snapshot.ProjectedLeaveMiB) { $line += "; leaves $(Format-LlamaMiB $Snapshot.ProjectedLeaveMiB)" }
            Write-Host $line
        }
    }

    if (@($Snapshot.CudaDevices).Count -gt 0 -or @($Snapshot.Backends).Count -gt 0 -or $Snapshot.FlashAttention -or $null -ne $Snapshot.ChatTemplateThinking) {
        Write-Host ""
        Write-Host "Runtime details:" -ForegroundColor Cyan
        foreach ($device in @($Snapshot.CudaDevices)) {
            Write-Host ("  CUDA{0}      : {1}, cc {2}, VRAM {3}" -f $device.Index, $device.Name, $device.ComputeCapability, (Format-LlamaMiB $device.VramMiB))
        }
        if (@($Snapshot.Backends).Count -gt 0) {
            $backendNames = (@($Snapshot.Backends) | ForEach-Object { $_.Name }) -join ', '
            Write-Host ("  Backends    : {0}" -f $backendNames)
        }
        if ($Snapshot.FlashAttention) {
            Write-Host ("  Flash attn  : {0}" -f $Snapshot.FlashAttention)
        } elseif ($Info -and $Info.PSObject.Properties.Name -contains 'FlashAttention') {
            Write-Host ("  Flash attn  : {0} (configured)" -f $Info.FlashAttention)
        }
        if ($null -ne $Snapshot.ChatTemplateThinking) {
            Write-Host ("  Thinking    : {0} (chat template)" -f $Snapshot.ChatTemplateThinking)
        }
    }
}

function Write-LlamaEndpointTelemetryStatus {
    param([pscustomobject]$Telemetry)

    if (-not $Telemetry) { return }

    $hasMetrics = $null -ne $Telemetry.PromptTokensPerSecond -or
        $null -ne $Telemetry.PredictedTokensPerSecond -or
        $null -ne $Telemetry.RequestsProcessing -or
        $null -ne $Telemetry.RequestsDeferred -or
        $null -ne $Telemetry.ContextHighWatermark
    $hasSlots = $Telemetry.TotalSlots -gt 0

    if (-not $hasMetrics -and -not $hasSlots) { return }

    Write-Host ""
    Write-Host "Live telemetry:" -ForegroundColor Cyan

    if ($null -ne $Telemetry.PromptTokensPerSecond) {
        Write-Host ("  Prompt avg  : {0:N1} tok/s" -f $Telemetry.PromptTokensPerSecond)
    }
    if ($null -ne $Telemetry.PredictedTokensPerSecond) {
        Write-Host ("  Gen avg     : {0:N1} tok/s" -f $Telemetry.PredictedTokensPerSecond)
    }
    if ($null -ne $Telemetry.RequestsProcessing -or $null -ne $Telemetry.RequestsDeferred) {
        Write-Host ("  Requests    : processing {0}, deferred {1}" -f (Format-LlamaNumber $Telemetry.RequestsProcessing), (Format-LlamaNumber $Telemetry.RequestsDeferred))
    }
    if ($null -ne $Telemetry.ContextHighWatermark) {
        Write-Host ("  Ctx high wm : {0} tokens" -f (Format-LlamaNumber $Telemetry.ContextHighWatermark))
    }

    if ($hasSlots) {
        $slotContexts = @($Telemetry.Slots | Where-Object { $null -ne $_.Context } | ForEach-Object { [double]$_.Context })
        $slotLine = "  Slots       : {0}/{1} busy" -f $Telemetry.BusySlots, $Telemetry.TotalSlots
        if ($slotContexts.Count -gt 0) {
            $slotLine += "; max ctx {0}" -f (Format-LlamaNumber (($slotContexts | Measure-Object -Maximum).Maximum))
        }
        Write-Host $slotLine

        foreach ($slot in @($Telemetry.Slots | Select-Object -First 4)) {
            $slotId = if ($null -ne $slot.Id) { $slot.Id } else { '?' }
            $line = "  Slot {0,-5}: {1}" -f $slotId, $slot.StateLabel
            if ($null -ne $slot.Context) {
                $line += "; ctx {0}" -f (Format-LlamaNumber $slot.Context)
            }
            if ($null -ne $slot.Past) {
                $line += "; past {0}" -f (Format-LlamaNumber $slot.Past)
            }
            if ($null -ne $slot.Decoded) {
                $line += "; decoded {0}" -f (Format-LlamaNumber $slot.Decoded)
            }
            if ($null -ne $slot.Remaining) {
                $line += "; remaining {0}" -f (Format-LlamaNumber $slot.Remaining)
            }
            Write-Host $line
        }
    }
}