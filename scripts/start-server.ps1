<#
.SYNOPSIS
    Launcher for the single ik_llama.cpp Qwen3.6 27B profile sized for an
    RTX 3090 24 GB GPU.

.DESCRIPTION
    Workflow:
      1. Ensures a local ik_llama.cpp llama-server.exe exists under
         tools\ik_llama.cpp, invoking scripts\install-llama.ps1 if needed.
      2. Resolves the single supported model and CPU mmproj sidecar from
         Hugging Face into <repo>\models.
      3. Launches llama-server in the background, exposing an OpenAI-compatible
         endpoint at http://127.0.0.1:<Port>/v1 for VS Code Copilot Chat
         or any other OpenAI-compatible client.

.PARAMETER Model
    Optional model key from scripts\models.ps1. The default and only supported
    key is qwen36-27b-mtp-iq4-ks.

.PARAMETER Port
    TCP port to bind. Default 8080.

.PARAMETER NoThink
    Override the profile and disable reasoning/thinking output.

.PARAMETER EnableReasoning
    Explicitly enable reasoning. This is already the profile default because
    the Reddit ik_llama.cpp recipe uses reasoning-format deepseek with
    preserve_thinking:true.

.PARAMETER ContextOverride
    Override the catalog's --ctx-size for this launch.

.EXAMPLE
    .\scripts\start-server.ps1

.EXAMPLE
    .\scripts\start-server.ps1 -NoThink
#>

[CmdletBinding()]
param(
    [string]$Model,
    [int]$Port = 8080,
    [string]$ListenHost = '127.0.0.1',
    [switch]$NoThink,
    [switch]$EnableReasoning,
    [int]$ContextOverride
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$logDir   = Join-Path $repoRoot 'logs'
$logFile  = Join-Path $logDir   'llama-server.log'
$pidFile  = Join-Path $logDir   'llama-server.pid'
$infoFile = Join-Path $logDir   'llama-server.info.json'
$modelsDir = Join-Path $repoRoot 'models'
$qwenTemplateFile = Join-Path $PSScriptRoot 'templates\qwen36-tool-fix.jinja'
$statusScript = Join-Path $PSScriptRoot 'status-server.ps1'

. (Join-Path $PSScriptRoot 'server-status.ps1')

function Wait-LlamaServerStartupStatus {
    param(
        [System.Diagnostics.Process]$Process,
        [string]$LogFile,
        [string]$ErrorLogFile,
        [int]$TimeoutSec = 60,
        [int]$PollSec = 3
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $lastBrief = $null

    Write-Host ""
    Write-Host ("Waiting for llama-server to load (up to {0} seconds)..." -f $TimeoutSec) -ForegroundColor Cyan

    while ($true) {
        $snapshot = Get-LlamaServerLogSnapshot -LogFile $LogFile -ErrorLogFile $ErrorLogFile
        $brief = Format-LlamaServerBriefStatus -Snapshot $snapshot

        if ($brief -and $brief -ne $lastBrief) {
            Write-Host ("  [{0:HH:mm:ss}] {1}" -f (Get-Date), $brief)
            $lastBrief = $brief
        }

        if ($snapshot.LoadState -eq 'ready') { return 'ready' }
        if ($snapshot.LoadState -eq 'failed') { return 'failed' }

        $Process.Refresh()
        if ($Process.HasExited) { return 'exited' }

        $remaining = ($deadline - (Get-Date)).TotalSeconds
        if ($remaining -le 0) { return 'timeout' }

        $sleepSeconds = [math]::Min($PollSec, [math]::Max(1, [int][math]::Ceiling($remaining)))
        Start-Sleep -Seconds $sleepSeconds
    }
}

function Find-LlamaServerExe {
    param([string]$Root)

    if (-not (Test-Path $Root)) { return $null }

    $candidates = @(
        'llama-server.exe',
        'bin\llama-server.exe',
        'build\bin\llama-server.exe',
        'build\bin\Release\llama-server.exe',
        'build\bin\RelWithDebInfo\llama-server.exe',
        'build\examples\server\llama-server.exe',
        'build\examples\server\Release\llama-server.exe',
        'build\examples\server\RelWithDebInfo\llama-server.exe'
    ) | ForEach-Object { Join-Path $Root $_ }

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    $found = Get-ChildItem -Path $Root -Filter 'llama-server.exe' -File -Recurse -ErrorAction SilentlyContinue |
        Sort-Object FullName |
        Select-Object -First 1

    if ($found) { return $found.FullName }
    return $null
}

function ConvertTo-WindowsCommandLineArgument {
    param([string]$Argument)

    if ($null -eq $Argument) { return '""' }
    if ($Argument.Length -eq 0) { return '""' }
    if ($Argument -notmatch '[\s"]') { return $Argument }

    $quoted = '"'
    $backslashes = 0
    foreach ($char in $Argument.ToCharArray()) {
        if ($char -eq '\') {
            $backslashes++
            continue
        }

        if ($char -eq '"') {
            $quoted += ('\' * (($backslashes * 2) + 1))
            $quoted += '"'
            $backslashes = 0
            continue
        }

        if ($backslashes -gt 0) {
            $quoted += ('\' * $backslashes)
            $backslashes = 0
        }
        $quoted += $char
    }

    if ($backslashes -gt 0) {
        $quoted += ('\' * ($backslashes * 2))
    }

    $quoted += '"'
    return $quoted
}

function Get-HuggingFaceResolveUrl {
    param(
        [string]$Repo,
        [string]$File
    )

    $encodedFile = (($File -split '/') | ForEach-Object { [uri]::EscapeDataString($_) }) -join '/'
    return "https://huggingface.co/$Repo/resolve/main/$encodedFile"
}

function Ensure-HuggingFaceFile {
    param(
        [string]$Repo,
        [string]$File,
        [string]$Kind
    )

    $safeRepo = $Repo -replace '[\\/]', '--'
    $targetDir = Join-Path $modelsDir $safeRepo
    $targetPath = Join-Path $targetDir $File

    if (Test-Path $targetPath) {
        return (Resolve-Path $targetPath).Path
    }

    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    $url = Get-HuggingFaceResolveUrl -Repo $Repo -File $File
    $partialPath = "$targetPath.part"
    Remove-Item $partialPath -Force -ErrorAction SilentlyContinue

    Write-Host ""
    Write-Host ("Downloading {0}: {1}/{2}" -f $Kind, $Repo, $File) -ForegroundColor Yellow
    Write-Host "  $targetPath" -ForegroundColor DarkGray

    $headers = @{}
    if ($env:HF_TOKEN) {
        $headers['Authorization'] = "Bearer $env:HF_TOKEN"
    }

    $oldProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $url -OutFile $partialPath -Headers $headers -UseBasicParsing
        Move-Item -Path $partialPath -Destination $targetPath -Force
    } catch {
        Remove-Item $partialPath -Force -ErrorAction SilentlyContinue
        throw
    } finally {
        $ProgressPreference = $oldProgress
    }

    return (Resolve-Path $targetPath).Path
}

if ($NoThink -and $EnableReasoning) {
    throw 'Choose either -NoThink or -EnableReasoning, not both.'
}

New-Item -ItemType Directory -Force -Path $logDir, $modelsDir | Out-Null

# -- Already running? ---------------------------------------------------------
if (Test-Path $pidFile) {
    $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
        Write-Host "llama-server is already running (PID $existingPid)." -ForegroundColor Yellow
        Write-Host "  scripts\status-server.ps1   # check status" -ForegroundColor Yellow
        Write-Host "  scripts\stop-server.ps1     # stop it" -ForegroundColor Yellow
        Write-Host ""
        & $statusScript
        return
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

# -- Ensure ik_llama.cpp is installed ----------------------------------------
$toolsDir = Join-Path $repoRoot 'tools\ik_llama.cpp'
$llamaServerExe = Find-LlamaServerExe -Root $toolsDir

if (-not $llamaServerExe) {
    Write-Host "ik_llama.cpp llama-server not found. Building ik_llama.cpp..." -ForegroundColor Cyan
    & (Join-Path $PSScriptRoot 'install-llama.ps1') | Out-Null
    $llamaServerExe = Find-LlamaServerExe -Root $toolsDir
    if (-not $llamaServerExe) {
        throw "Install failed: llama-server.exe missing under $toolsDir."
    }
}

$serverDir = Split-Path -Parent $llamaServerExe
$env:PATH = "$serverDir;$env:PATH"

# -- Load catalog -------------------------------------------------------------
. (Join-Path $PSScriptRoot 'models.ps1')

# -- Select model -------------------------------------------------------------
$keys = @($global:LlamaModelCatalog.Keys)
if ($keys.Count -ne 1) {
    throw "Expected exactly one model profile, found $($keys.Count)."
}

if (-not $Model) {
    $Model = $keys[0]
} elseif (-not $global:LlamaModelCatalog.Contains($Model)) {
    throw "Unsupported model '$Model'. The only supported model is '$($keys[0])'."
}

$profile = $global:LlamaModelCatalog[$Model]
$family  = $global:LlamaFamilyDefaults[$profile.Family]

$ctx = if ($ContextOverride) { $ContextOverride } else { $profile.Context }
$temp = if ($profile.ContainsKey('Temp')) { $profile.Temp } else { $family.Temp }
$topP = if ($profile.ContainsKey('TopP')) { $profile.TopP } else { $family.TopP }
$topK = if ($profile.ContainsKey('TopK')) { $profile.TopK } else { $family.TopK }
$minP = if ($profile.ContainsKey('MinP')) { $profile.MinP } else { $family.MinP }
$presencePenalty = if ($profile.ContainsKey('PresencePenalty')) { $profile.PresencePenalty } else { $family.PresencePenalty }
$repeatPenalty = if ($profile.ContainsKey('RepeatPenalty')) { $profile.RepeatPenalty } else { $family.RepeatPenalty }
$cacheTypeK = if ($profile.ContainsKey('CacheTypeK')) { $profile.CacheTypeK } else { 'q8_0' }
$cacheTypeV = if ($profile.ContainsKey('CacheTypeV')) { $profile.CacheTypeV } else { 'q8_0' }
$speculative = if ($profile.ContainsKey('Speculative')) { $profile.Speculative } else { $null }
$noMmproj = $profile.ContainsKey('NoMmproj') -and [bool]$profile.NoMmproj
$noMmprojOffload = $profile.ContainsKey('NoMmprojOffload') -and [bool]$profile.NoMmprojOffload
$batch = if ($profile.ContainsKey('Batch')) { $profile.Batch } else { $null }
$ubatch = if ($profile.ContainsKey('UBatch')) { $profile.UBatch } else { $null }
$threads = if ($profile.ContainsKey('Threads')) { $profile.Threads } else { $null }
$threadsBatch = if ($profile.ContainsKey('ThreadsBatch')) { $profile.ThreadsBatch } else { $null }
$threadsMtmd = if ($profile.ContainsKey('ThreadsMtmd')) { $profile.ThreadsMtmd } else { $null }
$gpuLayers = if ($profile.ContainsKey('GpuLayers')) { $profile.GpuLayers } else { 99 }
$splitMode = if ($profile.ContainsKey('SplitMode')) { $profile.SplitMode } else { $null }
$mainGpu = if ($profile.ContainsKey('MainGpu')) { $profile.MainGpu } else { $null }
$parallel = if ($profile.ContainsKey('Parallel')) { $profile.Parallel } else { 1 }
$predict = if ($profile.ContainsKey('Predict')) { $profile.Predict } else { $null }
$imageMinTokens = if ($profile.ContainsKey('ImageMinTokens')) { $profile.ImageMinTokens } else { $null }
$imageMaxTokens = if ($profile.ContainsKey('ImageMaxTokens')) { $profile.ImageMaxTokens } else { $null }

$reasoningMode = if ($profile.ContainsKey('Reasoning')) { [string]$profile.Reasoning } else { 'auto' }
if ($NoThink) {
    $reasoningMode = 'off'
} elseif ($EnableReasoning) {
    $reasoningMode = 'on'
}

$modelPath = Ensure-HuggingFaceFile -Repo $profile.HFRepo -File $profile.HFFile -Kind 'model'
$mmprojPath = $null
if ($profile.ContainsKey('MmprojHFRepo') -and $profile.ContainsKey('MmprojHFFile')) {
    $mmprojPath = Ensure-HuggingFaceFile -Repo $profile.MmprojHFRepo -File $profile.MmprojHFFile -Kind 'mmproj'
}

# -- Template kwargs ----------------------------------------------------------
$templateKwargs = [ordered]@{}
if ($profile.Family -eq 'qwen36') {
    $templateKwargs['tool_parser'] = 'qwen3_coder'
    if ($reasoningMode -eq 'on') {
        $templateKwargs['preserve_thinking'] = $true
    } elseif ($reasoningMode -eq 'off') {
        $templateKwargs['enable_thinking'] = $false
    }
}

if ($NoThink -and $profile.Family -ne 'qwen36') {
    $templateKwargs['enable_thinking'] = $false
}

if ($templateKwargs.Count -gt 0) {
    $kwargsJson = $templateKwargs | ConvertTo-Json -Compress
    $env:LLAMA_CHAT_TEMPLATE_KWARGS = $kwargsJson
} else {
    $kwargsJson = $null
    Remove-Item Env:\LLAMA_CHAT_TEMPLATE_KWARGS -ErrorAction SilentlyContinue
}

# -- Build llama-server arg vector -------------------------------------------
$llamaArgs = @(
    '--model',           $modelPath,
    '--alias',           $profile.Alias,
    '--host',            $ListenHost,
    '--port',            $Port,
    '--ctx-size',        $ctx,
    '--gpu-layers',      $gpuLayers,
    '--flash-attn',      'on',
    '--cache-type-k',    $cacheTypeK,
    '--cache-type-v',    $cacheTypeV,
    '--jinja',
    '--temp',            $temp,
    '--top-p',           $topP,
    '--top-k',           $topK,
    '--min-p',           $minP,
    '--presence-penalty',$presencePenalty,
    '--repeat-penalty',  $repeatPenalty,
    '--parallel',        $parallel,
    '--metrics',
    '--verbose'
)

if ($threads) { $llamaArgs += @('--threads', $threads) }
if ($threadsBatch) { $llamaArgs += @('--threads-batch', $threadsBatch) }
if ($threadsMtmd) { $llamaArgs += @('--threads-mtmd', $threadsMtmd) }
if ($batch) { $llamaArgs += @('--batch-size', $batch) }
if ($ubatch) { $llamaArgs += @('--ubatch-size', $ubatch) }
if ($null -ne $predict) { $llamaArgs += @('--predict', $predict) }
if ($splitMode) { $llamaArgs += @('--split-mode', $splitMode) }
if ($null -ne $mainGpu) { $llamaArgs += @('--main-gpu', $mainGpu) }

if ($mmprojPath) {
    $llamaArgs += @('--mmproj', $mmprojPath)
    if ($noMmprojOffload) { $llamaArgs += @('--no-mmproj-offload') }
    if ($imageMinTokens) { $llamaArgs += @('--image-min-tokens', $imageMinTokens) }
    if ($imageMaxTokens) { $llamaArgs += @('--image-max-tokens', $imageMaxTokens) }
} elseif ($noMmproj) {
    $llamaArgs += @('--no-mmproj')
}

if ($profile.ExtraArgs) {
    $llamaArgs += $profile.ExtraArgs
}

if ($profile.Family -eq 'qwen36') {
    if (-not (Test-Path $qwenTemplateFile)) {
        throw "Missing Qwen chat template file: $qwenTemplateFile"
    }
    $llamaArgs += @('--chat-template-file', $qwenTemplateFile)
}

if ($reasoningMode -eq 'on') {
    $reasoningFormat = if ($profile.ContainsKey('ReasoningFormat')) { $profile.ReasoningFormat } else { 'deepseek' }
    $llamaArgs += @('--reasoning', 'on', '--reasoning-format', $reasoningFormat)
} elseif ($reasoningMode -eq 'off') {
    $llamaArgs += @('--reasoning', 'off')
}

if ($kwargsJson) {
    $llamaArgs += @('--chat-template-kwargs', $kwargsJson)
}

Write-Host ""
Write-Host "Starting ik_llama.cpp llama-server..." -ForegroundColor Cyan
Write-Host "  Backend     : ik_llama.cpp"
Write-Host "  Binary      : $llamaServerExe"
Write-Host "  Model       : $($profile.Name)"
Write-Host "  Model file  : $modelPath  ($($profile.Quant), $($profile.Size))"
Write-Host "  Alias       : $($profile.Alias)"
Write-Host "  Context     : $ctx (native max $($profile.MaxContext))"
Write-Host "  GPU layers  : $gpuLayers"
Write-Host "  KV cache    : K $cacheTypeK, V $cacheTypeV"
if ($batch -or $ubatch) {
    Write-Host "  Batch       : n_batch $batch, n_ubatch $ubatch"
}
if ($threads -or $threadsBatch -or $threadsMtmd) {
    Write-Host "  Threads     : main $threads, batch $threadsBatch, mmproj $threadsMtmd"
}
if ($mmprojPath) {
    $visionMode = if ($noMmprojOffload) { 'CPU mmproj (--no-mmproj-offload)' } else { 'mmproj enabled' }
    Write-Host "  Vision      : $visionMode"
    Write-Host "  mmproj      : $mmprojPath"
} elseif ($noMmproj) {
    Write-Host "  Vision      : mmproj disabled"
}
if ($speculative) {
    Write-Host "  Speculative : $speculative"
}
Write-Host "  Reasoning   : $reasoningMode"
if ($kwargsJson) {
    Write-Host "  Template    : $kwargsJson"
}
Write-Host "  Model cache : $modelsDir"
Write-Host "  Listen      : http://$ListenHost`:$Port  (OpenAI-compatible /v1)"
Write-Host "  Log         : $logFile"

# Truncate previous log
Set-Content -Path $logFile -Value '' -Encoding ascii
Set-Content -Path "$logFile.err" -Value '' -Encoding ascii

$llamaArgumentLine = ($llamaArgs | ForEach-Object { ConvertTo-WindowsCommandLineArgument ([string]$_) }) -join ' '

$proc = Start-Process -FilePath $llamaServerExe `
    -ArgumentList $llamaArgumentLine `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError  "$logFile.err" `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Out-File -FilePath $pidFile -Encoding ascii -NoNewline

# Persist info for status-server.ps1
$info = [ordered]@{
    Pid              = $proc.Id
    Backend          = 'ik_llama.cpp'
    LlamaServerPath  = $llamaServerExe
    Model            = $Model
    ModelName        = $profile.Name
    Alias            = $profile.Alias
    Family           = $profile.Family
    HFRepo           = $profile.HFRepo
    HFFile           = $profile.HFFile
    ModelPath        = $modelPath
    MmprojPath       = $mmprojPath
    Quant            = $profile.Quant
    Size             = $profile.Size
    Context          = $ctx
    MaxContext       = $profile.MaxContext
    Host             = $ListenHost
    Port             = $Port
    BaseUrl          = "http://$ListenHost`:$Port/v1"
    StartedAt        = (Get-Date).ToString('o')
    NoThink          = [bool]$NoThink
    Reasoning        = $reasoningMode
    TemplateKwargs   = $kwargsJson
    GpuLayers        = $gpuLayers
    FlashAttention   = 'on'
    CacheTypeK       = $cacheTypeK
    CacheTypeV       = $cacheTypeV
    Speculative      = $speculative
    VerboseLogging   = $true
    NoMmproj         = $noMmproj
    NoMmprojOffload  = $noMmprojOffload
    Batch            = $batch
    UBatch           = $ubatch
    Threads          = $threads
    ThreadsBatch     = $threadsBatch
    ThreadsMtmd      = $threadsMtmd
    Parallel         = $parallel
    Temp             = $temp
    TopP             = $topP
    TopK             = $topK
    MinP             = $minP
    PresencePenalty  = $presencePenalty
    RepeatPenalty    = $repeatPenalty
}
$info | ConvertTo-Json | Set-Content -Path $infoFile -Encoding ascii

Write-Host ""
Write-Host "Started (PID $($proc.Id))." -ForegroundColor Green
Write-Host ""
Write-Host "VS Code Copilot Chat (BYOK):" -ForegroundColor Green
Write-Host "  Open settings.json and add to github.copilot.chat.customOAIModels:"
Write-Host @"
    {
      "name": "$($profile.Alias) (local)",
      "url":  "http://$ListenHost`:$Port/v1",
      "apiKey": "sk-no-key-required",
      "modelId": "$($profile.Alias)"
    }
"@ -ForegroundColor DarkGray
Write-Host ""
Write-Host "Tail log : Get-Content -Wait '$logFile'"
Write-Host "Status   : scripts\status-server.ps1"
Write-Host "Stop     : scripts\stop-server.ps1"

$startupResult = Wait-LlamaServerStartupStatus -Process $proc -LogFile $logFile -ErrorLogFile "$logFile.err" -TimeoutSec 60

Write-Host ""
if ($startupResult -eq 'ready') {
    Write-Host "Startup wait finished: server is ready." -ForegroundColor Green
} elseif ($startupResult -eq 'failed') {
    Write-Host "Startup wait finished: llama-server reported a load failure." -ForegroundColor Red
} elseif ($startupResult -eq 'exited') {
    Write-Host "Startup wait finished: llama-server exited before it became ready." -ForegroundColor Red
} else {
    Write-Host "Startup wait finished: 60 seconds elapsed; server may still be loading." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Full status:" -ForegroundColor Cyan
& $statusScript