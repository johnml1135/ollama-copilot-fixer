<#
.SYNOPSIS
    Smoke tests for llama-server status parsing helpers.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $repoRoot 'scripts\server-status.ps1')

function Assert-Equal {
    param(
        [object]$Actual,
        [object]$Expected,
        [string]$Name
    )

    if ($Actual -ne $Expected) {
        throw "${Name}: expected '$Expected', got '$Actual'"
    }
}

function Assert-MatchText {
    param(
        [string]$Actual,
        [string]$Pattern,
        [string]$Name
    )

    if ($Actual -notmatch $Pattern) {
        throw "${Name}: expected text to match /$Pattern/, got '$Actual'"
    }
}

Assert-Equal (Format-LlamaMiB 0) '0.00 MiB' 'zero MiB formatting'
Assert-Equal (Format-LlamaNumber 0) '0' 'zero number formatting'

$readyLog = @'
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24575 MiB):
  Device 0: NVIDIA GeForce RTX 3090, compute capability 8.6, VMM: yes, VRAM: 24575 MiB
load_backend: loaded CUDA backend from C:\repo\tools\llama.cpp\ggml-cuda.dll
load_backend: loaded CPU backend from C:\repo\tools\llama.cpp\ggml-cpu-alderlake.dll
build_info: b8946-0f1bb602d
system_info: n_threads = 12 (n_threads_batch = 12) / 20 | CUDA : USE_GRAPHS = 1 | CPU : AVX2 = 1 |
main: loading model
common_params_fit_impl: projected to use 21483 MiB of device memory vs. 23154 MiB of free device memory
common_params_fit_impl: will leave 1670 >= 1024 MiB of free device memory, no changes needed
load_tensors: offloading output layer to GPU
load_tensors: offloading 63 repeating layers to GPU
load_tensors: offloaded 65/65 layers to GPU
load_tensors:   CPU_Mapped model buffer size =   682.03 MiB
load_tensors:        CUDA0 model buffer size = 14032.23 MiB
llama_context: n_ctx_train           = 262144
llama_context: n_ctx         = 200192
llama_context: n_batch       = 2048
llama_context: n_ubatch      = 512
llama_context: flash_attn    = enabled
llama_kv_cache:      CUDA0 KV buffer size =  6647.00 MiB
llama_kv_cache: size = 6647.00 MiB (200192 cells,  16 layers,  1/1 seqs), K (q8_0): 3323.50 MiB, V (q8_0): 3323.50 MiB
llama_memory_recurrent:      CUDA0 RS buffer size =   149.62 MiB
sched_reserve:      CUDA0 compute buffer size =   654.78 MiB
sched_reserve:  CUDA_Host compute buffer size =   411.29 MiB
srv          init: init: chat template, thinking = 0
main: model loaded
main: server is listening on http://127.0.0.1:8080
'@

$ready = Get-LlamaServerLogSnapshot -Text $readyLog
Assert-Equal $ready.LoadState 'ready' 'ready LoadState'
Assert-Equal $ready.CudaDeviceCount 1 'CUDA device count'
Assert-Equal $ready.CudaDevices[0].Name 'NVIDIA GeForce RTX 3090' 'CUDA device name'
Assert-Equal $ready.OffloadedLayers 65 'offloaded layers'
Assert-Equal $ready.TotalLayers 65 'total layers'
Assert-Equal $ready.CpuMappedModelMiB 682.03 'CPU mapped model buffer'
Assert-Equal $ready.GpuModelBuffers[0].MiB 14032.23 'CUDA model buffer'
Assert-Equal $ready.ContextAllocated 200192 'allocated context'
Assert-Equal $ready.NativeContext 262144 'native context'
Assert-Equal $ready.KvCacheMiB 6647.00 'KV cache total'
Assert-Equal $ready.KvCacheLayers 16 'KV cache layers'
Assert-Equal $ready.KvCacheKType 'q8_0' 'KV K type'
Assert-Equal $ready.KvCacheVType 'q8_0' 'KV V type'
Assert-Equal $ready.ChatTemplateThinking 0 'chat template thinking'

$brief = Format-LlamaServerBriefStatus -Snapshot $ready
Assert-MatchText $brief 'ready' 'brief readiness'
Assert-MatchText $brief '65/65 layers on GPU' 'brief offload split'
Assert-MatchText $brief 'ctx 200,192' 'brief context'

$ikReadyLog = @'
INFO [                    main] model loaded | tid="508" timestamp=1779282646
INFO [                    main] HTTP server listening | tid="508" timestamp=1779282646 hostname="127.0.0.1" port="8094" n_threads_http="19"
VERB [              start_loop] wait for new task | tid="508" timestamp=1779282646
'@
$ikReady = Get-LlamaServerLogSnapshot -Text $ikReadyLog
Assert-Equal $ikReady.LoadState 'ready' 'ik ready LoadState'
Assert-Equal $ikReady.ListenUrl 'http://127.0.0.1:8094' 'ik listen URL'

$loading = Get-LlamaServerLogSnapshot -Text @'
main: loading model
load_tensors: loading model tensors, this can take a while... (mmap = true, direct_io = false)
'@
Assert-Equal $loading.LoadState 'loading' 'loading LoadState'

$failed = Get-LlamaServerLogSnapshot -Text @'
main: loading model
error loading model: out of memory
'@
Assert-Equal $failed.LoadState 'failed' 'failed LoadState'
Assert-MatchText $failed.FailureLine 'out of memory' 'failure line'

$mtpLog = @'
llama_context: n_ctx         = 262144
llama_kv_cache:      CUDA0 KV buffer size =  5120.00 MiB
llama_kv_cache: size = 5120.00 MiB (262144 cells,  16 layers,  1/1 seqs), K (q4_1): 2560.00 MiB, V (q4_1): 2560.00 MiB
srv    load_model: creating MTP draft context against the target model
llama_kv_cache:      CUDA0 KV buffer size =   320.00 MiB
llama_kv_cache: size =  320.00 MiB (262144 cells,   1 layers,  1/1 seqs), K (q4_1):  160.00 MiB, V (q4_1):  160.00 MiB
common_speculative_init: adding speculative implementation 'draft-mtp'
main: server is listening on http://127.0.0.1:8080
'@
$mtp = Get-LlamaServerLogSnapshot -Text $mtpLog
Assert-Equal $mtp.LoadState 'ready' 'MTP LoadState'
Assert-Equal $mtp.KvCacheMiB 5120.00 'MTP primary KV cache'
Assert-Equal $mtp.KvCacheTotalMiB 5440.00 'MTP total KV cache'
Assert-Equal @($mtp.KvCacheDetails).Count 2 'MTP KV cache detail count'
Assert-Equal $mtp.KvCacheLayers 16 'MTP primary KV layers'

$longLogFile = [System.IO.Path]::GetTempFileName()
try {
    $longLog = @()
    $longLog += 1..1300 | ForEach-Object { "download noise $_" }
    $longLog += 'common_params_fit_impl: projected to use 22210 MiB of device memory vs. 23154 MiB of free device memory'
    $longLog += 1..1000 | ForEach-Object { "load noise $_" }
    $longLog += 'main: server is listening on http://127.0.0.1:8080'
    Set-Content -Path $longLogFile -Value $longLog -Encoding ascii

    $longSnapshot = Get-LlamaServerLogSnapshot -LogFile $longLogFile
    Assert-Equal $longSnapshot.LoadState 'ready' 'long log LoadState'
    Assert-Equal $longSnapshot.ProjectedUseMiB 22210 'long log projected use retained'
    Assert-Equal $longSnapshot.ProjectedFreeMiB 23154 'long log projected free retained'
}
finally {
    Remove-Item $longLogFile -Force -ErrorAction SilentlyContinue
}

Write-Host 'server-status tests passed' -ForegroundColor Green