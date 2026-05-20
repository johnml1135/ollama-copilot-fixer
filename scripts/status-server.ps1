<#
.SYNOPSIS
    Reports whether llama-server (started via start-server.ps1) is running,
    which model it is serving, its load/runtime details, and whether its HTTP
    endpoint responds.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$logDir   = Join-Path $repoRoot 'logs'
$logFile  = Join-Path $logDir   'llama-server.log'
$pidFile  = Join-Path $logDir   'llama-server.pid'
$infoFile = Join-Path $logDir   'llama-server.info.json'

. (Join-Path $PSScriptRoot 'server-status.ps1')

function Test-LlamaInfoProperty {
    param(
        [pscustomobject]$Info,
        [string]$Name
    )

    if (-not $Info -or -not ($Info.PSObject.Properties.Name -contains $Name)) { return $false }
    $value = $Info.PSObject.Properties[$Name].Value
    if ($null -eq $value) { return $false }
    if ($value -is [string] -and $value -eq '') { return $false }
    return $true
}

function Write-LlamaModelInfo {
    param([pscustomobject]$Info)

    if (-not $Info) { return }

    Write-Host ""
    Write-Host "Model:" -ForegroundColor Cyan
    if (Test-LlamaInfoProperty $Info 'ModelName') { Write-Host ("  Name        : {0}" -f $Info.ModelName) }
    Write-Host ("  Model key   : {0}" -f $Info.Model)
    Write-Host ("  Alias       : {0}" -f $Info.Alias)
    if (Test-LlamaInfoProperty $Info 'Family') { Write-Host ("  Family      : {0}" -f $Info.Family) }
    Write-Host ("  HF repo     : {0}" -f $Info.HFRepo)
    if (Test-LlamaInfoProperty $Info 'HFFile') {
        $fileLine = "  HF file     : $($Info.HFFile)"
        if (Test-LlamaInfoProperty $Info 'Quant') { $fileLine += "  ($($Info.Quant))" }
        if (Test-LlamaInfoProperty $Info 'Size') { $fileLine += "  weights $($Info.Size)" }
        Write-Host $fileLine
    } elseif (Test-LlamaInfoProperty $Info 'Quant') {
        Write-Host ("  Quant       : {0}" -f $Info.Quant)
    }
    if (Test-LlamaInfoProperty $Info 'BaseUrl') { Write-Host ("  Endpoint    : {0}" -f $Info.BaseUrl) }
    if (Test-LlamaInfoProperty $Info 'NoThink') { Write-Host ("  NoThink     : {0}" -f $Info.NoThink) }
    if (Test-LlamaInfoProperty $Info 'Reasoning') { Write-Host ("  Reasoning   : {0}" -f $Info.Reasoning) }
    if (Test-LlamaInfoProperty $Info 'CacheTypeK') { Write-Host ("  KV cache    : K {0}, V {1}" -f $Info.CacheTypeK, $Info.CacheTypeV) }
    if (Test-LlamaInfoProperty $Info 'Batch') { Write-Host ("  Batch       : n_batch {0}, n_ubatch {1}" -f $Info.Batch, $Info.UBatch) }
    if (Test-LlamaInfoProperty $Info 'NoMmprojOffload' -and $Info.NoMmprojOffload) {
        Write-Host "  Vision      : CPU mmproj (--no-mmproj-offload)"
        if (Test-LlamaInfoProperty $Info 'MmprojPath') { Write-Host ("  mmproj      : {0}" -f $Info.MmprojPath) }
    } elseif (Test-LlamaInfoProperty $Info 'NoMmproj' -and $Info.NoMmproj) {
        Write-Host "  Vision      : mmproj disabled"
    }
    if (Test-LlamaInfoProperty $Info 'Speculative') { Write-Host ("  Speculative : {0}" -f $Info.Speculative) }
    if (Test-LlamaInfoProperty $Info 'TemplateKwargs') { Write-Host ("  Template    : {0}" -f $Info.TemplateKwargs) }
    if (Test-LlamaInfoProperty $Info 'Temp') {
        Write-Host ("  Sampler     : temp {0}, top_p {1}, top_k {2}, min_p {3}, repeat {4}, presence {5}" -f $Info.Temp, $Info.TopP, $Info.TopK, $Info.MinP, $Info.RepeatPenalty, $Info.PresencePenalty)
    }
    if (Test-LlamaInfoProperty $Info 'Parallel') { Write-Host ("  Parallel    : {0}" -f $Info.Parallel) }
}

$info = $null
if (Test-Path $infoFile) {
    $info = Get-Content $infoFile -Raw | ConvertFrom-Json
}

$snapshot = Get-LlamaServerLogSnapshot -LogFile $logFile -ErrorLogFile "$logFile.err"

if (-not (Test-Path $pidFile)) {
    Write-Host "llama-server: NOT RUNNING (no PID file)." -ForegroundColor Yellow
    Write-LlamaModelInfo -Info $info
    Write-LlamaServerLogStatus -Snapshot $snapshot -Info $info
    return
}

$serverPid = (Get-Content $pidFile | Select-Object -First 1).Trim()
$proc = Get-Process -Id $serverPid -ErrorAction SilentlyContinue

if (-not $proc) {
    Write-Host "llama-server: NOT RUNNING (stale PID $serverPid)." -ForegroundColor Yellow
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    Write-LlamaModelInfo -Info $info
    Write-LlamaServerLogStatus -Snapshot $snapshot -Info $info
    return
}

Write-Host "llama-server: RUNNING" -ForegroundColor Green
Write-Host ("  PID         : {0}" -f $proc.Id)
Write-Host ("  Started     : {0:yyyy-MM-dd HH:mm:ss}" -f $proc.StartTime)
$mem = [math]::Round($proc.WorkingSet64 / 1GB, 2)
Write-Host ("  RAM (RSS)   : {0} GB" -f $mem)

Write-LlamaModelInfo -Info $info
Write-LlamaServerLogStatus -Snapshot $snapshot -Info $info

# -- Probe HTTP endpoint ------------------------------------------------------
$baseUrl = $null
if ($info -and (Test-LlamaInfoProperty $info 'Host') -and (Test-LlamaInfoProperty $info 'Port')) {
    $baseUrl = "http://$($info.Host):$($info.Port)"
} elseif ($snapshot -and $snapshot.ListenUrl) {
    $baseUrl = $snapshot.ListenUrl.TrimEnd('/')
} elseif ($info -and (Test-LlamaInfoProperty $info 'BaseUrl')) {
    $baseUrl = $info.BaseUrl.TrimEnd('/')
    if ($baseUrl.EndsWith('/v1')) {
        $baseUrl = $baseUrl.Substring(0, $baseUrl.Length - 3)
    }
}

if ($baseUrl) {
    Write-Host ""
    Write-Host "HTTP:" -ForegroundColor Cyan
    $health = "$baseUrl/health"
    $models = "$baseUrl/v1/models"
    try {
        $h = Invoke-RestMethod -Uri $health -TimeoutSec 3
        $status = if ($h.status) { $h.status } else { 'ok' }
        Write-Host ("  HTTP /health: {0}" -f $status) -ForegroundColor Green
    } catch {
        Write-Host ("  HTTP /health: UNREACHABLE ({0})" -f $_.Exception.Message) -ForegroundColor Yellow
        Write-Host "                (server may still be loading the model — check the log)" -ForegroundColor DarkGray
        return
    }
    try {
        $m = Invoke-RestMethod -Uri $models -TimeoutSec 3
        $names = ($m.data | ForEach-Object { $_.id }) -join ', '
        Write-Host ("  /v1/models  : {0}" -f $names)
    } catch {
        # not fatal
    }

    $metricsText = $null
    $slots = @()
    try {
        $metricsResponse = Invoke-WebRequest -Uri "$baseUrl/metrics" -TimeoutSec 3
        $metricsText = $metricsResponse.Content
    } catch {
        # not fatal
    }
    try {
        $slots = @(Invoke-RestMethod -Uri "$baseUrl/slots" -TimeoutSec 3)
    } catch {
        # not fatal
    }

    $telemetry = Get-LlamaEndpointTelemetrySnapshot -MetricsText $metricsText -Slots $slots
    Write-LlamaEndpointTelemetryStatus -Telemetry $telemetry
}

# -- nvidia-smi snapshot if available ----------------------------------------
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    Write-Host ""
    Write-Host "nvidia-smi:" -ForegroundColor Cyan
    & nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    if ($proc) {
        $gpuApps = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null
        $serverGpuApps = @($gpuApps | Where-Object { $_ -match "^\s*$($proc.Id)\s*," })
        if ($serverGpuApps.Count -gt 0) {
            Write-Host "  llama-server GPU process:" -ForegroundColor DarkGray
            $serverGpuApps | ForEach-Object { Write-Host ("  {0}" -f $_) }
        }
    }
}
