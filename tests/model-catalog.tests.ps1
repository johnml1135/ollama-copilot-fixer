<#
.SYNOPSIS
    Smoke tests for curated model catalog entries.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $repoRoot 'scripts\models.ps1')

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Name
    )

    if (-not $Condition) { throw $Name }
}

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

function Assert-ContainsValue {
    param(
        [object[]]$Values,
        [object]$Expected,
        [string]$Name
    )

    if ($Expected -notin @($Values)) {
        throw "${Name}: expected list to contain '$Expected'"
    }
}

$expectedOrder = @(
    'qwen36-27b-mtp-iq4-ks'
)

$actualOrder = @($global:LlamaModelCatalog.Keys)
Assert-Equal $actualOrder.Count $expectedOrder.Count 'catalog profile count'
for ($index = 0; $index -lt $expectedOrder.Count; $index++) {
    Assert-Equal $actualOrder[$index] $expectedOrder[$index] "catalog order $index"
}

$qwenKeys = @($global:LlamaModelCatalog.Keys | Where-Object { $_ -like 'qwen36-*' })
Assert-Equal $qwenKeys.Count 1 'active Qwen profile count'

$profile = $global:LlamaModelCatalog['qwen36-27b-mtp-iq4-ks']
Assert-Equal $profile.HFRepo 'ubergarm/Qwen3.6-27B-GGUF' 'ik Qwen repo'
Assert-Equal $profile.HFFile 'Qwen3.6-27B-MTP-IQ4_KS.gguf' 'ik Qwen file'
Assert-Equal $profile.Quant 'IQ4_KS + MTP' 'ik Qwen quant'
Assert-Equal $profile.Alias 'qwen3.6-27b-mtp-iq4-ks' 'ik Qwen alias'
Assert-Equal $profile.Context 156000 'ik Qwen context'
Assert-Equal $profile.CacheTypeK 'q8_0' 'ik Qwen K cache type'
Assert-Equal $profile.CacheTypeV 'q8_0' 'ik Qwen V cache type'
Assert-Equal $profile.Batch 2048 'ik Qwen batch size'
Assert-Equal $profile.UBatch 512 'ik Qwen ubatch size'
Assert-Equal $profile.Threads 8 'ik Qwen CPU threads'
Assert-Equal $profile.ThreadsBatch 8 'ik Qwen batch CPU threads'
Assert-Equal $profile.ThreadsMtmd 8 'ik Qwen mmproj CPU threads'
Assert-Equal $profile.MmprojHFRepo 'unsloth/Qwen3.6-27B-MTP-GGUF' 'ik Qwen mmproj repo'
Assert-Equal $profile.MmprojHFFile 'mmproj-BF16.gguf' 'ik Qwen mmproj file'
Assert-Equal $profile.NoMmprojOffload $true 'ik Qwen keeps mmproj on CPU'
Assert-Equal $profile.Reasoning 'on' 'ik Qwen reasoning default'
Assert-ContainsValue $profile.ExtraArgs '--multi-token-prediction' 'ik Qwen uses built-in MTP'
Assert-ContainsValue $profile.ExtraArgs '--draft-max' 'ik Qwen sets draft max'
Assert-ContainsValue $profile.ExtraArgs '4' 'ik Qwen draft max value'
Assert-ContainsValue $profile.ExtraArgs '--draft-p-min' 'ik Qwen sets draft p-min'
Assert-ContainsValue $profile.ExtraArgs '0.0' 'ik Qwen draft p-min value'
Assert-ContainsValue $profile.ExtraArgs '--merge-qkv' 'ik Qwen merges qkv'
Assert-ContainsValue $profile.ExtraArgs '--merge-up-gate-experts' 'ik Qwen merges up/gate experts'
Assert-ContainsValue $profile.ExtraArgs '--cache-ram' 'ik Qwen enables RAM cache'
Assert-ContainsValue $profile.ExtraArgs '32768' 'ik Qwen RAM cache size'
Assert-ContainsValue $profile.ExtraArgs '--ctx-checkpoints' 'ik Qwen enables context checkpoints'
Assert-ContainsValue $profile.ExtraArgs '32' 'ik Qwen context checkpoint count'

Write-Host 'model-catalog tests passed' -ForegroundColor Green