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
    'qwen36-27b-mtp-q5',
    'qwen36-27b-mtp-quality-max',
    'qwen36-27b-q5',
    'qwen36-27b-quality-max',
    'gemma4-26b-a4b',
    'gemma4-31b'
)

$actualOrder = @($global:LlamaModelCatalog.Keys)
Assert-Equal $actualOrder.Count $expectedOrder.Count 'catalog profile count'
for ($index = 0; $index -lt $expectedOrder.Count; $index++) {
    Assert-Equal $actualOrder[$index] $expectedOrder[$index] "catalog order $index"
}

$qwenKeys = @($global:LlamaModelCatalog.Keys | Where-Object { $_ -like 'qwen36-*' })
Assert-Equal $qwenKeys.Count 4 'active Qwen profile count'

$mtpQ5 = $global:LlamaModelCatalog['qwen36-27b-mtp-q5']
Assert-Equal $mtpQ5.HFRepo 'unsloth/Qwen3.6-27B-MTP-GGUF' 'MTP Q5 repo'
Assert-Equal $mtpQ5.HFFile 'Qwen3.6-27B-Q5_K_M.gguf' 'MTP Q5 file'
Assert-Equal $mtpQ5.Quant 'Q5_K_M + MTP' 'MTP Q5 quant'
Assert-Equal $mtpQ5.Context 160000 'MTP Q5 context'
Assert-Equal $mtpQ5.CacheTypeK 'q4_1' 'MTP Q5 K cache type'
Assert-Equal $mtpQ5.CacheTypeV 'q4_1' 'MTP Q5 V cache type'
Assert-Equal $mtpQ5.NoMmproj $true 'MTP Q5 disables mmproj'
Assert-ContainsValue $mtpQ5.ExtraArgs 'draft-mtp' 'MTP Q5 uses draft-mtp'
Assert-ContainsValue $mtpQ5.ExtraArgs '2' 'MTP Q5 uses conservative draft limit'

$mtpQualityMax = $global:LlamaModelCatalog['qwen36-27b-mtp-quality-max']
Assert-Equal $mtpQualityMax.HFRepo 'unsloth/Qwen3.6-27B-MTP-GGUF' 'MTP quality max repo'
Assert-Equal $mtpQualityMax.HFFile 'Qwen3.6-27B-UD-Q4_K_XL.gguf' 'MTP quality max file'
Assert-Equal $mtpQualityMax.Quant 'UD-Q4_K_XL + MTP' 'MTP quality max quant'
Assert-Equal $mtpQualityMax.Context 245760 'MTP quality max context'
Assert-Equal $mtpQualityMax.CacheTypeK 'q4_1' 'MTP quality max K cache type'
Assert-Equal $mtpQualityMax.CacheTypeV 'q4_1' 'MTP quality max V cache type'
Assert-Equal $mtpQualityMax.NoMmproj $true 'MTP quality max disables mmproj'
Assert-ContainsValue $mtpQualityMax.ExtraArgs 'draft-mtp' 'MTP quality max uses draft-mtp'
Assert-ContainsValue $mtpQualityMax.ExtraArgs '2' 'MTP quality max uses conservative draft limit'

$qualityMax = $global:LlamaModelCatalog['qwen36-27b-quality-max']
Assert-Equal $qualityMax.HFRepo 'unsloth/Qwen3.6-27B-GGUF' 'quality max Qwen repo'
Assert-Equal $qualityMax.HFFile 'Qwen3.6-27B-UD-Q4_K_XL.gguf' 'quality max Qwen file'
Assert-Equal $qualityMax.Quant 'UD-Q4_K_XL' 'quality max Qwen quant'
Assert-Equal $qualityMax.Context 262144 'quality max Qwen context'
Assert-Equal $qualityMax.CacheTypeK 'q4_1' 'quality max Qwen K cache type'
Assert-Equal $qualityMax.CacheTypeV 'q4_1' 'quality max Qwen V cache type'
Assert-Equal $qualityMax.NoMmproj $true 'quality max Qwen disables mmproj'

$q5 = $global:LlamaModelCatalog['qwen36-27b-q5']
Assert-Equal $q5.HFRepo 'unsloth/Qwen3.6-27B-GGUF' 'Q5 Qwen repo'
Assert-Equal $q5.HFFile 'Qwen3.6-27B-Q5_K_M.gguf' 'Q5 Qwen file'
Assert-Equal $q5.Quant 'Q5_K_M' 'Q5 Qwen quant'
Assert-Equal $q5.Context 200000 'Q5 Qwen context'
Assert-Equal $q5.CacheTypeK 'q4_1' 'Q5 Qwen K cache type'
Assert-Equal $q5.CacheTypeV 'q4_1' 'Q5 Qwen V cache type'
Assert-Equal $q5.NoMmproj $true 'Q5 Qwen disables mmproj'

Write-Host 'model-catalog tests passed' -ForegroundColor Green