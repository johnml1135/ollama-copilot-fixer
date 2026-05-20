<#
.SYNOPSIS
    Installs local-Copilot minimal-context skills and LLM Gateway settings.

.DESCRIPTION
    Copies the skill bundle from this repository into either a target repository
    or the current user's Copilot skills profile, then merges recommended
    GitHub Copilot LLM Gateway settings into the matching VS Code settings file.

.PARAMETER Scope
    Install target: User or Repo. If omitted, an interactive menu is shown.

.PARAMETER RepoPath
    Repository folder to initialize when Scope is Repo. If omitted, the script
    prompts for a folder and defaults to this helper repository.

.PARAMETER ServerUrl
    Base URL for the local OpenAI-compatible server. Do not include /v1.

.PARAMETER Force
    Replace existing installed skill folders.

.EXAMPLE
    .\scripts\initialize-copilot-local-agent.ps1

.EXAMPLE
    .\scripts\initialize-copilot-local-agent.ps1 -Scope User

.EXAMPLE
    .\scripts\initialize-copilot-local-agent.ps1 -Scope Repo -RepoPath C:\src\my-repo -Force
#>

[CmdletBinding()]
param(
    [ValidateSet('User', 'Repo')]
    [string]$Scope,
    [string]$RepoPath,
    [string]$ServerUrl = 'http://127.0.0.1:8080',
    [int]$RequestTimeout = 600000,
    [int]$DefaultMaxTokens = 16384,
    [int]$DefaultMaxOutputTokens = 4096,
    [bool]$ParallelToolCalling = $false,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceSkillsDir = Join-Path $repoRoot '.github\skills'

function Read-InstallScope {
    Write-Host ''
    Write-Host 'Install local Copilot helpers to:' -ForegroundColor Cyan
    Write-Host '  [1] User profile  (~\.copilot\skills and VS Code User settings)'
    Write-Host '  [2] Repository    (.github\skills and .vscode\settings.json)'
    Write-Host ''

    while ($true) {
        $choice = Read-Host 'Choice [1-2]'
        switch ($choice) {
            '1' { return 'User' }
            '2' { return 'Repo' }
            default { Write-Host 'Enter 1 or 2.' -ForegroundColor Yellow }
        }
    }
}

function Get-VSCodeUserSettingsPath {
    if ($env:APPDATA) {
        $userDir = Join-Path $env:APPDATA 'Code\User'
    } elseif ($IsMacOS) {
        $userDir = Join-Path $HOME 'Library/Application Support/Code/User'
    } else {
        $userDir = Join-Path $HOME '.config/Code/User'
    }

    return Join-Path $userDir 'settings.json'
}

function Remove-JsonComments {
    param([string]$Text)

    $builder = [System.Text.StringBuilder]::new()
    $inString = $false
    $isEscaped = $false
    $inLineComment = $false
    $inBlockComment = $false

    for ($i = 0; $i -lt $Text.Length; $i++) {
        $char = $Text[$i]
        $next = if ($i + 1 -lt $Text.Length) { $Text[$i + 1] } else { [char]0 }

        if ($inLineComment) {
            if ($char -eq "`n") {
                $inLineComment = $false
                [void]$builder.Append($char)
            }
            continue
        }

        if ($inBlockComment) {
            if ($char -eq '*' -and $next -eq '/') {
                $inBlockComment = $false
                $i++
            }
            continue
        }

        if ($inString) {
            [void]$builder.Append($char)
            if ($isEscaped) {
                $isEscaped = $false
            } elseif ($char -eq '\') {
                $isEscaped = $true
            } elseif ($char -eq '"') {
                $inString = $false
            }
            continue
        }

        if ($char -eq '"') {
            $inString = $true
            [void]$builder.Append($char)
            continue
        }

        if ($char -eq '/' -and $next -eq '/') {
            $inLineComment = $true
            $i++
            continue
        }

        if ($char -eq '/' -and $next -eq '*') {
            $inBlockComment = $true
            $i++
            continue
        }

        [void]$builder.Append($char)
    }

    return $builder.ToString()
}

function ConvertTo-OrderedHashtable {
    param($Value)

    $result = [ordered]@{}
    if ($null -eq $Value) {
        return $result
    }

    foreach ($property in $Value.PSObject.Properties) {
        if ($property.Value -is [System.Management.Automation.PSCustomObject]) {
            $result[$property.Name] = ConvertTo-OrderedHashtable $property.Value
        } else {
            $result[$property.Name] = $property.Value
        }
    }

    return $result
}

function Read-SettingsJsonc {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return [ordered]@{}
    }

    $raw = Get-Content -Path $Path -Raw
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return [ordered]@{}
    }

    $json = (Remove-JsonComments $raw) -replace ',(\s*[}\]])', '$1'
    $parsed = $json | ConvertFrom-Json
    return ConvertTo-OrderedHashtable $parsed
}

function Write-SettingsJson {
    param(
        [string]$Path,
        [System.Collections.IDictionary]$Settings
    )

    $settingsDir = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $settingsDir | Out-Null

    $backupPath = $null
    if (Test-Path $Path) {
        $backupPath = "$Path.bak-$(Get-Date -Format 'yyyyMMddHHmmss')"
        Copy-Item -Path $Path -Destination $backupPath -Force
    }

    $Settings | ConvertTo-Json -Depth 20 | Set-Content -Path $Path -Encoding utf8

    if ($backupPath) {
        Write-Host "Backed up existing settings to $backupPath" -ForegroundColor DarkGray
    }
}

function Invoke-SkillBundleInstall {
    param([string]$TargetSkillsDir)

    if (-not (Test-Path $sourceSkillsDir)) {
        throw "Source skills not found at $sourceSkillsDir"
    }

    New-Item -ItemType Directory -Force -Path $TargetSkillsDir | Out-Null

    $sourceFull = (Resolve-Path $sourceSkillsDir).Path
    $targetFull = (Resolve-Path $TargetSkillsDir).Path
    if ($sourceFull -ieq $targetFull) {
        Write-Host "Skills already live at $TargetSkillsDir" -ForegroundColor Green
        return
    }

    foreach ($skill in Get-ChildItem -Path $sourceSkillsDir -Directory) {
        $dest = Join-Path $TargetSkillsDir $skill.Name
        if (Test-Path $dest) {
            if ($Force) {
                Remove-Item -Path $dest -Recurse -Force
            } else {
                Write-Host "Skipping existing skill $($skill.Name) (use -Force to replace)." -ForegroundColor Yellow
                continue
            }
        }

        Copy-Item -Path $skill.FullName -Destination $dest -Recurse
        Write-Host "Installed skill $($skill.Name)" -ForegroundColor Green
    }
}

if (-not $Scope) {
    $Scope = Read-InstallScope
}

if ($Scope -eq 'Repo') {
    if (-not $RepoPath) {
        $answer = Read-Host "Repository folder [$repoRoot]"
        $RepoPath = if ([string]::IsNullOrWhiteSpace($answer)) { $repoRoot } else { $answer.Trim('"') }
    }

    if (-not (Test-Path $RepoPath)) {
        throw "Repository folder not found: $RepoPath"
    }

    $targetRoot = (Resolve-Path $RepoPath).Path
    $targetSkillsDir = Join-Path $targetRoot '.github\skills'
    $settingsPath = Join-Path $targetRoot '.vscode\settings.json'
} else {
    $targetRoot = $HOME
    $targetSkillsDir = Join-Path $HOME '.copilot\skills'
    $settingsPath = Get-VSCodeUserSettingsPath
}

$settings = Read-SettingsJsonc $settingsPath
$normalizedServerUrl = $ServerUrl.TrimEnd('/')

$recommendedSettings = [ordered]@{
    'github.copilot.chat.agentDebugLog.fileLogging.enabled' = $true
    'github.copilot.llm-gateway.serverUrl' = $normalizedServerUrl
    'github.copilot.llm-gateway.requestTimeout' = $RequestTimeout
    'github.copilot.llm-gateway.defaultMaxTokens' = $DefaultMaxTokens
    'github.copilot.llm-gateway.defaultMaxOutputTokens' = $DefaultMaxOutputTokens
    'github.copilot.llm-gateway.enableToolCalling' = $true
    'github.copilot.llm-gateway.parallelToolCalling' = $ParallelToolCalling
    'github.copilot.llm-gateway.agentTemperature' = 0.0
}

foreach ($entry in $recommendedSettings.GetEnumerator()) {
    $settings[$entry.Key] = $entry.Value
}

Write-Host ''
Write-Host "Target scope : $Scope" -ForegroundColor Cyan
Write-Host "Skills path  : $targetSkillsDir"
Write-Host "Settings     : $settingsPath"
Write-Host "Server URL   : $normalizedServerUrl"
Write-Host "Max tokens   : $DefaultMaxTokens"
Write-Host "Parallel tools: $ParallelToolCalling"
Write-Host ''

Invoke-SkillBundleInstall $targetSkillsDir

Write-SettingsJson -Path $settingsPath -Settings $settings
Write-Host "Updated settings at $settingsPath" -ForegroundColor Green

Write-Host ''
Write-Host 'Next steps:' -ForegroundColor Cyan
Write-Host '  1. Install or enable GitHub Copilot LLM Gateway if needed.'
Write-Host '  2. Start llama-server with scripts\start-server.ps1.'
Write-Host '  3. Run "GitHub Copilot LLM Gateway: Refresh Models" from the Command Palette.'
Write-Host '  4. Select the local model in Copilot Chat and type / to see installed skills.'