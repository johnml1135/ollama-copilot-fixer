<#
.SYNOPSIS
    Summarizes Copilot Chat request sizes and llama-server prompt token counts.

.DESCRIPTION
    Reads Copilot Chat debug logs and llama-server stderr to help identify
    prompt/context bloat when using a local OpenAI-compatible model provider.

.PARAMETER LogDir
    Optional Copilot debug log directory. If omitted, the newest directory with
    a main.jsonl under VS Code workspaceStorage is used.

.PARAMETER LlamaLog
    Optional llama-server stderr log. Defaults to logs\llama-server.log.err.

.EXAMPLE
    .\scripts\inspect-copilot-context.ps1

.EXAMPLE
    .\scripts\inspect-copilot-context.ps1 -LogDir 'C:\path\to\debug-log-dir'
#>

[CmdletBinding()]
param(
    [string]$LogDir,
    [string]$LlamaLog
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $LlamaLog) {
    $LlamaLog = Join-Path $repoRoot 'logs\llama-server.log.err'
}

if (-not $LogDir) {
    $storageRoot = Join-Path $env:APPDATA 'Code\User\workspaceStorage'
    $mainLog = Get-ChildItem $storageRoot -Recurse -Filter 'main.jsonl' -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -like '*GitHub.copilot-chat*debug-logs*' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $mainLog) {
        throw "No Copilot Chat debug log found under $storageRoot. Enable github.copilot.chat.agentDebugLog.fileLogging.enabled and try again."
    }
    $LogDir = Split-Path -Parent $mainLog.FullName
}

$mainJsonl = Join-Path $LogDir 'main.jsonl'
if (-not (Test-Path $mainJsonl)) {
    throw "main.jsonl not found in $LogDir"
}

Write-Host "Copilot log : $LogDir" -ForegroundColor Cyan
Write-Host "llama log   : $LlamaLog" -ForegroundColor Cyan
Write-Host ""

Write-Host "Log files" -ForegroundColor Cyan
Get-ChildItem $LogDir |
    Sort-Object Length -Descending |
    Select-Object Name, @{Name='KB'; Expression={ [math]::Round($_.Length / 1KB, 1) }}, LastWriteTime |
    Format-Table -AutoSize

Write-Host "Prompt sidecars" -ForegroundColor Cyan
Get-ChildItem $LogDir -File |
    Where-Object { $_.Name -match '^(system_prompt_\d+|tools_\d+|models)\.json$' } |
    Sort-Object Length -Descending |
    ForEach-Object {
        [pscustomobject]@{
            Name         = $_.Name
            KB           = [math]::Round($_.Length / 1KB, 1)
            ApproxTokens = [math]::Round($_.Length / 4)
            LastWrite    = $_.LastWriteTime
        }
    } |
    Format-Table -AutoSize

$largestSystemPrompt = Get-ChildItem $LogDir -Filter 'system_prompt_*.json' -File |
    Sort-Object Length -Descending |
    Select-Object -First 1

if ($largestSystemPrompt) {
    try {
        $systemPromptJson = Get-Content $largestSystemPrompt.FullName -Raw | ConvertFrom-Json
        $systemPromptText = ($systemPromptJson.content -as [string]) -replace '\\n', "`n"
        $skillsSection = [regex]::Match($systemPromptText, '(?s)<skills>.*?</skills>').Value

        Write-Host "Largest system prompt summary" -ForegroundColor Cyan
        [pscustomobject]@{
            File               = $largestSystemPrompt.Name
            Chars              = $systemPromptText.Length
            ApproxTokens       = [math]::Round($systemPromptText.Length / 4)
            SkillCount         = ([regex]::Matches($skillsSection, '<skill>')).Count
            SkillSectionKB     = [math]::Round($skillsSection.Length / 1KB, 1)
            SkillSectionTokens = [math]::Round($skillsSection.Length / 4)
        } |
            Format-List
    } catch {
        Write-Host "Could not parse $($largestSystemPrompt.Name): $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

$latestTools = Get-ChildItem $LogDir -Filter 'tools_*.json' -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($latestTools) {
    try {
        $toolsOuter = Get-Content $latestTools.FullName -Raw | ConvertFrom-Json
        $toolsContent = $toolsOuter.content -as [string]
        $tools = $toolsContent | ConvertFrom-Json

        Write-Host "Tool schema summary" -ForegroundColor Cyan
        [pscustomobject]@{
            File         = $latestTools.Name
            ToolCount    = $tools.Count
            ContentKB    = [math]::Round($toolsContent.Length / 1KB, 1)
            ApproxTokens = [math]::Round($toolsContent.Length / 4)
        } |
            Format-List

        Write-Host "Largest tool schemas" -ForegroundColor Cyan
        $tools |
            ForEach-Object {
                $toolJson = $_ | ConvertTo-Json -Depth 100 -Compress
                [pscustomobject]@{
                    Name = $_.name
                    KB   = [math]::Round($toolJson.Length / 1KB, 1)
                }
            } |
            Sort-Object KB -Descending |
            Select-Object -First 15 |
            Format-Table -AutoSize
    } catch {
        Write-Host "Could not parse $($latestTools.Name): $($_.Exception.Message)" -ForegroundColor Yellow
    }
}

$events = Get-Content $mainJsonl | Where-Object { $_ } | ForEach-Object { $_ | ConvertFrom-Json }

Write-Host "LLM requests" -ForegroundColor Cyan
$events |
    Where-Object { $_.type -eq 'llm_request' } |
    ForEach-Object {
        $attrs = $_.attrs
        [pscustomobject]@{
            Time         = ([datetimeoffset]::FromUnixTimeMilliseconds([int64]$_.ts)).LocalDateTime.ToString('HH:mm:ss')
            Model        = $attrs.model
            InputTokens  = $attrs.inputTokens
            MaxTokens    = $attrs.maxTokens
            OutputTokens = $attrs.outputTokens
            DurationSec  = [math]::Round($_.dur / 1000, 1)
            ToolsFile    = $attrs.toolsFile
            User         = (($attrs.userRequest -as [string]) -replace '\s+', ' ').Substring(0, [math]::Min(80, (($attrs.userRequest -as [string]) -replace '\s+', ' ').Length))
        }
    } |
    Format-Table -AutoSize

Write-Host "Largest tool results" -ForegroundColor Cyan
$events |
    Where-Object { $_.type -eq 'tool_call' } |
    ForEach-Object {
        $attrs = $_.attrs
        $result = $attrs.result -as [string]
        [pscustomobject]@{
            Time     = ([datetimeoffset]::FromUnixTimeMilliseconds([int64]$_.ts)).LocalDateTime.ToString('HH:mm:ss')
            Tool     = $_.name
            Status   = $_.status
            ResultKB = [math]::Round(($result.Length) / 1KB, 1)
            ArgsKB   = [math]::Round((($attrs.args -as [string]).Length) / 1KB, 1)
        }
    } |
    Sort-Object ResultKB -Descending |
    Select-Object -First 15 |
    Format-Table -AutoSize

if (Test-Path $LlamaLog) {
    Write-Host "llama-server prompt sizes" -ForegroundColor Cyan
    Select-String -Path $LlamaLog -Pattern 'task\.n_tokens = \d+|cancel task|prompt processing progress' |
        Select-Object -Last 30 |
        ForEach-Object { $_.Line }
} else {
    Write-Host "No llama-server log found at $LlamaLog" -ForegroundColor Yellow
}