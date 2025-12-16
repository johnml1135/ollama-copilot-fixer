<#
.SYNOPSIS
    Fix and configure GGUF models (including sharded) for GitHub Copilot with Tool capability. 

.DESCRIPTION
    This script handles multiple scenarios: 
    1. Downloads models from HuggingFace repositories
    2. Processes local GGUF files
    3. Automatically detects and merges sharded GGUF files
    4. Applies tool-enabled templates for GitHub Copilot compatibility
    5. Creates optimized Ollama models with full Tool support

.PARAMETER ModelSource
    Can be:
    - Local file path:  "C:\Models\model.gguf"
    - HuggingFace repo:  "unsloth/Llama-3.2-3B-Instruct-GGUF"
    - HuggingFace URL: "https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF"
    - First shard of split model: "C:\Models\model-00001-of-00005.gguf"

.PARAMETER ModelName
    (Optional) Custom name for the model in Ollama. 

.PARAMETER Architecture
    (Optional) Force specific architecture (llama3, mistral, phi3, gemma2, qwen).

.PARAMETER ContextLength
    (Optional) Set custom context length.  Default is 8192.

. PARAMETER Temperature
    (Optional) Set default temperature. Default is 0.7.

.PARAMETER QuantizationType
    (Optional) Specific quantization to download from HF (e.g., Q4_K_M, Q5_K_M, Q6_K).

.PARAMETER LlamaCppPath
    (Optional) Path to llama.cpp installation (for merging sharded files).

.PARAMETER KeepDownloads
    (Optional) Keep downloaded/merged files after import.  Default is false.

.EXAMPLE
    .\Setup-OllamaToolModel.ps1 -ModelSource "unsloth/Llama-3.2-3B-Instruct-GGUF"

.EXAMPLE
    .\Setup-OllamaToolModel. ps1 -ModelSource "C:\Models\model-00001-of-00005.gguf" -ModelName "my-model"

.EXAMPLE
    .\Setup-OllamaToolModel.ps1 -ModelSource "bartowski/Llama-3.3-70B-Instruct-GGUF" -QuantizationType "Q4_K_M"
#>

param(
    [Parameter(Mandatory=$true, HelpMessage="Model source:  local path, HF repo, or HF URL")]
    [string]$ModelSource,
    
    [Parameter(Mandatory=$false)]
    [string]$ModelName,
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("llama3", "mistral", "phi3", "gemma2", "qwen", "auto")]
    [string]$Architecture = "auto",
    
    [Parameter(Mandatory=$false)]
    [int]$ContextLength = 8192,
    
    [Parameter(Mandatory=$false)]
    [decimal]$Temperature = 0.7,
    
    [Parameter(Mandatory=$false)]
    [string]$QuantizationType,
    
    [Parameter(Mandatory=$false)]
    [string]$LlamaCppPath,
    
    [Parameter(Mandatory=$false)]
    [bool]$KeepDownloads = $false
)

# Color output functions
function Write-Success { param([string]$Message); Write-Host "✓ $Message" -ForegroundColor Green }
function Write-Info { param([string]$Message); Write-Host "ℹ $Message" -ForegroundColor Cyan }
function Write-Error-Custom { param([string]$Message); Write-Host "✗ $Message" -ForegroundColor Red }
function Write-Warning-Custom { param([string]$Message); Write-Host "⚠ $Message" -ForegroundColor Yellow }

# Check if command exists
function Test-CommandExists {
    param([string]$Command)
    try {
        $null = Get-Command $Command -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

# Check dependencies
function Test-Dependencies {
    $deps = @{
        "Ollama" = Test-CommandExists "ollama"
        "HuggingFace CLI" = Test-CommandExists "hf"
    }
    
    $allGood = $true
    foreach ($dep in $deps.GetEnumerator()) {
        if ($dep.Value) {
            Write-Success "$($dep.Key) is installed"
        } else {
            if ($dep.Key -eq "Ollama") {
                Write-Error-Custom "$($dep.Key) is required but not found"
                $allGood = $false
            } else {
                Write-Warning-Custom "$($dep.Key) is not installed (required for HF downloads)"
            }
        }
    }
    
    return $allGood
}

# Find llama.cpp installation
function Find-LlamaCpp {
    param([string]$CustomPath)
    
    if ($CustomPath -and (Test-Path $CustomPath)) {
        $llamaSplit = Join-Path $CustomPath "llama-gguf-split. exe"
        if (Test-Path $llamaSplit) {
            return $llamaSplit
        }
    }
    
    # Check common locations
    $commonPaths = @(
        "C:\llama.cpp\bin\llama-gguf-split.exe",
        "C:\llama.cpp\llama-gguf-split. exe",
        "$env:USERPROFILE\llama. cpp\bin\llama-gguf-split.exe",
        "$env:USERPROFILE\llama.cpp\llama-gguf-split. exe"
    )
    
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            return $path
        }
    }
    
    # Check if in PATH
    if (Test-CommandExists "llama-gguf-split") {
        return "llama-gguf-split"
    }
    
    return $null
}

# Parse HuggingFace repo from URL or ID
function Get-HFRepoId {
    param([string]$Source)
    
    if ($Source -match "huggingface\. co/([^/]+/[^/]+)") {
        return $Matches[1]
    } elseif ($Source -match "^[^/]+/[^/]+$") {
        return $Source
    }
    
    return $null
}

# Download from HuggingFace
function Get-ModelFromHF {
    param(
        [string]$RepoId,
        [string]$QuantType,
        [string]$DestPath
    )
    
    Write-Info "Downloading from HuggingFace: $RepoId"
    
    if (-not (Test-CommandExists "hf")) {
        Write-Error-Custom "HuggingFace CLI not found. Install with:"
        Write-Host 'powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"' -ForegroundColor Yellow
        return $null
    }
    
    # Create temp download directory
    $downloadDir = Join-Path $DestPath "hf_download_$(Get-Random)"
    New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
    
    try {
        # Build download command
        $hfArgs = @("download", $RepoId, "--local-dir", $downloadDir)
        
        if ($QuantType) {
            $pattern = "*$QuantType*. gguf"
            $hfArgs += @("--include", $pattern)
            Write-Info "Filtering for quantization: $QuantType"
        } else {
            $hfArgs += @("--include", "*. gguf")
        }
        
        Write-Info "Executing download (this may take a while)..."
        $process = Start-Process -FilePath "hf" -ArgumentList $hfArgs -NoNewWindow -Wait -PassThru
        
        if ($process.ExitCode -ne 0) {
            Write-Error-Custom "HuggingFace download failed"
            return $null
        }
        
        # Find downloaded GGUF files
        $ggufFiles = Get-ChildItem -Path $downloadDir -Filter "*.gguf" -Recurse | Sort-Object Name
        
        if ($ggufFiles.Count -eq 0) {
            Write-Error-Custom "No GGUF files found after download"
            return $null
        }
        
        Write-Success "Downloaded $($ggufFiles.Count) GGUF file(s)"
        
        # Return first file or first shard
        return $ggufFiles[0]. FullName
        
    } catch {
        Write-Error-Custom "Error downloading from HuggingFace: $_"
        return $null
    }
}

# Detect if file is part of a sharded set
function Test-IsShardedModel {
    param([string]$FilePath)
    
    $filename = Split-Path $FilePath -Leaf
    
    # Check for common shard patterns
    if ($filename -match "-\d{5}-of-\d{5}\. gguf$" -or 
        $filename -match "-part-\d+\.gguf$" -or
        $filename -match "\. part\d+\.gguf$") {
        return $true
    }
    
    # Check if other shards exist in same directory
    $directory = Split-Path $FilePath -Parent
    $baseName = $filename -replace "-\d{5}-of-\d{5}\.gguf$", ""
    $baseName = $baseName -replace "-part-\d+\.gguf$", ""
    
    $relatedFiles = Get-ChildItem -Path $directory -Filter "$baseName*.gguf" | Where-Object { $_.Name -match "-\d{5}-of-\d{5}|part-?\d+" }
    
    return ($relatedFiles.Count -gt 1)
}

# Get all shard files for a model
function Get-ShardFiles {
    param([string]$FirstShardPath)
    
    $directory = Split-Path $FirstShardPath -Parent
    $filename = Split-Path $FirstShardPath -Leaf
    
    # Extract base name pattern
    $baseName = $filename -replace "-\d{5}-of-\d{5}\. gguf$", ""
    $baseName = $baseName -replace "-part-\d+\. gguf$", ""
    
    $shardFiles = Get-ChildItem -Path $directory -Filter "$baseName*.gguf" | 
                  Where-Object { $_.Name -match "-\d{5}-of-\d{5}|part-?\d+" } |
                  Sort-Object Name
    
    return $shardFiles
}

# Merge sharded GGUF files
function Merge-ShardedModel {
    param(
        [string]$FirstShardPath,
        [string]$OutputPath,
        [string]$LlamaSplitPath
    )
    
    Write-Info "Detected sharded model, preparing to merge..."
    
    if (-not $LlamaSplitPath) {
        Write-Error-Custom "llama-gguf-split not found.  Please install llama.cpp from:"
        Write-Host "https://github.com/ggerganov/llama.cpp/releases" -ForegroundColor Yellow
        return $null
    }
    
    $shards = Get-ShardFiles -FirstShardPath $FirstShardPath
    Write-Info "Found $($shards.Count) shard file(s)"
    
    foreach ($shard in $shards) {
        Write-Info "  - $($shard.Name) ($([math]::Round($shard.Length / 1GB, 2)) GB)"
    }
    
    Write-Info "Merging shards (this may take several minutes)..."
    
    try {
        $mergeArgs = @("--merge", $FirstShardPath, $OutputPath)
        $process = Start-Process -FilePath $LlamaSplitPath -ArgumentList $mergeArgs -NoNewWindow -Wait -PassThru
        
        if ($process.ExitCode -ne 0) {
            Write-Error-Custom "Merge failed with exit code $($process.ExitCode)"
            return $null
        }
        
        if (Test-Path $OutputPath) {
            $mergedSize = (Get-Item $OutputPath).Length
            Write-Success "Merge completed:  $([math]::Round($mergedSize / 1GB, 2)) GB"
            return $OutputPath
        } else {
            Write-Error-Custom "Merge completed but output file not found"
            return $null
        }
        
    } catch {
        Write-Error-Custom "Error during merge: $_"
        return $null
    }
}

# Detect architecture from GGUF metadata
function Get-ModelArchitecture {
    param([string]$Path)
    
    Write-Info "Analyzing model file to detect architecture..."
    
    try {
        $bytes = [System.IO.File]::ReadAllBytes($Path) | Select-Object -First 10240
        $content = [System.Text.Encoding]::ASCII.GetString($bytes).ToLower()
        
        if ($content -match "llama.*3\.[0-9]|llama3|llama-3") { return "llama3" }
        elseif ($content -match "mistral|mixtral") { return "mistral" }
        elseif ($content -match "phi-3|phi3|phi-4|phi4") { return "phi3" }
        elseif ($content -match "gemma.*2|gemma-2") { return "gemma2" }
        elseif ($content -match "qwen.*2|qwen-2") { return "qwen" }
        elseif ($content -match "llama") { return "llama3" }
        
        $filename = Split-Path $Path -Leaf
        if ($filename -match "llama.*3|nemotron") { return "llama3" }
        elseif ($filename -match "mistral|mixtral") { return "mistral" }
        elseif ($filename -match "phi") { return "phi3" }
        elseif ($filename -match "gemma") { return "gemma2" }
        elseif ($filename -match "qwen") { return "qwen" }
        
        Write-Warning-Custom "Could not detect architecture.  Defaulting to llama3."
        return "llama3"
    }
    catch {
        Write-Warning-Custom "Error reading file.  Defaulting to llama3."
        return "llama3"
    }
}

# Get model template with tool support
function Get-ModelTemplate {
    param([string]$Arch)
    
    $templates = @{
        "llama3" = @"
{{ if . Messages }}
{{- if or .System . Tools }}<|start_header_id|>system<|end_header_id|>
{{- if .System }}

{{ . System }}
{{- end }}
{{- if .Tools }}

You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original user question.
{{- end }}<|eot_id|>
{{- end }}
{{- range .Messages }}
<|start_header_id|>{{ .Role }}<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- end }}
{{- else }}
<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{{ . Prompt }}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{{- end }}
"@
        
        "mistral" = @"
{{ if .Messages }}
{{- if or .System .Tools }}[INST]
{{- if .System }}{{ .System }}
{{- end }}
{{- if .Tools }}

You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original user question. 
{{- end }}[/INST]
{{- end }}
{{- range .Messages }}
{{- if eq .Role "user" }}[INST] {{ .Content }} [/INST]
{{- else if eq .Role "assistant" }}{{ .Content }}</s>
{{- end }}
{{- end }}
{{- else }}[INST] {{ if .System }}{{ .System }}

{{ end }}{{ . Prompt }} [/INST]
{{- end }}
"@
        
        "phi3" = @"
{{ if . Messages }}
{{- if or . System .Tools }}<|system|>
{{- if .System }}{{ .System }}
{{- end }}
{{- if .Tools }}

You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original user question. 
{{- end }}<|end|>
{{- end }}
{{- range .Messages }}
<|{{ . Role }}|>
{{ .Content }}<|end|>
{{- end }}
<|assistant|>
{{- else }}<|system|>
{{ . System }}<|end|>
<|user|>
{{ . Prompt }}<|end|>
<|assistant|>
{{- end }}
"@
        
        "gemma2" = @"
{{ if .Messages }}
{{- if or .System .Tools }}<start_of_turn>model
{{- if .System }}{{ . System }}
{{- end }}
{{- if .Tools }}

You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original user question.
{{- end }}<end_of_turn>
{{- end }}
{{- range .Messages }}
<start_of_turn>{{ .Role }}
{{ .Content }}<end_of_turn>
{{- end }}
<start_of_turn>model
{{- else }}<start_of_turn>system
{{ .System }}<end_of_turn>
<start_of_turn>user
{{ . Prompt }}<end_of_turn>
<start_of_turn>model
{{- end }}
"@

        "qwen" = @"
{{ if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original user question. 
{{- end }}<|im_end|>
{{- end }}
{{- range .Messages }}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{- end }}
<|im_start|>assistant
{{- else }}<|im_start|>system
{{ . System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{- end }}
"@
    }
    
    return $templates[$Arch]
}

# Get stop sequences
function Get-StopSequences {
    param([string]$Arch)
    
    $stopSequences = @{
        "llama3" = @("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")
        "mistral" = @("</s>", "[INST]", "[/INST]")
        "phi3" = @("<|end|>", "<|system|>", "<|user|>", "<|assistant|>")
        "gemma2" = @("<end_of_turn>", "<start_of_turn>")
        "qwen" = @("<|im_start|>", "<|im_end|>")
    }
    
    return $stopSequences[$Arch]
}

# Generate Modelfile
function New-Modelfile {
    param(
        [string]$ModelPath,
        [string]$Architecture,
        [int]$ContextLength,
        [decimal]$Temperature
    )
    
    $template = Get-ModelTemplate -Arch $Architecture
    $stopSeqs = Get-StopSequences -Arch $Architecture
    
    $modelfile = @"
# Auto-generated Modelfile with Tool capability for GitHub Copilot
# Architecture: $Architecture
# Generated: $(Get-Date -Format "yyyy-MM-dd HH: mm:ss")

FROM $ModelPath

# Template with Tool support
TEMPLATE """$template"""

# Stop sequences
"@
    
    foreach ($seq in $stopSeqs) {
        $modelfile += "`nPARAMETER stop `"$seq`""
    }
    
    $modelfile += @"


# Model parameters
PARAMETER temperature $Temperature
PARAMETER num_ctx $ContextLength
PARAMETER num_predict -1

# System message
SYSTEM """You are a helpful AI assistant with tool calling capabilities. You can help with code, answer questions, and use tools when needed."""
"@
    
    return $modelfile
}

# Main execution
Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       Ollama Copilot Fixer - Model Setup Tool             ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Step 1: Check dependencies
Write-Info "Step 1: Checking dependencies..."
if (-not (Test-Dependencies)) {
    exit 1
}

# Find llama.cpp if needed
$llamaSplit = Find-LlamaCpp -CustomPath $LlamaCppPath
if ($llamaSplit) {
    Write-Success "llama.cpp found at $(Split-Path $llamaSplit -Parent)"
} else {
    Write-Warning-Custom "llama-gguf-split not found (needed for sharded models)"
}

# Step 2: Process model source
Write-Info "`nStep 2: Processing model source..."

$workingFile = $null
$tempDir = Join-Path $env:TEMP "ollama_fixer_$(Get-Random)"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

try {
    # Check if source is HuggingFace
    $hfRepo = Get-HFRepoId -Source $ModelSource
    
    if ($hfRepo) {
        Write-Info "Detected HuggingFace repository:  $hfRepo"
        $workingFile = Get-ModelFromHF -RepoId $hfRepo -QuantType $QuantizationType -DestPath $tempDir
        
        if (-not $workingFile) {
            Write-Error-Custom "Failed to download from HuggingFace"
            exit 1
        }
    } elseif (Test-Path $ModelSource) {
        Write-Success "Using local file: $ModelSource"
        $workingFile = $ModelSource
    } else {
        Write-Error-Custom "Model source not found:  $ModelSource"
        Write-Info "Provide a local path or HuggingFace repo (e.g., 'unsloth/Llama-3.2-3B-Instruct-GGUF')"
        exit 1
    }
    
    # Step 3: Check for sharded files
    Write-Info "`nStep 3: Checking for sharded files..."
    
    $finalFile = $workingFile
    $isSharded = Test-IsShardedModel -FilePath $workingFile
    
    if ($isSharded) {
        Write-Warning-Custom "Sharded model detected"
        
        if (-not $llamaSplit) {
            Write-Error-Custom "Cannot merge sharded files without llama-gguf-split"
            Write-Info "Install llama.cpp from: https://github.com/ggerganov/llama.cpp/releases"
            exit 1
        }
        
        $mergedFile = Join-Path $tempDir "merged_model.gguf"
        $finalFile = Merge-ShardedModel -FirstShardPath $workingFile -OutputPath $mergedFile -LlamaSplitPath $llamaSplit
        
        if (-not $finalFile) {
            Write-Error-Custom "Failed to merge sharded files"
            exit 1
        }
    } else {
        Write-Success "Model is a single file (no merging needed)"
    }
    
    $modelFileInfo = Get-Item $finalFile
    Write-Success "Working with:  $($modelFileInfo.Name) ($([math]::Round($modelFileInfo.Length / 1GB, 2)) GB)"
    
    # Step 4: Determine model name
    if ([string]::IsNullOrWhiteSpace($ModelName)) {
        $ModelName = [System.IO.Path]::GetFileNameWithoutExtension($modelFileInfo.Name).ToLower() -replace '[^a-z0-9-_]', '-'
        $ModelName = $ModelName -replace "-\d{5}-of-\d{5}$", ""
        Write-Info "`nStep 4: Auto-generated model name: $ModelName"
    } else {
        Write-Info "`nStep 4: Using custom model name: $ModelName"
    }
    
    # Step 5: Detect architecture
    Write-Info "`nStep 5: Determining model architecture..."
    if ($Architecture -eq "auto") {
        $Architecture = Get-ModelArchitecture -Path $finalFile
    }
    Write-Success "Using architecture: $Architecture"
    
    # Step 6: Generate Modelfile
    Write-Info "`nStep 6: Generating Modelfile with Tool capability..."
    $modelfilePath = Join-Path $tempDir "Modelfile"
    
    $absoluteModelPath = (Resolve-Path $finalFile).Path
    $modelfileContent = New-Modelfile -ModelPath $absoluteModelPath -Architecture $Architecture -ContextLength $ContextLength -Temperature $Temperature
    Set-Content -Path $modelfilePath -Value $modelfileContent -Encoding UTF8
    
    Write-Success "Modelfile generated"
    
    # Step 7: Create model in Ollama
    Write-Info "`nStep 7: Creating model in Ollama..."
    Write-Info "This may take a few moments..."
    
    try {
        $createOutput = & ollama create $ModelName -f $modelfilePath 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Model created successfully:  $ModelName"
        } else {
            Write-Error-Custom "Failed to create model in Ollama"
            Write-Host $createOutput -ForegroundColor Red
            exit 1
        }
    }
    catch {
        Write-Error-Custom "Error executing Ollama:  $_"
        exit 1
    }
    
    # Step 8: Verify
    Write-Info "`nStep 8: Verifying model..."
    Start-Sleep -Seconds 2
    
    $listOutput = & ollama list 2>&1 | Out-String
    if ($listOutput -match [regex]::Escape($ModelName)) {
        Write-Success "Model is registered in Ollama"
    } else {
        Write-Warning-Custom "Model may not be properly registered"
    }
    
    # Step 9: Test
    Write-Info "`nStep 9: Testing model..."
    try {
        $testOutput = & ollama run $ModelName "Hello, can you help me with code?" 2>&1 | Out-String
        if ($testOutput. Length -gt 0 -and -not ($testOutput -match "error|failed")) {
            Write-Success "Model responded successfully"
        }
    }
    catch {
        Write-Warning-Custom "Could not test model automatically"
    }
    
    # Success summary
    Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                    SETUP COMPLETE!                           ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════════╝`n" -ForegroundColor Green
    
    Write-Host "Model Name:        " -NoNewline -ForegroundColor Cyan
    Write-Host $ModelName -ForegroundColor White
    Write-Host "Architecture:      " -NoNewline -ForegroundColor Cyan
    Write-Host $Architecture -ForegroundColor White
    Write-Host "Context Length:   " -NoNewline -ForegroundColor Cyan
    Write-Host $ContextLength -ForegroundColor White
    Write-Host "Temperature:      " -NoNewline -ForegroundColor Cyan
    Write-Host $Temperature -ForegroundColor White
    
    Write-Host "`nNext Steps:" -ForegroundColor Yellow
    Write-Host "  1. In VS Code, open GitHub Copilot settings" -ForegroundColor White
    Write-Host "  2. Go to 'Manage Language Models'" -ForegroundColor White
    Write-Host "  3. Your model '$ModelName' should appear with 'Tool' capability" -ForegroundColor White
    Write-Host "  4. Enable it for use with GitHub Copilot Chat`n" -ForegroundColor White
    
    Write-Success "Your model is ready for use with GitHub Copilot!  🎉`n"
    
} finally {
    # Cleanup
    if (-not $KeepDownloads) {
        Write-Info "Cleaning up temporary files..."
        if (Test-Path $tempDir) {
            Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    } else {
        Write-Info "Temporary files kept at: $tempDir"
    }
}