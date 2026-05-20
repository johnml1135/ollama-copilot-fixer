<#
.SYNOPSIS
    Clones and builds ikawrakow/ik_llama.cpp for the local launcher.

.DESCRIPTION
    ik_llama.cpp does not publish the same Windows binary release assets as
    upstream llama.cpp, so this installer builds from source into
    tools\ik_llama.cpp. The default CUDA build is targeted at an RTX 3090
    compute capability 8.6 GPU.
#>
[CmdletBinding()]
param(
    [ValidateSet('cuda', 'cpu')]
    [string]$Backend = 'cuda',

    [Alias('Tag')]
    [string]$Ref = 'main',

    [string]$RepoUrl = 'https://github.com/ikawrakow/ik_llama.cpp.git',

    [string]$CudaArchitectures = '86',

    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$toolsDir = Join-Path $repoRoot 'tools\ik_llama.cpp'
$buildDir = Join-Path $toolsDir 'build'

function Assert-Command {
    param(
        [string]$Name,
        [string]$InstallHint
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found on PATH. $InstallHint"
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

function Invoke-CheckedCommand {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE."
    }
}

function Import-VisualStudioBuildEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) { return }

    $vswhereCandidates = @(
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'),
        (Join-Path $env:ProgramFiles 'Microsoft Visual Studio\Installer\vswhere.exe')
    ) | Where-Object { $_ -and (Test-Path $_) }

    $installPath = $null
    foreach ($vswhere in $vswhereCandidates) {
        $installPath = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($installPath) { break }
    }

    $vcvarsCandidates = @()
    if ($installPath) {
        $vcvarsCandidates += (Join-Path $installPath 'VC\Auxiliary\Build\vcvarsall.bat')
    }
    $vcvarsCandidates += @(
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat'),
        (Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat')
    )

    $vcvars = $vcvarsCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
    if (-not $vcvars) {
        throw "cl.exe was not found on PATH, and Visual Studio Build Tools 2022 could not be located. Install the C++ build tools from https://aka.ms/vs/17/release/vs_buildtools.exe, or run this script from a Developer PowerShell."
    }

    Write-Host "Importing Visual Studio C++ build environment..." -ForegroundColor Cyan
    $envLines = & cmd.exe /s /c "`"$vcvars`" x64 >nul && set"
    foreach ($line in $envLines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }

    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "Visual Studio environment import completed, but cl.exe is still not on PATH. Run from a Developer PowerShell and try again."
    }
}

function Copy-CudaRuntimeDlls {
    param([string]$Destination)

    if ($Backend -ne 'cuda') { return }

    $cudaRoot = $env:CUDA_PATH
    if (-not $cudaRoot) {
        $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
        if ($nvcc) {
            $cudaRoot = Split-Path -Parent (Split-Path -Parent $nvcc.Source)
        }
    }

    if (-not $cudaRoot) { return }

    $cudaBin = Join-Path $cudaRoot 'bin'
    if (-not (Test-Path $cudaBin)) { return }

    foreach ($pattern in @('cublas64_*.dll', 'cublasLt64_*.dll', 'cudart64_*.dll')) {
        Get-ChildItem -Path $cudaBin -Filter $pattern -File -ErrorAction SilentlyContinue |
            Copy-Item -Destination $Destination -Force
    }

    $openMpDll = Join-Path $env:WINDIR 'System32\libomp140.x86_64.dll'
    if (Test-Path $openMpDll) {
        Copy-Item -Path $openMpDll -Destination $Destination -Force
    }
}

# Remove the common WinGet upstream llama.cpp path from this session so the
# launcher cannot accidentally pick the wrong backend through PATH precedence.
$env:PATH = ($env:PATH -split ';' | Where-Object { $_ -notmatch 'WinGet\\Packages\\ggml\.llamacpp' }) -join ';'

$existingServer = Find-LlamaServerExe -Root $toolsDir
if ($existingServer -and (-not $Force)) {
    Write-Host "Local ik_llama.cpp already exists at $existingServer" -ForegroundColor Green
    $env:PATH = "$(Split-Path -Parent $existingServer);$env:PATH"
    return (Split-Path -Parent $existingServer)
}

Assert-Command -Name 'git' -InstallHint 'Install Git for Windows: https://git-scm.com/download/win'
Assert-Command -Name 'cmake' -InstallHint 'Install CMake, or install Visual Studio Build Tools 2022 with the C++ workload.'

if ($env:OS -eq 'Windows_NT') {
    Import-VisualStudioBuildEnvironment
}

$clCommand = $null
$nvccCommand = $null
if ($env:OS -eq 'Windows_NT') {
    $clCommand = (Get-Command cl.exe -ErrorAction Stop).Source
}

if ($Backend -eq 'cuda') {
    Assert-Command -Name 'nvcc' -InstallHint 'Install the NVIDIA CUDA Toolkit. The RTX 3090 target uses CMAKE_CUDA_ARCHITECTURES=86 by default.'
    $nvccCommand = (Get-Command nvcc -ErrorAction Stop).Source
}

if ($Force -and (Test-Path $toolsDir)) {
    Remove-Item $toolsDir -Recurse -Force
}

if (-not (Test-Path $toolsDir)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $toolsDir) | Out-Null
    Write-Host "Cloning ik_llama.cpp ($Ref)..." -ForegroundColor Cyan
    Invoke-CheckedCommand -FilePath 'git' -Arguments @('clone', '--depth', '1', '--branch', $Ref, $RepoUrl, $toolsDir)
} else {
    Write-Host "Using existing ik_llama.cpp source at $toolsDir" -ForegroundColor Cyan
}

if ((Test-Path $buildDir) -and (-not $existingServer)) {
    Write-Host "Removing incomplete ik_llama.cpp build cache..." -ForegroundColor Cyan
    Remove-Item $buildDir -Recurse -Force
}

$cmakeConfigureArgs = @(
    '-S', $toolsDir,
    '-B', $buildDir,
    '-DCMAKE_BUILD_TYPE=Release',
    '-DGGML_NATIVE=ON',
    '-DLLAMA_CURL=OFF'
)

if (Get-Command ninja -ErrorAction SilentlyContinue) {
    $cmakeConfigureArgs = @('-G', 'Ninja') + $cmakeConfigureArgs
}

if ($env:OS -eq 'Windows_NT') {
    $cmakeConfigureArgs += @(
        "-DCMAKE_C_COMPILER=$clCommand",
        "-DCMAKE_CXX_COMPILER=$clCommand",
        '-DCMAKE_CXX_FLAGS=/EHsc /FIcstdint'
    )
}

if ($Backend -eq 'cuda') {
    $cmakeConfigureArgs += @(
        '-DGGML_CUDA=ON',
        "-DCMAKE_CUDA_COMPILER=$nvccCommand",
        "-DCMAKE_CUDA_HOST_COMPILER=$clCommand",
        "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures",
        '-DGGML_CUDA_USE_GRAPHS=ON',
        '-DGGML_SCHED_MAX_COPIES=1',
        '-DGGML_OPENMP=ON'
    )
}

Write-Host "Configuring ik_llama.cpp ($Backend)..." -ForegroundColor Cyan
Invoke-CheckedCommand -FilePath 'cmake' -Arguments $cmakeConfigureArgs

Write-Host "Building ik_llama.cpp llama-server..." -ForegroundColor Cyan
Invoke-CheckedCommand -FilePath 'cmake' -Arguments @('--build', $buildDir, '--config', 'Release')

$serverExe = Find-LlamaServerExe -Root $toolsDir
if (-not $serverExe) {
    throw "Build completed but llama-server.exe was not found under $toolsDir."
}

$serverDir = Split-Path -Parent $serverExe
Copy-CudaRuntimeDlls -Destination $serverDir

$env:PATH = "$serverDir;$env:PATH"
Write-Host "Successfully built ik_llama.cpp llama-server at $serverExe" -ForegroundColor Green
return $serverDir