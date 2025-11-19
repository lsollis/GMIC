#requires -Version 3.0

<#
.SYNOPSIS
    Run FL client with Docker Compose

.DESCRIPTION
    This script builds and runs a federated learning client using Docker Compose.

.PARAMETER DataDir
    Path to the data directory to mount in the container

.PARAMETER GPUs
    GPU devices to use. Options: "all", "none", or specific GPU IDs like "0,1"
    Default: "all"

.EXAMPLE
    .\run_client.ps1 -DataDir ".\data"
    
.EXAMPLE
    .\run_client.ps1 -DataDir ".\data" -GPUs "0,1"
    
.EXAMPLE
    .\run_client.ps1 -DataDir ".\data" -GPUs "none"
#>

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$DataDir,
    
    [Parameter(Position=1)]
    [string]$GPUs = "all"
)

# Always run from the script's directory so Compose finds docker-compose.yml
Set-Location -Path $PSScriptRoot
$env:CURRENT_DIR = $PSScriptRoot

# Validate -GPUs format: 'all', 'none', or '0,1,2'
if ($GPUs -notmatch '^(all|none|\d+(,\d+)*)$') {
    Write-Host "[ERROR] Invalid -GPUs value '$GPUs' (use 'all', 'none', or e.g. '0,1')" -ForegroundColor Red
    exit 1
}

# Set error action preference
$ErrorActionPreference = "Stop"

# ---- Messaging helpers ----
function Write-StatusMessage {
    param(
        [string]$Message,
        [string]$Icon = "[INFO]",
        [string]$Color = "White"
    )
    Write-Host "$Icon $Message" -ForegroundColor $Color
}
function Write-SuccessMessage { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[SUCCESS]" -Color "Green" }
function Write-ErrorMessage   { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[ERROR]"   -Color "Red"   }
function Write-InfoMessage    { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[INFO]"    -Color "Cyan"  }
function Write-BuildMessage   { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[BUILD]"   -Color "Yellow"}
function Write-LaunchMessage  { param([string]$Message) Write-StatusMessage -Message $Message -Icon "[LAUNCH]"  -Color "Magenta"}

# Ensure UID/GID environment variables exist for Compose interpolation
try {
    if (-not $env:UID) {
        # Windows users don’t have a numeric UID, so use 1000 (Linux default)
        $env:UID = "1000"
    }
    if (-not $env:GID) {
        $env:GID = "1000"
    }
    Write-InfoMessage "Environment UID=$($env:UID), GID=$($env:GID)"
}
catch {
    Write-ErrorMessage "Failed to set UID/GID environment variables: $($_.Exception.Message)"
    exit 1
}

# Main script logic
try {
    # === Validate parameters ===
    if (-not $DataDir) {
        Write-ErrorMessage "Usage: .\run_client.ps1 -DataDir /path/to/data [-GPUs gpus]"
        Write-Host ""
        Write-Host "Examples:"
        Write-Host "  .\run_client.ps1 -DataDir `".\data`""
        Write-Host "  .\run_client.ps1 -DataDir `".\data`" -GPUs `"0,1`""
        Write-Host "  .\run_client.ps1 -DataDir `".\data`" -GPUs `"none`""
        exit 1
    }

    # === Validate data directory ===
    if (-not (Test-Path $DataDir -PathType Container)) {
        Write-ErrorMessage "Data directory '$DataDir' does not exist."
        exit 1
    }

    # === Normalize paths ===
    $DataDirResolved = Resolve-Path $DataDir

    # === Set environment variables for Compose interpolation ===
    $env:DATA_DIR     = $DataDirResolved.Path
    $env:GPU_DEVICES  = $GPUs

    # === Display summary ===
    Write-Host ""
    Write-Host "[CONFIG] Federated Learning Client Configuration" -ForegroundColor Blue
    Write-Host ("=" * 50) -ForegroundColor Blue
    Write-InfoMessage "Script directory: $env:CURRENT_DIR"
    Write-InfoMessage "Data directory:   $($env:DATA_DIR)"
    Write-InfoMessage "GPU devices:      $($env:GPU_DEVICES)"
    Write-Host ""

    # === Check Docker CLI present and daemon running ===
    try {
        docker --version | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Docker CLI not found" }

        docker info | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Docker daemon not running" }

        Write-InfoMessage "Docker Desktop and engine detected."
    }
    catch {
        Write-ErrorMessage "Docker Desktop is not installed or not running."
        exit 1
    }

    # === Check Docker Compose (new or legacy) ===
    $composeCommand = $null
    try {
        $composeVersion = docker compose version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $composeCommand = "docker compose"
            Write-InfoMessage "Docker Compose: $composeVersion"
        } else {
            $composeVersion = docker-compose --version 2>$null
            if ($LASTEXITCODE -eq 0) {
                $composeCommand = "docker-compose"
                Write-InfoMessage "Docker Compose: $composeVersion"
            } else {
                throw "Docker Compose not found"
            }
        }
    }
    catch {
        Write-ErrorMessage "Docker Compose is not available."
        Write-Host "Please ensure Docker Compose is installed."
        exit 1
    }

    Write-Host ""

    # === Build Docker image ===
    Write-BuildMessage "Building Docker image..."
    Write-Host ""
    try {
        if ($composeCommand -eq "docker compose") {
            & docker compose build
        } else {
            & docker-compose build
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Docker build failed with exit code $LASTEXITCODE"
        }
        Write-SuccessMessage "Docker image built successfully"
    }
    catch {
        Write-ErrorMessage "Failed to build Docker image: $($_.Exception.Message)"
        exit 1
    }

    Write-Host ""

    # === Run container ===
    Write-LaunchMessage "Starting FL client container..."
    Write-Host ""
    try {
        if ($composeCommand -eq "docker compose") {
            & docker compose up
        } else {
            & docker-compose up
        }
        $containerExitCode = $LASTEXITCODE

        Write-Host ""
        if ($containerExitCode -eq 0) {
            Write-SuccessMessage "FL client completed successfully"
        } else {
            Write-ErrorMessage "FL client exited with code $containerExitCode"
        }
    }
    catch {
        Write-ErrorMessage "Failed to run container: $($_.Exception.Message)"
        exit 1
    }
}
catch {
    Write-ErrorMessage "Script failed: $($_.Exception.Message)"
    exit 1
}
finally {
    # === Cleanup ===
    Write-Host ""
    Write-InfoMessage "Cleaning up environment variables..."
    Remove-Item Env:DATA_DIR     -ErrorAction SilentlyContinue
    Remove-Item Env:GPU_DEVICES  -ErrorAction SilentlyContinue
    Remove-Item Env:CURRENT_DIR  -ErrorAction SilentlyContinue
}

Write-Host ""
Write-SuccessMessage "Script completed"