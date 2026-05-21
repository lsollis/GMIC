#requires -Version 3.0

<#
.SYNOPSIS
    Build and start the FL client container, then open an interactive shell inside it.

.DESCRIPTION
    This script builds the container image using Docker Compose, starts the container detached,
    then drops the user into the running container shell. From inside the container, the user
    manually runs the startup script that creates a tmux session and launches the NVFLARE client.

.PARAMETER DataDir
    Path to the data directory to mount in the container

.PARAMETER GPUs
    GPU devices to use. Options: "all", "none", or specific GPU IDs like "0,1"
    Default: "all"

.EXAMPLE
    .\run_container_windows.ps1 -DataDir ".\data"

.EXAMPLE
    .\run_container_windows.ps1 -DataDir ".\data" -GPUs "0,1"

.EXAMPLE
    .\run_container_windows.ps1 -DataDir ".\data" -GPUs "none"
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

# Ensure UID/GID environment variables exist for Compose interpolation (harmless if unused)
try {
    if (-not $env:UID) { $env:UID = "1000" }
    if (-not $env:GID) { $env:GID = "1000" }
    Write-InfoMessage "Environment UID=$($env:UID), GID=$($env:GID)"
}
catch {
    Write-ErrorMessage "Failed to set UID/GID environment variables: $($_.Exception.Message)"
    exit 1
}

try {
    # === Validate data directory ===
    if (-not (Test-Path $DataDir -PathType Container)) {
        Write-ErrorMessage "Data directory '$DataDir' does not exist."
        exit 1
    }

    # === Normalize paths ===
    $DataDirResolved = Resolve-Path $DataDir

    # === Set environment variables for Compose interpolation ===
    $env:DATA_DIR    = $DataDirResolved.Path
    $env:GPU_DEVICES = $GPUs

    # === Display summary ===
    Write-Host ""
    Write-Host "[CONFIG] FL Container Configuration" -ForegroundColor Blue
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
    $composeMode = $null  # "v2" or "legacy"
    try {
        $composeVersion = docker compose version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $composeMode = "v2"
            Write-InfoMessage "Docker Compose (v2): $composeVersion"
        } else {
            $composeVersion = docker-compose --version 2>$null
            if ($LASTEXITCODE -eq 0) {
                $composeMode = "legacy"
                Write-InfoMessage "Docker Compose (legacy): $composeVersion"
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

    # Helper to run compose with either v2 or legacy
    function Invoke-Compose {
        param([Parameter(Mandatory=$true)][string[]]$Args)
        if ($composeMode -eq "v2") {
            & docker compose @Args
        } else {
            & docker-compose @Args
        }
        return $LASTEXITCODE
    }

    Write-Host ""

    # === Build image ===
    Write-BuildMessage "Building image..."
    Write-Host ""
    $rc = Invoke-Compose -Args @("build","gmic_fl")
    if ($rc -ne 0) {
        Write-ErrorMessage "Build failed with exit code $rc"
        exit $rc
    }
    Write-SuccessMessage "Image built successfully"
    Write-Host ""

    # === Start container detached ===
    Write-LaunchMessage "Starting container (detached)..."
    Write-Host ""
    $rc = Invoke-Compose -Args @("up","-d","gmic_fl")
    if ($rc -ne 0) {
        Write-ErrorMessage "Failed to start container (compose up -d) with exit code $rc"
        exit $rc
    }
    Write-SuccessMessage "Container started: gmic_fl_container"
    Write-Host ""

    # === Drop user into container shell ===
    Write-InfoMessage "Opening interactive shell in container..."
    Write-Host ""
    & docker exec -it gmic_fl_container bash -lc @'
cd /workspace
echo "Inside container."
echo ""
echo "Next steps:"
echo "  1) Try:  bash /workspace/start_fl_client.sh"
echo "     or:   chmod +x /workspace/start_fl_client.sh && /workspace/start_fl_client.sh"
echo ""
echo "  2) Fallback (if needed):"
echo "       cd /workspace/start"
echo "       chmod +x start_fl.sh"
echo "       ./start_fl.sh"
echo ""
echo "  3) Attach later:"
echo "       tmux attach -t fl_client"
echo ""
exec bash
'@

    $rc = $LASTEXITCODE
    Write-Host ""
    if ($rc -eq 0) {
        Write-SuccessMessage "Exited container shell."
    } else {
        Write-ErrorMessage "docker exec returned exit code $rc"
        exit $rc
    }
}
catch {
    Write-ErrorMessage "Script failed: $($_.Exception.Message)"
    exit 1
}
finally {
    Write-Host ""
    Write-InfoMessage "Cleaning up environment variables..."
    Remove-Item Env:DATA_DIR     -ErrorAction SilentlyContinue
    Remove-Item Env:GPU_DEVICES  -ErrorAction SilentlyContinue
    Remove-Item Env:CURRENT_DIR  -ErrorAction SilentlyContinue
}

Write-Host ""
Write-SuccessMessage "Done."