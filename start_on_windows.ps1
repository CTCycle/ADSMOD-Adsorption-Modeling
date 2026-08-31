[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$AppDir = Join-Path $RepoRoot "app"
$BackendDir = Join-Path $AppDir "backend"
$ClientDir = Join-Path $AppDir "client"
$TestsDir = Join-Path $AppDir "tests"
$DefaultResourcesDir = Join-Path $AppDir "resources"
$ResourcesDir = $DefaultResourcesDir
$LogDir = Join-Path $ResourcesDir "logs"
$CheckpointsDir = Join-Path $ResourcesDir "checkpoints"
$ConfigFile = Join-Path $ResourcesDir "adsmod.json"
$RuntimesDir = Join-Path $RepoRoot "runtimes"
$StartupTempDir = Join-Path $BackendDir ".startup-temp"
$PythonDir = Join-Path $RuntimesDir "python"
$UvDir = Join-Path $RuntimesDir "uv"
$NodeDir = Join-Path $RuntimesDir "nodejs"
$RuntimeCacheDir = Join-Path $RuntimesDir "cache"
$TestCacheDir = Join-Path $TestsDir "cache"
$RuntimeTempDir = Join-Path $RuntimeCacheDir "temp"
$PytestCacheDir = Join-Path $TestCacheDir "pytest"
$PytestTempDir = Join-Path $TestCacheDir "pytest-tmp"
$RuffCacheDir = Join-Path $TestCacheDir "ruff"
$PythonCacheDir = Join-Path $TestCacheDir "python"
$MypyCacheDir = Join-Path $TestCacheDir "mypy"
$AngularCacheDir = Join-Path $TestCacheDir "angular"
$script:NextProgressId = 1
$script:ActiveProgressIds = [Collections.Generic.HashSet[int]]::new()

$PythonVersion = "3.14.2"
$PythonExe = Join-Path $PythonDir "python.exe"
$PythonPth = Join-Path $PythonDir "python314._pth"
$UvExe = Join-Path $UvDir "uv.exe"
$NodeExe = Join-Path $NodeDir "node.exe"
$NpmCmd = Join-Path $NodeDir "npm.cmd"
$VenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"
$UvCacheDir = $RuntimeCacheDir

$PythonArchive = "python-$PythonVersion-embed-amd64.zip"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVersion/$PythonArchive"
$UvUrl = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") {
    "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-pc-windows-msvc.zip"
} else {
    "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
}
$NodeVersion = "22.13.0"
$NodeArchive = "node-v$NodeVersion-win-x64.zip"
$NodeUrl = "https://nodejs.org/dist/v$NodeVersion/$NodeArchive"

# -----------------------------------------------------------------------------
# Console output and path-safety helpers
# -----------------------------------------------------------------------------
function Write-Step([string]$Message) {
    Clear-LauncherProgress
    Write-Host "[STEP] $Message" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Clear-LauncherProgress
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Clear-LauncherProgress
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Fatal([string]$Message) {
    Clear-LauncherProgress
    Write-Host "[FATAL] $Message" -ForegroundColor Red
}

function Start-LauncherProgress {
    param([Parameter(Mandatory)][string]$Activity, [Parameter(Mandatory)][string]$Status)
    $id = $script:NextProgressId++
    [void]$script:ActiveProgressIds.Add($id)
    Write-Progress -Id $id -Activity $Activity -Status $Status
    return $id
}

function Update-LauncherProgress {
    param(
        [Parameter(Mandatory)][int]$Id,
        [Parameter(Mandatory)][string]$Activity,
        [Parameter(Mandatory)][string]$Status,
        [Nullable[int]]$PercentComplete
    )
    if (-not $script:ActiveProgressIds.Contains($Id)) { return }
    $progress = @{ Id = $Id; Activity = $Activity; Status = $Status }
    if ($null -ne $PercentComplete) { $progress.PercentComplete = $PercentComplete }
    Write-Progress @progress
}

function Complete-LauncherProgress([int]$Id) {
    if ($script:ActiveProgressIds.Contains($Id)) {
        Write-Progress -Id $Id -Activity 'ADSMOD launcher' -Completed
        [void]$script:ActiveProgressIds.Remove($Id)
    }
}

function Clear-LauncherProgress {
    foreach ($id in @($script:ActiveProgressIds)) {
        Write-Progress -Id $id -Activity 'ADSMOD launcher' -Completed
        [void]$script:ActiveProgressIds.Remove($id)
    }
}

function Invoke-TrackedLauncherAction {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][scriptblock]$Action
    )
    Write-Step "Starting $Name"
    try {
        & $Action
        Write-Ok "$Name completed"
    } catch {
        Write-Fatal "$Name failed: $($_.Exception.Message)"
        throw
    }
}

function Assert-LastExitCode([string]$Operation) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

function Resolve-CanonicalPath([string]$ConfiguredPath) {
    if ([string]::IsNullOrWhiteSpace($ConfiguredPath)) {
        throw "The canonical configuration contains an empty path."
    }

    $expandedPath = [Environment]::ExpandEnvironmentVariables($ConfiguredPath.Trim())
    if (-not [System.IO.Path]::IsPathRooted($expandedPath)) {
        $expandedPath = Join-Path $RepoRoot $expandedPath
    }
    return [System.IO.Path]::GetFullPath($expandedPath)
}

function Remove-RepoPath([string]$Path) {
    $repoPrefix = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\') + '\'
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the repository: $fullPath"
    }

    try {
        $item = Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop
    } catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound) {
            return $true
        }
        Write-Warn "Skipping inaccessible path '$fullPath': $($_.Exception.Message)"
        return $false
    }
    try {
        Remove-Item -LiteralPath $item.FullName -Recurse -Force -Confirm:$false -ErrorAction Stop
        return $true
    } catch {
        if (-not $item.PSIsContainer) {
            Write-Warn "Skipping locked or inaccessible path '$fullPath': $($_.Exception.Message)"
            return $false
        }

        $enumerationErrors = @()
        $entries = @(Get-ChildItem -LiteralPath $item.FullName -Force -Recurse -ErrorAction SilentlyContinue -ErrorVariable enumerationErrors |
            Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
        $success = $enumerationErrors.Count -eq 0
        foreach ($enumerationError in $enumerationErrors) {
            Write-Warn "Skipping inaccessible path below '$fullPath': $($enumerationError.Exception.Message)"
        }
        foreach ($entry in $entries) {
            try {
                Remove-Item -LiteralPath $entry.FullName -Force -Confirm:$false -ErrorAction Stop
            } catch {
                $success = $false
                Write-Warn "Skipping locked or inaccessible path '$($entry.FullName)': $($_.Exception.Message)"
            }
        }
        if (Test-Path -LiteralPath $item.FullName -ErrorAction SilentlyContinue) {
            try {
                Remove-Item -LiteralPath $item.FullName -Force -Confirm:$false -ErrorAction Stop
            } catch {
                $success = $false
                Write-Warn "Skipping locked or inaccessible path '$($item.FullName)': $($_.Exception.Message)"
            }
        }
        return $success -and -not (Test-Path -LiteralPath $item.FullName -ErrorAction SilentlyContinue)
    }
}

function Remove-RepoDirectoryContents([string]$Path) {
    $repoPrefix = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\') + '\'
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove contents outside the repository: $fullPath"
    }
    try {
        if (-not (Test-Path -LiteralPath $fullPath -PathType Container)) {
            return
        }

        $enumerationErrors = @()
        $items = @(Get-ChildItem -LiteralPath $fullPath -Force -ErrorAction SilentlyContinue -ErrorVariable enumerationErrors |
            Sort-Object @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
        foreach ($enumerationError in $enumerationErrors) {
            Write-Warn "Skipping inaccessible cache contents below '$fullPath': $($enumerationError.Exception.Message)"
        }
        $progressId = Start-LauncherProgress -Activity "ADSMOD: remove repository contents" -Status "0 of $($items.Count) items"
        try {
            for ($index = 0; $index -lt $items.Count; $index++) {
                $item = $items[$index]
                $percent = if ($items.Count -eq 0) { 100 } else { [int](($index + 1) * 100 / $items.Count) }
                Update-LauncherProgress -Id $progressId -Activity "ADSMOD: remove repository contents" -Status "$($index + 1) of $($items.Count): $($item.Name)" -PercentComplete $percent
                [void](Remove-RepoPath $item.FullName)
            }
        } finally {
            Complete-LauncherProgress $progressId
        }
    } catch {
        Write-Warn "Skipping inaccessible cache contents under '$fullPath': $($_.Exception.Message)"
    }
}

function Remove-ResourcePath([string]$Path) {
    $resourcePrefix = [System.IO.Path]::GetFullPath($ResourcesDir).TrimEnd('\') + '\'
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($resourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the selected resource directory: $fullPath"
    }
    try {
        $item = Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop
    } catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound) {
            return $true
        }
        Write-Warn "Skipping inaccessible user-data path '$fullPath': $($_.Exception.Message)"
        return $false
    }
    try {
        Remove-Item -LiteralPath $item.FullName -Recurse -Force -Confirm:$false -ErrorAction Stop
        return $true
    } catch {
        if (-not $item.PSIsContainer) {
            Write-Warn "Skipping locked or inaccessible user-data path '$fullPath': $($_.Exception.Message)"
            return $false
        }

        $enumerationErrors = @()
        $entries = @(Get-ChildItem -LiteralPath $item.FullName -Force -Recurse -ErrorAction SilentlyContinue -ErrorVariable enumerationErrors |
            Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
        $success = $enumerationErrors.Count -eq 0
        foreach ($enumerationError in $enumerationErrors) {
            Write-Warn "Skipping inaccessible user-data path below '$fullPath': $($enumerationError.Exception.Message)"
        }
        foreach ($entry in $entries) {
            try {
                Remove-Item -LiteralPath $entry.FullName -Force -Confirm:$false -ErrorAction Stop
            } catch {
                $success = $false
                Write-Warn "Skipping locked or inaccessible user-data path '$($entry.FullName)': $($_.Exception.Message)"
            }
        }
        if (Test-Path -LiteralPath $item.FullName -ErrorAction SilentlyContinue) {
            try {
                Remove-Item -LiteralPath $item.FullName -Force -Confirm:$false -ErrorAction Stop
            } catch {
                $success = $false
                Write-Warn "Skipping locked or inaccessible user-data path '$($item.FullName)': $($_.Exception.Message)"
            }
        }
        return $success -and -not (Test-Path -LiteralPath $item.FullName -ErrorAction SilentlyContinue)
    }
}

function Remove-ResourceDirectoryContents([string]$Path) {
    $resourcePrefix = [System.IO.Path]::GetFullPath($ResourcesDir).TrimEnd('\') + '\'
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($resourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove resource contents outside the selected resource directory: $fullPath"
    }
    if (-not (Test-Path -LiteralPath $fullPath -PathType Container)) {
        return
    }

    $enumerationErrors = @()
    $items = @(Get-ChildItem -LiteralPath $fullPath -Force -ErrorAction SilentlyContinue -ErrorVariable enumerationErrors |
        Sort-Object @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    foreach ($enumerationError in $enumerationErrors) {
        Write-Warn "Skipping inaccessible resource contents below '$fullPath': $($enumerationError.Exception.Message)"
    }
    $progressId = Start-LauncherProgress -Activity "ADSMOD: remove resource contents" -Status "0 of $($items.Count) items"
    try {
        for ($index = 0; $index -lt $items.Count; $index++) {
            $item = $items[$index]
            if ($item.Name -eq '.gitkeep') { continue }
            $percent = if ($items.Count -eq 0) { 100 } else { [int](($index + 1) * 100 / $items.Count) }
            Update-LauncherProgress -Id $progressId -Activity "ADSMOD: remove resource contents" -Status "$($index + 1) of $($items.Count): $($item.Name)" -PercentComplete $percent
            [void](Remove-ResourcePath $item.FullName)
        }
    } finally {
        Complete-LauncherProgress $progressId
    }
}

# -----------------------------------------------------------------------------
# Portable runtimes, dependencies, and application startup
# -----------------------------------------------------------------------------

function Remove-UvCache {
    $expectedCachePath = [System.IO.Path]::GetFullPath($RuntimeCacheDir)
    $actualCachePath = [System.IO.Path]::GetFullPath($UvCacheDir)
    if ($actualCachePath -ne $expectedCachePath) {
        throw "Refusing to remove an unexpected uv cache path: $actualCachePath"
    }
    Remove-RepoDirectoryContents $actualCachePath
}

function Download-AndExtract {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][uri]$Url,
        [Parameter(Mandatory)][string]$ArchivePath,
        [Parameter(Mandatory)][string]$DestinationPath,
        [switch]$FlattenSingleDirectory
    )
    $previousProgressPreference = $ProgressPreference
    $activity = "ADSMOD: download and extract $([IO.Path]::GetFileName($ArchivePath))"
    $progressId = Start-LauncherProgress -Activity $activity -Status "Downloading $Url"
    try {
        $ProgressPreference = 'SilentlyContinue'
        New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
        Invoke-WebRequest -Uri $Url -OutFile $ArchivePath
        $ProgressPreference = $previousProgressPreference
        Update-LauncherProgress -Id $progressId -Activity $activity -Status 'Extracting archive'
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $DestinationPath -Force
        if ($FlattenSingleDirectory) {
            Update-LauncherProgress -Id $progressId -Activity $activity -Status 'Flattening extracted directory'
            $children = @(Get-ChildItem -LiteralPath $DestinationPath -Force)
            if ($children.Count -eq 1 -and $children[0].PSIsContainer) {
                $nestedRoot = $children[0].FullName
                Get-ChildItem -LiteralPath $nestedRoot -Force | Move-Item -Destination $DestinationPath -Force
                Remove-Item -LiteralPath $nestedRoot -Force -ErrorAction SilentlyContinue
            }
        }
    } finally {
        $ProgressPreference = $previousProgressPreference
        Remove-Item -LiteralPath $ArchivePath -Force -ErrorAction SilentlyContinue
        Complete-LauncherProgress $progressId
    }
}

function Patch-PythonPth {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Python path configuration was not found: $Path"
    }
    $content = Get-Content -LiteralPath $Path -Raw
    if ($content -notmatch '(?m)^python314\.zip\s*$') {
        $content = "python314.zip`r`n$content"
    }
    if ($content -match '(?m)^#import site\s*$') {
        $content = $content -replace '(?m)^#import site\s*$', 'import site'
    }
    # The embedded interpreter rejects a UTF-8 BOM before the first ._pth path.
    Set-Content -LiteralPath $Path -Value $content -Encoding ascii
}

function Get-PythonVersion {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$PythonExe)
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        throw "Python executable was not found: $PythonExe"
    }
    $version = & $PythonExe -c "import platform; print(platform.python_version())"
    if ($LASTEXITCODE -ne 0 -or -not $version) {
        throw "Could not determine the Python version from $PythonExe."
    }
    $version
}

function Move-UvExe {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$SearchRoot,
        [Parameter(Mandatory)][string]$DestinationPath
    )
    $uv = Get-ChildItem -LiteralPath $SearchRoot -Recurse -Filter 'uv.exe' -File | Select-Object -First 1
    if (-not $uv) {
        throw "uv.exe was not found under $SearchRoot after extraction."
    }
    if ($uv.FullName -ne [System.IO.Path]::GetFullPath($DestinationPath)) {
        Copy-Item -LiteralPath $uv.FullName -Destination $DestinationPath -Force
    }
}

function Wait-ForHealth {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][uri]$Url,
        [ValidateRange(1, 600)][int]$TimeoutSeconds = 60
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $activity = "ADSMOD: wait for health $Url"
    $progressId = Start-LauncherProgress -Activity $activity -Status "Waiting up to $TimeoutSeconds seconds"
    try {
        do {
            $elapsed = [int](([DateTime]::UtcNow - $deadline.AddSeconds(-$TimeoutSeconds)).TotalSeconds)
            Update-LauncherProgress -Id $progressId -Activity $activity -Status "Waiting for healthy response; ${elapsed}s elapsed"
            try {
                $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
                if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 300) {
                    return
                }
            } catch {
                if ([DateTime]::UtcNow -ge $deadline) {
                    throw "No healthy response from $Url within $TimeoutSeconds seconds."
                }
            }
            Start-Sleep -Seconds 1
        } while ([DateTime]::UtcNow -lt $deadline)
        throw "No healthy response from $Url within $TimeoutSeconds seconds."
    } finally {
        Complete-LauncherProgress $progressId
    }
}

function Import-Settings {
    if (-not (Test-Path -LiteralPath $ConfigFile)) {
        throw "Missing canonical configuration: $ConfigFile"
    }
    $canonical = Get-Content -LiteralPath $ConfigFile -Raw | ConvertFrom-Json
    if (-not $canonical.runtime -or -not $canonical.storage) {
        throw "Canonical configuration is missing runtime or storage settings: $ConfigFile"
    }
    $mode = [string]$canonical.runtime.mode
    if ($mode -notin @('core', 'core-ml')) {
        throw "runtime.mode must be core or core-ml."
    }
    $host = [string]$canonical.runtime.host
    if ([string]::IsNullOrWhiteSpace($host)) {
        throw "runtime.host must be configured."
    }
    $script:ResourcesDir = Resolve-CanonicalPath ([string]$canonical.storage.root)
    $script:LogDir = Join-Path $script:ResourcesDir "logs"
    $script:CheckpointsDir = Join-Path $script:ResourcesDir "checkpoints"
    return [pscustomobject]@{
        Mode = $mode
        Host = $host
        CorePort = [int]$canonical.runtime.core_port
        MlPort = [int]$canonical.runtime.ml_port
        FrontendPort = [int]$canonical.runtime.frontend_port
        MlRestartAttempts = [int]$canonical.runtime.ml_restart_attempts
    }
}

function Set-RuntimeEnvironment {
    foreach ($directory in @(
        $RuntimeCacheDir,
        $RuntimeTempDir,
        $TestCacheDir,
        $PytestCacheDir,
        $PytestTempDir,
        $RuffCacheDir,
        $PythonCacheDir,
        $MypyCacheDir,
        $AngularCacheDir
    )) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    $env:UV_CACHE_DIR = $UvCacheDir
    Remove-Item Env:UV_NO_CACHE -ErrorAction SilentlyContinue
    $env:PIP_CACHE_DIR = Join-Path $RuntimeCacheDir "pip"
    $env:npm_config_cache = Join-Path $RuntimeCacheDir "npm"
    $env:XDG_CACHE_HOME = $RuntimeCacheDir
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $BackendDir ".venv"
    $env:PYTHONPYCACHEPREFIX = $PythonCacheDir
    $env:PYTEST_CACHE_DIR = $PytestCacheDir
    $env:RUFF_CACHE_DIR = $RuffCacheDir
    $env:MYPY_CACHE_DIR = $MypyCacheDir
    $env:COVERAGE_FILE = Join-Path $TestCacheDir ".coverage"
    $env:TEMP = $RuntimeTempDir
    $env:TMP = $RuntimeTempDir
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue
    $env:PATH = "$NodeDir;$UvDir;$env:PATH"
}

function Initialize-NodeRuntime {
    Write-Step "Ensuring portable Node.js $NodeVersion"
    New-Item -ItemType Directory -Path $NodeDir -Force | Out-Null
    $nodeNeedsInstall = -not (Test-Path -LiteralPath $NodeExe) -or -not (Test-Path -LiteralPath $NpmCmd)
    if (-not $nodeNeedsInstall) {
        $existingNodeVersion = (& $NodeExe --version).Trim()
        $nodeNeedsInstall = $LASTEXITCODE -ne 0 -or $existingNodeVersion -ne "v$NodeVersion"
        if ($nodeNeedsInstall) {
            Write-Warn "Replacing portable Node.js $existingNodeVersion with v$NodeVersion."
            Remove-RepoPath $NodeDir
        }
    }
    if ($nodeNeedsInstall) {
        Download-AndExtract `
            -Url $NodeUrl `
            -ArchivePath (Join-Path $NodeDir $NodeArchive) `
            -DestinationPath $NodeDir `
            -FlattenSingleDirectory
    }
    if (-not (Test-Path -LiteralPath $NodeExe) -or -not (Test-Path -LiteralPath $NpmCmd)) {
        throw "Portable Node.js extraction did not produce node.exe and npm.cmd in $NodeDir."
    }
    $nodeVersionOutput = & $NodeExe --version
    Assert-LastExitCode "Node.js version check"
    Write-Ok "Node.js ready: $nodeVersionOutput"
}

function Initialize-Runtimes {
    Write-Step "Ensuring portable runtime directories"
    foreach ($directory in @($RuntimesDir, $PythonDir, $UvDir, $NodeDir)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }

    Write-Step "Ensuring portable Python $PythonVersion"
    if (-not (Test-Path -LiteralPath $PythonExe)) {
        Download-AndExtract `
            -Url $PythonUrl `
            -ArchivePath (Join-Path $PythonDir $PythonArchive) `
            -DestinationPath $PythonDir
    }
    Patch-PythonPth -Path $PythonPth
    $detectedPython = Get-PythonVersion -PythonExe $PythonExe
    Write-Ok "Python ready: $detectedPython"

    Write-Step "Ensuring portable uv"
    if (-not (Test-Path -LiteralPath $UvExe)) {
        Download-AndExtract `
            -Url $UvUrl `
            -ArchivePath (Join-Path $UvDir "uv.zip") `
            -DestinationPath $UvDir
        Move-UvExe -SearchRoot $UvDir -DestinationPath $UvExe
    }
    $uvVersion = & $UvExe --version
    Assert-LastExitCode "uv version check"
    Write-Ok $uvVersion

    Initialize-NodeRuntime
    Set-RuntimeEnvironment
}

function Sync-FrontendDependencies {
    param(
        [switch]$BuildFrontend
    )

    Write-Step "Installing frontend dependencies"
    Push-Location $ClientDir
    try {
        if (-not (Test-Path -LiteralPath (Join-Path $ClientDir 'package-lock.json'))) {
            throw "Missing frontend lockfile: $(Join-Path $ClientDir 'package-lock.json')"
        }
        & $NpmCmd ci
        Assert-LastExitCode "npm dependency installation"

        if ($BuildFrontend) {
            Write-Step "Building frontend"
            & $NpmCmd run build
            Assert-LastExitCode "frontend build"
        }
    } finally {
        Pop-Location
    }
    Write-Ok "Frontend dependencies are ready."
}

function Sync-Dependencies {
    param(
        [switch]$BuildFrontend,
        [switch]$RuntimesReady,
        [ValidateSet('Standard', 'Development')]
        [string]$InstallationType = 'Standard'
    )

    $settings = Import-Settings
    if (-not $RuntimesReady) {
        Initialize-Runtimes
    }
    Set-RuntimeEnvironment

    Write-Step "Syncing Python dependencies"
    Push-Location $BackendDir
    try {
        $arguments = @('sync', '--locked', '--all-packages', '--python', $PythonExe)
        if ($InstallationType -eq 'Development') {
            $arguments += '--group', 'dev'
        }
        else {
            $arguments += '--no-dev'
        }
        & $UvExe @arguments
        Assert-LastExitCode "uv sync"
    } finally {
        Pop-Location
    }
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "Backend virtual-environment Python was not created at $VenvPython."
    }
    Write-Ok "Python dependencies are ready."

    Sync-FrontendDependencies `
        -BuildFrontend:$BuildFrontend
}

function Test-DependenciesReady {
    $frontendPackage = Join-Path $ClientDir 'package.json'
    $frontendLock = Join-Path $ClientDir 'package-lock.json'
    $frontendModules = Join-Path $ClientDir 'node_modules'
    $frontendInstallState = Join-Path $frontendModules '.package-lock.json'
    $frontendRunner = Join-Path $frontendModules '@angular/cli/bin/ng.js'
    $frontendBuild = Join-Path $ClientDir 'dist\browser\index.html'
    $backendEntrypoint = Join-Path $BackendDir 'core/src/adsmod_core/cli.py'
    $backendLock = Join-Path $BackendDir 'uv.lock'

    if (-not (Test-Path -LiteralPath $PythonExe) -or
        -not (Test-Path -LiteralPath $UvExe) -or
        -not (Test-Path -LiteralPath $NodeExe) -or
        -not (Test-Path -LiteralPath $NpmCmd) -or
        -not (Test-Path -LiteralPath $VenvPython) -or
        -not (Test-Path -LiteralPath $backendLock) -or
        -not (Test-Path -LiteralPath $backendEntrypoint) -or
        -not (Test-Path -LiteralPath $frontendPackage) -or
        -not (Test-Path -LiteralPath $frontendLock) -or
        -not (Test-Path -LiteralPath $frontendInstallState) -or
        -not (Test-Path -LiteralPath $frontendRunner) -or
        -not (Test-Path -LiteralPath $frontendBuild -PathType Leaf)) {
        return $false
    }

    & $PythonExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $UvExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $NodeExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $VenvPython -c 'import adsmod_core.app, fastapi, uvicorn' *> $null
    if ($LASTEXITCODE -ne 0) { return $false }

    return $true
}

function Stop-ListenerOnPort([int]$Port) {
    $lines = netstat -ano | Select-String -Pattern ":$Port\s+.*LISTENING\s+(\d+)\s*$"
    $processIds = @($lines | ForEach-Object {
        if ($_.Matches.Count -gt 0) { [int]$_.Matches[0].Groups[1].Value }
    } | Sort-Object -Unique)
    foreach ($processId in $processIds) {
        if ($processId -eq $PID) {
            throw "Refusing to terminate the launcher process on port $Port."
        }
        Write-Warn "Stopping process $processId on port $Port."
        & taskkill.exe /PID $processId /T /F | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stop process $processId on port $Port."
        }
    }
}

function Get-ListenerPid([int]$Port) {
    $match = netstat -ano | Select-String -Pattern ":$Port\s+.*LISTENING\s+(\d+)\s*$" | Select-Object -First 1
    if ($match -and $match.Matches.Count -gt 0) {
        return [int]$match.Matches[0].Groups[1].Value
    }
    return $null
}

function Start-Application {
    $settings = Import-Settings
    Set-RuntimeEnvironment
    if (-not (Test-DependenciesReady)) {
        Write-Step "Required application environments or frontend build output are missing or unusable; repairing dependencies and frontend build."
        Sync-Dependencies -BuildFrontend
    } else {
        Write-Ok "Application environments are ready; skipped dependency installation."
    }
    Set-RuntimeEnvironment

    $backendPort = $settings.CorePort
    $uiPort = $settings.FrontendPort
    Stop-ListenerOnPort -Port $backendPort
    Stop-ListenerOnPort -Port $uiPort
    if ($settings.Mode -eq 'core-ml') {
        Stop-ListenerOnPort -Port $settings.MlPort
    }

    $backendArguments = @(
        '-m', 'adsmod_core.cli',
        '--config', $ConfigFile
    )
    Write-Step "Starting Core service"
    $backendProcess = Start-Process -FilePath $VenvPython `
        -ArgumentList $backendArguments `
        -WorkingDirectory $RepoRoot `
        -WindowStyle Hidden `
        -PassThru

    $healthUrl = "http://$($settings.Host):$($settings.CorePort)/health/ready"
    Write-Step "Waiting for Core readiness at $healthUrl"
    $mlProcess = $null
    try {
        Wait-ForHealth -Url $healthUrl -TimeoutSeconds 60
    } catch {
        if ($backendProcess -and -not $backendProcess.HasExited) {
            Stop-Process -Id $backendProcess.Id -Force
        }
        throw
    }
    $backendPid = Get-ListenerPid -Port $backendPort

    $mlHealthUrl = $null
    if ($settings.Mode -eq 'core-ml') {
        $mlArguments = @(
            '-m', 'adsmod_ml.cli',
            '--config', $ConfigFile
        )
        Write-Step "Starting ML service"
        $mlProcess = Start-Process -FilePath $VenvPython `
            -ArgumentList $mlArguments `
            -WorkingDirectory $RepoRoot `
            -WindowStyle Hidden `
            -PassThru
        $mlHealthUrl = "http://$($settings.Host):$($settings.MlPort)/health/ready"
        Write-Step "Waiting for ML readiness at $mlHealthUrl"
        try {
            Wait-ForHealth -Url $mlHealthUrl -TimeoutSeconds 60
        } catch {
            if ($mlProcess -and -not $mlProcess.HasExited) { Stop-Process -Id $mlProcess.Id -Force }
            if ($backendProcess -and -not $backendProcess.HasExited) { Stop-Process -Id $backendProcess.Id -Force }
            throw
        }
    }

    Write-Step "Starting frontend preview"
    $frontendProcess = Start-Process -FilePath $NpmCmd `
        -ArgumentList @('run', 'preview', '--', '--host', $settings.Host, '--port', $settings.FrontendPort) `
        -WorkingDirectory $ClientDir `
        -WindowStyle Hidden `
        -PassThru

    $frontendUrl = "http://$($settings.Host):$($settings.FrontendPort)"
    try {
        Wait-ForHealth -Url $frontendUrl -TimeoutSeconds 60
    } catch {
        if (-not $frontendProcess.HasExited) { Stop-Process -Id $frontendProcess.Id -Force }
        if ($mlProcess -and -not $mlProcess.HasExited) { Stop-Process -Id $mlProcess.Id -Force }
        if ($backendProcess -and -not $backendProcess.HasExited) { Stop-Process -Id $backendProcess.Id -Force }
        throw
    }
    $frontendPid = Get-ListenerPid -Port $uiPort

    Start-Process $frontendUrl
    Write-Host ""
    Write-Ok "ADSMOD started successfully."
    Write-Host "Backend: $healthUrl (PID $backendPid)" -ForegroundColor Green
    if ($mlHealthUrl) {
        $mlPid = Get-ListenerPid -Port $settings.MlPort
        Write-Host "ML: $mlHealthUrl (PID $mlPid)" -ForegroundColor Green
    }
    Write-Host "Frontend: $frontendUrl (PID $frontendPid)" -ForegroundColor Green
}

function Install-UpdateDependencies {
    Initialize-Runtimes
    Write-Ok "Portable runtimes ready."
    $installationType = Read-InstallationType
    Sync-Dependencies -BuildFrontend -RuntimesReady -InstallationType $installationType
    Remove-UvCache
    Write-Ok "Dependencies installed and frontend built successfully."
}

function Rebuild-Frontend {
    Initialize-NodeRuntime
    Set-RuntimeEnvironment
    Sync-FrontendDependencies -BuildFrontend
    Write-Ok "Frontend rebuilt successfully."
}

function Read-InstallationType {
    Write-Host "  [1] Development - include Ruff, Pyright, and pytest"
    Write-Host "  [2] Standard    - install runtime dependencies only"
    $selection = (Read-Host "  Select installation profile [1-2]").Trim()
    switch ($selection) {
        '1' { return 'Development' }
        '2' { return 'Standard' }
        default { throw "Invalid installation profile. Enter 1 for Development or 2 for Standard." }
    }
}

function Initialize-Database {
    Import-Settings | Out-Null
    Initialize-Runtimes
    Set-RuntimeEnvironment
    Write-Step "Initializing database"
    Push-Location $RepoRoot
    try {
        & $UvExe run --project $BackendDir --python $PythonExe python app/scripts/initialize_database.py --config $ConfigFile
        Assert-LastExitCode "database initialization"
    } finally {
        Pop-Location
    }
    Write-Ok "Database initialized successfully."
}

function Invoke-TestSuite {
    Import-Settings | Out-Null
    $testScript = Join-Path $TestsDir "run_tests.bat"
    if (-not (Test-Path -LiteralPath $testScript)) {
        throw "Missing test runner: $testScript"
    }
    Write-Step "Running test suite"
    & $testScript
    Assert-LastExitCode "test suite"
    Write-Ok "Test suite completed successfully."
}

# -----------------------------------------------------------------------------
# Data and maintenance actions
# -----------------------------------------------------------------------------

function Remove-Logs {
    Import-Settings | Out-Null
    Write-Step "Removing log files"
    if (Test-Path -LiteralPath $LogDir) {
        Get-ChildItem -LiteralPath $LogDir -Filter '*.log' -File -ErrorAction SilentlyContinue |
            Sort-Object @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false } |
            ForEach-Object {
                try {
                    Remove-Item -LiteralPath $_.FullName -Force -Confirm:$false -ErrorAction Stop
                } catch {
                    Write-Warn "Skipping locked or inaccessible log '$($_.FullName)': $($_.Exception.Message)"
                }
            }
    }
    Write-Ok "Log files removed."
}

function Clear-Cache {
    Write-Step "Clearing runtime and test-tool caches"
    foreach ($cacheDirectory in @($RuntimeCacheDir, $TestCacheDir)) {
        New-Item -ItemType Directory -Path $cacheDirectory -Force | Out-Null
        Remove-RepoDirectoryContents $cacheDirectory
    }

    $legacyCacheNames = @('__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache')
    $legacyCacheDirectories = @(
        @(Get-ChildItem -LiteralPath $RepoRoot -Directory -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in $legacyCacheNames })
        @(Get-ChildItem -LiteralPath $AppDir -Directory -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in $legacyCacheNames })
        @(Get-ChildItem -LiteralPath $BackendDir -Directory -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in $legacyCacheNames })
    )
    $legacySearchRoots = @(
        (Join-Path $AppDir 'scripts'),
        (Join-Path $BackendDir 'common'),
        (Join-Path $BackendDir 'core'),
        (Join-Path $BackendDir 'ml'),
        $TestsDir
    )
    foreach ($searchRoot in $legacySearchRoots) {
        $legacyCacheDirectories += @(Get-ChildItem -LiteralPath $searchRoot -Directory -Recurse -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in $legacyCacheNames -and
                $_.FullName -ne [System.IO.Path]::GetFullPath($RuntimeCacheDir) -and
                $_.FullName -ne [System.IO.Path]::GetFullPath($TestCacheDir) })
    }
    $legacyCacheDirectories = @($legacyCacheDirectories |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false } -Unique)
    foreach ($legacyCacheDirectory in $legacyCacheDirectories) {
        [void](Remove-RepoPath $legacyCacheDirectory.FullName)
    }

    $legacyToolCache = Join-Path $ClientDir '.angular'
    [void](Remove-RepoPath $legacyToolCache)

    Write-Ok "Cache cleanup completed; locked or inaccessible items were skipped."
}

function Uninstall-Application {
    Write-Step "Removing local application runtimes and build artifacts"
    $runtimeContents = @()
    if (Test-Path -LiteralPath $RuntimesDir) {
        $runtimeContents = @(Get-ChildItem -LiteralPath $RuntimesDir -Force |
            Where-Object { $_.Name -ne '.gitkeep' } |
            Sort-Object @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false } |
            Select-Object -ExpandProperty FullName)
    }
    $paths = @($runtimeContents) + @(
        (Join-Path $BackendDir '.venv'),
        $StartupTempDir,
        (Join-Path $RepoRoot '.venv'),
        (Join-Path $ClientDir 'node_modules'),
        (Join-Path $ClientDir '.angular'),
        (Join-Path $ClientDir 'dist')
    )
    foreach ($path in $paths) {
        Remove-RepoPath $path
    }
    Get-ChildItem -LiteralPath $RepoRoot -Directory -Filter '__pycache__' -Recurse -Force -ErrorAction SilentlyContinue |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false } |
        ForEach-Object { Remove-RepoPath $_.FullName }
    Write-Ok "Application runtimes, dependencies, and build outputs removed. Dependency lockfiles and user data were preserved."
}

function Get-ConfiguredDatabasePath {
    $canonical = Get-Content -LiteralPath $ConfigFile -Raw | ConvertFrom-Json
    $database = $canonical.application.database
    if (-not $database.embedded_database) {
        return $null
    }

    $configuredPath = [string]$database.sqlite_path
    if ([string]::IsNullOrWhiteSpace($configuredPath)) {
        throw "Embedded database configuration is missing application.database.sqlite_path."
    }

    $expandedPath = [Environment]::ExpandEnvironmentVariables($configuredPath.Trim())
    if ([System.IO.Path]::IsPathRooted($expandedPath)) {
        return [System.IO.Path]::GetFullPath($expandedPath)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ResourcesDir $expandedPath))
}

function Remove-DatabaseFiles {
    $databasePath = Get-ConfiguredDatabasePath
    if ($null -eq $databasePath) {
        Write-Warn "An external database is configured; local data was cleared, but the external database was not modified."
        return
    }

    $protectedPaths = @(
        [System.IO.Path]::GetFullPath($ConfigFile),
        [System.IO.Path]::GetFullPath((Join-Path $DefaultResourcesDir 'adsmod.schema.json'))
    )
    foreach ($path in @($databasePath, "$databasePath-wal", "$databasePath-shm")) {
        $fullPath = [System.IO.Path]::GetFullPath($path)
        if ($fullPath -in $protectedPaths) {
            throw "Refusing to remove an application configuration file: $fullPath"
        }
        try {
            if (Test-Path -LiteralPath $fullPath) {
                Remove-Item -LiteralPath $fullPath -Force -ErrorAction Stop
            }
        } catch {
            Write-Warn "Skipping locked or inaccessible database file '$fullPath': $($_.Exception.Message)"
        }
    }
}

function Clear-CheckpointFiles {
    Write-Step "Removing saved checkpoints"
    Remove-ResourceDirectoryContents -Path $CheckpointsDir
    Write-Ok "Saved checkpoints removed."
}

function Remove-Checkpoints {
    Import-Settings | Out-Null
    Write-Warn "This removes all saved training checkpoints."
    $confirmation = ([string](Read-Host "Continue removing all saved training checkpoints? [y/N]")).Trim()
    if ($confirmation -notmatch '^(?i:y|yes)$') {
        Write-Warn "Remove Checkpoints cancelled."
        return
    }
    Clear-CheckpointFiles
}

function Remove-All-Data {
    Import-Settings | Out-Null
    Write-Warn "This removes the local database, uploaded dataset records, saved checkpoints, and generated logs."
    $confirmation = ([string](Read-Host "Continue removing all local user-generated data? [y/N]")).Trim()
    if ($confirmation -notmatch '^(?i:y|yes)$') {
        Write-Warn "Remove All Data cancelled."
        return
    }

    Write-Step "Removing local user-generated data"
    Remove-DatabaseFiles
    Clear-CheckpointFiles
    Remove-ResourceDirectoryContents -Path $LogDir
    Write-Ok "All local user-generated data was removed; application files and settings were preserved."
}

# -----------------------------------------------------------------------------
# Repository update actions
# -----------------------------------------------------------------------------

function Invoke-Git {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$Arguments)

    $display = "git $($Arguments -join ' ')"
    Write-Step $display
    Push-Location $RepoRoot
    try {
        & git @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $exitCode."
    }
    Write-Ok "Completed $display"
}

function Get-GitText {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$Arguments)

    Push-Location $RepoRoot
    try {
        $output = @(& git @Arguments 2>&1)
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $exitCode."
    }
    return (($output | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine).Trim()
}

function Get-GitExitCode {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$Arguments)

    Push-Location $RepoRoot
    try {
        & git @Arguments *> $null
        return $LASTEXITCODE
    } finally {
        Pop-Location
    }
}

function Get-WorkingTreeChanges {
    $status = Get-GitText -Arguments @('status', '--porcelain')
    if ([string]::IsNullOrWhiteSpace($status)) {
        return @()
    }
    return @($status -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Update-Application {
    $changes = @(Get-WorkingTreeChanges)
    if ($changes.Count -gt 0) {
        throw "Update requires a clean working tree. Commit or stash the existing changes before updating from main."
    }

    $currentBranch = Get-GitText -Arguments @('branch', '--show-current')
    if ($currentBranch -ne 'main') {
        throw "Update requires the main branch to be checked out; current branch is '$currentBranch'. No files were changed."
    }

    Write-Step "Pulling the latest application version from main"
    Invoke-Git -Arguments @('pull', '--ff-only', 'origin', 'main')
    Write-Ok "Application update completed from main."
}

function Check-For-Updates {
    try {
        $localMainCommit = Get-GitText -Arguments @('rev-parse', '--verify', 'refs/heads/main')
        $remoteLine = Get-GitText -Arguments @('ls-remote', '--exit-code', 'origin', 'refs/heads/main')
        $remoteMainCommit = ($remoteLine -split '\s+')[0]
        if ($remoteMainCommit -notmatch '^[0-9a-f]{40}$') {
            throw "The remote main commit could not be read."
        }

        if ($localMainCommit -eq $remoteMainCommit) {
            Write-Ok "No update available. Local main is up to date with origin/main."
            return
        }

        if ((Get-GitExitCode -Arguments @('merge-base', '--is-ancestor', $localMainCommit, $remoteMainCommit)) -eq 0) {
            Write-Warn "A newer version is available on origin/main. No files were downloaded or changed."
            return
        }

        if ((Get-GitExitCode -Arguments @('merge-base', '--is-ancestor', $remoteMainCommit, $localMainCommit)) -eq 0) {
            Write-Ok "No newer version is available; local main is ahead of origin/main. No files were changed."
            return
        }

        Write-Warn "Local main and origin/main have diverged. Review the branches before updating; no files were changed."
    } catch {
        Write-Warn "Could not check for updates: $($_.Exception.Message)"
    }
}

function Wait-ForMenu {
    Write-Host ""
    Write-Host "Press any key to return to menu..." -ForegroundColor DarkGray
    [Console]::ReadKey($true) | Out-Null
}

function Write-MenuItem {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Number,
        [Parameter(Mandatory)][string]$Label,
        [string]$Hint,
        [ConsoleColor]$NumberColor = [ConsoleColor]::Cyan
    )
    Write-Host ("  [{0}] " -f $Number) -ForegroundColor $NumberColor -NoNewline
    Write-Host $Label -NoNewline
    if ($Hint) {
        Write-Host ("  {0}" -f $Hint) -ForegroundColor DarkGray
    } else {
        Write-Host ""
    }
}

function Show-MainMenu {
    try { Clear-Host } catch { }

    $menuWidth = 62
    $rule = "=" * $menuWidth
    $subtleRule = "-" * $menuWidth

    Write-Host ""
    Write-Host $rule -ForegroundColor DarkCyan
    Write-Host "  ADSMOD" -ForegroundColor Cyan -NoNewline
    Write-Host "  |  Adsorption Modeling" -ForegroundColor White
    Write-Host "  Local workspace launcher and maintenance console" -ForegroundColor DarkGray
    Write-Host $rule -ForegroundColor DarkCyan
    Write-Host ""
    Write-Host "  APPLICATION" -ForegroundColor DarkCyan
    Write-MenuItem -Number '1' -Label 'Launch Application' -Hint 'Start the local web workspace'
    Write-MenuItem -Number '2' -Label 'Initialize Database' -Hint 'Create or upgrade the Alembic-managed data store'
    Write-Host ""
    Write-Host "  DEVELOPMENT AND SETUP" -ForegroundColor DarkCyan
    Write-MenuItem -Number '3' -Label 'Install / Update Dependencies' -Hint 'Refresh local runtimes and packages'
    Write-MenuItem -Number '4' -Label 'Rebuild Frontend' -Hint 'Install frontend packages and rebuild the bundle'
    Write-MenuItem -Number '5' -Label 'Run Test Suite' -Hint 'Execute the repository checks'
    Write-Host ""
    Write-Host "  UPDATES" -ForegroundColor DarkCyan
    Write-MenuItem -Number '6' -Label 'Check for Updates' -Hint 'Report whether origin/main has a newer version'
    Write-MenuItem -Number '7' -Label 'Update' -Hint 'Pull origin/main (clean main branch required)'
    Write-Host ""
    Write-Host "  DATA AND MAINTENANCE" -ForegroundColor DarkCyan
    Write-MenuItem -Number '8' -Label 'Remove Logs' -Hint 'Delete generated log files' -NumberColor Yellow
    Write-MenuItem -Number '9' -Label 'Clear Cache' -Hint 'Remove runtime and test-tool caches' -NumberColor Yellow
    Write-MenuItem -Number '10' -Label 'Remove Checkpoints' -Hint 'Delete saved training checkpoints' -NumberColor Yellow
    Write-MenuItem -Number '11' -Label 'Remove All Data' -Hint 'Delete local database and user-generated files' -NumberColor Yellow
    Write-MenuItem -Number '12' -Label 'Uninstall Application' -Hint 'Remove local runtimes and build artifacts' -NumberColor Yellow
    Write-Host ""
    Write-Host "  EXIT" -ForegroundColor DarkCyan
    Write-Host $subtleRule -ForegroundColor DarkGray
    Write-MenuItem -Number '13' -Label 'Exit' -Hint 'Close this console' -NumberColor DarkGray
    Write-Host $rule -ForegroundColor DarkCyan
}

$exitMenu = $false
while (-not $exitMenu) {
    Show-MainMenu
    $selection = Read-Host "  Select an option [1-13]"

    if ($selection -notmatch '^(?:[1-9]|1[0-3])$') {
        Write-Warn "Please select a number from 1 to 13."
        Wait-ForMenu
        continue
    }

    try {
        Invoke-TrackedLauncherAction -Name "menu option $selection" -Action {
            switch ($selection) {
                '1' {
                    Start-Application
                    exit 0
                }
                '2' { Initialize-Database; Wait-ForMenu }
                '3' { Install-UpdateDependencies; Wait-ForMenu }
                '4' { Rebuild-Frontend; Wait-ForMenu }
                '5' { Invoke-TestSuite; Wait-ForMenu }
                '6' { Check-For-Updates; Wait-ForMenu }
                '7' { Update-Application; Wait-ForMenu }
                '8' { Remove-Logs; Wait-ForMenu }
                '9' { Clear-Cache; Wait-ForMenu }
                '10' { Remove-Checkpoints; Wait-ForMenu }
                '11' { Remove-All-Data; Wait-ForMenu }
                '12' { Uninstall-Application; Wait-ForMenu }
                '13' { $exitMenu = $true }
            }
        }
    } catch {
        Write-Fatal $_.Exception.Message
        if ($selection -eq '1') {
            exit 1
        }
        Wait-ForMenu
    }
}
Clear-LauncherProgress
