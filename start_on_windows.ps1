[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$AppDir = Join-Path $RepoRoot "app"
$BackendDir = Join-Path $AppDir "server"
$ClientDir = Join-Path $AppDir "client"
$TestsDir = Join-Path $AppDir "tests"
$DefaultResourcesDir = Join-Path $AppDir "resources"
$ResourcesDir = $DefaultResourcesDir
$LogDir = Join-Path $ResourcesDir "logs"
$CheckpointsDir = Join-Path $ResourcesDir "checkpoints"
$ConfigFile = Join-Path $ResourcesDir "adsmod.json"
$RuntimesDir = Join-Path $RepoRoot "runtimes"
$PythonDir = Join-Path $RuntimesDir "python"
$UvDir = Join-Path $RuntimesDir "uv"
$NodeDir = Join-Path $RuntimesDir "nodejs"
$RuntimeCacheDir = Join-Path $RuntimesDir "cache"
$UvCacheDir = Join-Path $RuntimeCacheDir "uv"
$PipCacheDir = Join-Path $RuntimeCacheDir "pip"
$NpmCacheDir = Join-Path $RuntimeCacheDir "npm"
$RuntimeTempDir = Join-Path $RuntimeCacheDir "temp"
$PytestCacheDir = Join-Path $RuntimeCacheDir "pytest"
$PytestTempDir = Join-Path $RuntimeCacheDir "pytest-tmp"
$RuffCacheDir = Join-Path $RuntimeCacheDir "ruff"
$PythonCacheDir = Join-Path $RuntimeCacheDir "python"
$MypyCacheDir = Join-Path $RuntimeCacheDir "mypy"
$AngularCacheDir = Join-Path $RuntimeCacheDir "angular"
$PlaywrightCacheDir = Join-Path $RuntimeCacheDir "playwright"
$CoverageDir = Join-Path $RuntimeCacheDir "coverage"
$script:NextProgressId = 1
$script:ActiveProgressActivities = [Collections.Generic.Dictionary[int, string]]::new()
$script:CacheScanErrors = [Collections.Generic.List[string]]::new()
$script:LauncherInteractive = -not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected
$script:BackendProcess = $null
$script:FrontendProcess = $null

$PythonVersion = "3.14.2"
$PythonExe = Join-Path $PythonDir "python.exe"
$PythonPth = Join-Path $PythonDir "python314._pth"
$UvExe = Join-Path $UvDir "uv.exe"
$NodeExe = Join-Path $NodeDir "node.exe"
$NpmCmd = Join-Path $NodeDir "npm.cmd"
$VenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"

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
    $script:ActiveProgressActivities[$id] = $Activity
    if ($script:LauncherInteractive) {
        Write-Progress -Id $id -Activity $Activity -Status $Status
    }
    return $id
}

function Update-LauncherProgress {
    param(
        [Parameter(Mandatory)][int]$Id,
        [Parameter(Mandatory)][string]$Activity,
        [Parameter(Mandatory)][string]$Status,
        [Nullable[int]]$PercentComplete
    )
    if (-not $script:ActiveProgressActivities.ContainsKey($Id)) { return }
    $activity = $script:ActiveProgressActivities[$Id]
    $progress = @{ Id = $Id; Activity = $activity; Status = $Status }
    if ($null -ne $PercentComplete) { $progress.PercentComplete = $PercentComplete }
    if ($script:LauncherInteractive) { Write-Progress @progress }
}

function Complete-LauncherProgress([int]$Id) {
    if ($script:ActiveProgressActivities.ContainsKey($Id)) {
        $activity = $script:ActiveProgressActivities[$Id]
        try {
            if ($script:LauncherInteractive) {
                try { Write-Progress -Id $Id -Activity $activity -Completed } catch { }
            }
        }
        finally {
            [void]$script:ActiveProgressActivities.Remove($Id)
        }
    }
}

function Clear-LauncherProgress {
    foreach ($id in @($script:ActiveProgressActivities.Keys)) {
        Complete-LauncherProgress -Id $id
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
    finally {
        Clear-LauncherProgress
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

function Remove-LauncherPath {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [switch]$KeepRoot,
        [string[]]$PreserveNames = @(),
        [switch]$Strict,
        [switch]$WhatIf,
        [string]$Activity = 'ADSMOD: remove files'
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $removed = [Collections.Generic.List[string]]::new()
    $skipped = [Collections.Generic.List[string]]::new()
    $preserved = [Collections.Generic.List[string]]::new()
    $enumerationErrors = [Collections.Generic.List[string]]::new()
    $result = [ordered]@{
        Target = $fullPath
        Path = $fullPath
        Planned = 0
        PlannedCount = 0
        Removed = 0
        RemovedCount = 0
        RemovedPaths = $removed
        Preserved = 0
        PreservedEntries = $preserved
        Skipped = 0
        SkippedPaths = $skipped
        EnumerationErrors = $enumerationErrors
        WhatIf = [bool]$WhatIf
    }

    try {
        $item = Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop
    }
    catch {
        if ($_.CategoryInfo.Category -eq [System.Management.Automation.ErrorCategory]::ObjectNotFound) {
            return [pscustomobject]$result
        }
        [void]$enumerationErrors.Add("$fullPath ($($_.Exception.Message))")
        Write-Warn "Skipping inaccessible path '$fullPath': $($_.Exception.Message)"
        if ($Strict) { throw }
        return [pscustomobject]$result
    }

    $entries = if ($item.PSIsContainer) {
        $errors = @()
        $found = @(Get-ChildItem -LiteralPath $item.FullName -Force -Recurse -ErrorAction SilentlyContinue -ErrorVariable errors)
        foreach ($errorRecord in $errors) {
            [void]$enumerationErrors.Add("$($errorRecord.Exception.Message)")
            Write-Warn "Skipping inaccessible path below '$fullPath': $($errorRecord.Exception.Message)"
        }
        if (-not $KeepRoot) { $found += $item }
        $found
    }
    else { @($item) }

    $protectedDirectories = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $preservedPaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($entry in @($entries)) {
        if ($entry.Name -in $PreserveNames) {
            [void]$preservedPaths.Add($entry.FullName)
            [void]$preserved.Add($entry.FullName)
            $ancestor = [IO.Path]::GetDirectoryName($entry.FullName)
            while ($ancestor -and $ancestor.StartsWith($item.FullName.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
                [void]$protectedDirectories.Add($ancestor)
                $ancestor = [IO.Path]::GetDirectoryName($ancestor)
            }
        }
    }

    $candidates = @($entries |
        Where-Object { -not $preservedPaths.Contains($_.FullName) -and -not $protectedDirectories.Contains($_.FullName) } |
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true }, @{ Expression = { $_.FullName.ToUpperInvariant() }; Descending = $false })
    $result.Planned = $candidates.Count
    $result.PlannedCount = $candidates.Count
    $result.Preserved = $preserved.Count
    $progressId = $null
    try {
        if ($candidates.Count -gt 0) {
            $progressId = Start-LauncherProgress -Activity $Activity -Status "0 of $($candidates.Count) items"
        }
        for ($index = 0; $index -lt $candidates.Count; $index++) {
            $candidate = $candidates[$index]
            $percent = [int](($index + 1) * 100 / [Math]::Max(1, $candidates.Count))
            if ($null -ne $progressId) {
                Update-LauncherProgress -Id $progressId -Activity $Activity -Status "$($index + 1) of $($candidates.Count): $($candidate.Name)" -PercentComplete $percent
            }
            if ($WhatIf) { continue }
            try {
                Remove-Item -LiteralPath $candidate.FullName -Force -Recurse -Confirm:$false -ErrorAction Stop
                [void]$removed.Add($candidate.FullName)
            }
            catch {
                [void]$skipped.Add("$($candidate.FullName) ($($_.Exception.Message))")
                Write-Warn "Skipping locked or inaccessible path '$($candidate.FullName)': $($_.Exception.Message)"
            }
        }
    }
    finally {
        if ($null -ne $progressId) { Complete-LauncherProgress -Id $progressId }
    }
    $result.Removed = $removed.Count
    $result.RemovedCount = $removed.Count
    $result.Skipped = $skipped.Count

    if ($Strict -and ($skipped.Count -gt 0 -or $enumerationErrors.Count -gt 0)) {
        throw "Removal of '$fullPath' was incomplete. Skipped $($skipped.Count) item(s) and encountered $($enumerationErrors.Count) enumeration error(s)."
    }
    return [pscustomobject]$result
}

function Remove-RepoPath([string]$Path) {
    $repoPrefix = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\') + '\'
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the repository: $fullPath"
    }
    $result = Remove-LauncherPath -Path $fullPath -Activity "ADSMOD: remove $([IO.Path]::GetFileName($fullPath))"
    return $result.Skipped -eq 0 -and $result.EnumerationErrors.Count -eq 0
}

function Remove-RepoDirectoryContents([string]$Path) {
    $repoPrefix = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\') + '\'
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove contents outside the repository: $fullPath"
    }
    if (-not (Test-Path -LiteralPath $fullPath -PathType Container)) { return }
    [void](Remove-LauncherPath -Path $fullPath -KeepRoot -Activity 'ADSMOD: remove repository contents')
}

function Remove-ResourcePath([string]$Path) {
    $resourcePrefix = [System.IO.Path]::GetFullPath($ResourcesDir).TrimEnd('\') + '\'
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($resourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the selected resource directory: $fullPath"
    }
    $result = Remove-LauncherPath -Path $fullPath -PreserveNames @('.gitkeep') -Activity "ADSMOD: remove user data"
    return $result.Skipped -eq 0 -and $result.EnumerationErrors.Count -eq 0
}

function Remove-ResourceDirectoryContents([string]$Path) {
    $resourcePrefix = [System.IO.Path]::GetFullPath($ResourcesDir).TrimEnd('\') + '\'
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($resourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove resource contents outside the selected resource directory: $fullPath"
    }
    if (-not (Test-Path -LiteralPath $fullPath -PathType Container)) { return }
    [void](Remove-LauncherPath -Path $fullPath -KeepRoot -PreserveNames @('.gitkeep') -Activity 'ADSMOD: remove resource contents')
}

# -----------------------------------------------------------------------------
# Portable runtimes, dependencies, and application startup
# -----------------------------------------------------------------------------

function Ensure-CacheLayout {
    foreach ($directory in @(
        $RuntimeCacheDir,
        $UvCacheDir,
        $PipCacheDir,
        $NpmCacheDir,
        $RuntimeTempDir,
        $PytestCacheDir,
        $PytestTempDir,
        $RuffCacheDir,
        $PythonCacheDir,
        $MypyCacheDir,
        $AngularCacheDir,
        $PlaywrightCacheDir,
        $CoverageDir
    )) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
}

function Remove-UvCache {
    $expectedCachePath = [System.IO.Path]::GetFullPath((Join-Path $RuntimeCacheDir 'uv'))
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
    if (-not (Test-Path -LiteralPath $ConfigFile)) { throw "Missing canonical configuration: $ConfigFile" }
    $canonical = Get-Content -LiteralPath $ConfigFile -Raw | ConvertFrom-Json
    if (-not $canonical.runtime -or -not $canonical.storage) { throw "Canonical configuration is missing runtime or storage settings: $ConfigFile" }
    $runtimeHost = [string]$canonical.runtime.host
    if ([string]::IsNullOrWhiteSpace($runtimeHost)) { throw "runtime.host must be configured." }
    $script:ResourcesDir = Resolve-CanonicalPath ([string]$canonical.storage.root)
    $script:LogDir = Join-Path $script:ResourcesDir "logs"
    $script:CheckpointsDir = Join-Path $script:ResourcesDir "checkpoints"
    return [pscustomobject]@{ Host = $runtimeHost; BackendPort = [int]$canonical.runtime.backend_port; FrontendPort = [int]$canonical.runtime.frontend_port }
}

function Set-RuntimeEnvironment {
    Ensure-CacheLayout
    $env:UV_CACHE_DIR = $UvCacheDir
    Remove-Item Env:UV_NO_CACHE -ErrorAction SilentlyContinue
    $env:PIP_CACHE_DIR = $PipCacheDir
    $env:NPM_CONFIG_CACHE = $NpmCacheDir
    $env:npm_config_cache = $NpmCacheDir
    $env:XDG_CACHE_HOME = $RuntimeCacheDir
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $BackendDir ".venv"
    $env:PYTHONPYCACHEPREFIX = $PythonCacheDir
    $env:PYTEST_CACHE_DIR = $PytestCacheDir
    $env:PYTEST_ADDOPTS = "--basetemp=`"$PytestTempDir`""
    $env:RUFF_CACHE_DIR = $RuffCacheDir
    $env:MYPY_CACHE_DIR = $MypyCacheDir
    $env:COVERAGE_FILE = Join-Path $CoverageDir ".coverage"
    $env:PLAYWRIGHT_BROWSERS_PATH = $PlaywrightCacheDir
    $env:TEMP = $RuntimeTempDir
    $env:TMP = $RuntimeTempDir
    $env:TMPDIR = $RuntimeTempDir
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    $env:PYTHONPATH = $AppDir
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
    param([switch]$BuildFrontend, [switch]$RuntimesReady, [ValidateSet('Standard', 'Development')][string]$InstallationType = 'Standard', [ValidateSet('Base', 'ML')][string]$FeatureSet = 'Base')
    Import-Settings | Out-Null
    if (-not $RuntimesReady) { Initialize-Runtimes }
    Set-RuntimeEnvironment
    Write-Step "Syncing Python dependencies ($FeatureSet)"
    Push-Location $BackendDir
    try {
        $arguments = @('sync', '--locked', '--python', $PythonExe)
        if ($FeatureSet -eq 'ML') { $arguments += '--extra', 'ml' }
        if ($InstallationType -eq 'Development') { $arguments += '--group', 'dev' } else { $arguments += '--no-dev' }
        & $UvExe @arguments
        Assert-LastExitCode "uv sync"
    } finally { Pop-Location }
    if (-not (Test-Path -LiteralPath $VenvPython)) { throw "Backend virtual-environment Python was not created at $VenvPython." }
    Write-Ok "Python dependencies are ready."
    if ($InstallationType -eq 'Development') {
        Write-Step "Installing Playwright Chromium browser"
        & $VenvPython -m playwright install chromium
        Assert-LastExitCode "Playwright Chromium installation"
        Write-Ok "Playwright Chromium is ready under $PlaywrightCacheDir."
    }
    Sync-FrontendDependencies -BuildFrontend:$BuildFrontend
}

function Test-FrontendBuildCurrent {
    param([Parameter(Mandatory)][string]$BuildPath)

    if (-not (Test-Path -LiteralPath $BuildPath -PathType Leaf)) {
        return $false
    }

    $buildTimestamp = (Get-Item -LiteralPath $BuildPath).LastWriteTimeUtc
    $sourcePaths = @(
        (Join-Path $ClientDir 'angular.json'),
        (Join-Path $ClientDir 'index.html'),
        (Join-Path $ClientDir 'tsconfig.json'),
        (Join-Path $ClientDir 'tsconfig.app.json'),
        (Join-Path $ClientDir 'src'),
        (Join-Path $ClientDir 'public')
    )
    foreach ($sourcePath in $sourcePaths) {
        if (-not (Test-Path -LiteralPath $sourcePath)) {
            continue
        }
        $source = Get-Item -LiteralPath $sourcePath
        if ($source.LastWriteTimeUtc -gt $buildTimestamp) {
            return $false
        }
        if ($source.PSIsContainer) {
            $newerFile = Get-ChildItem -LiteralPath $sourcePath -File -Recurse -Force -ErrorAction SilentlyContinue |
                Where-Object { $_.LastWriteTimeUtc -gt $buildTimestamp } |
                Select-Object -First 1
            if ($null -ne $newerFile) {
                return $false
            }
        }
    }
    return $true
}

function Test-DependenciesReady {
    param([switch]$IgnoreFrontendBuild)

    $frontendPackage = Join-Path $ClientDir 'package.json'
    $frontendLock = Join-Path $ClientDir 'package-lock.json'
    $frontendModules = Join-Path $ClientDir 'node_modules'
    $frontendInstallState = Join-Path $frontendModules '.package-lock.json'
    $frontendRunner = Join-Path $frontendModules '@angular/cli/bin/ng.js'
    $frontendBuild = Join-Path $ClientDir 'dist\browser\index.html'
    $backendEntrypoint = Join-Path $BackendDir 'cli.py'
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
        (-not $IgnoreFrontendBuild -and -not (Test-FrontendBuildCurrent -BuildPath $frontendBuild))) {
        return $false
    }

    & $PythonExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $UvExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $NodeExe --version *> $null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $VenvPython -c 'import server.app, fastapi, uvicorn' *> $null
    if ($LASTEXITCODE -ne 0) { return $false }

    return $true
}

function Get-ListeningProcess([int]$Port) {
    $connections = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
    $processIds = @(
        $connections |
            Select-Object -ExpandProperty OwningProcess |
            Sort-Object -Unique
    )
    foreach ($processId in $processIds) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue | Select-Object -First 1
        [pscustomobject]@{
            Id = [int]$processId
            Name = if ($process) { $process.ProcessName } else { 'unknown' }
        }
    }
}

function Assert-PortAvailable([int]$Port) {
    $owners = @(Get-ListeningProcess -Port $Port)
    if ($owners.Count -eq 0) {
        return
    }

    $ownerText = ($owners | ForEach-Object { "PID $($_.Id) ($($_.Name))" }) -join ', '
    throw "Port $Port is already in use by $ownerText. ADSMOD will not terminate unowned processes; stop the owning application and retry."
}

function Get-ApplicationProcessRecords {
    param([Parameter(Mandatory)][pscustomobject]$Settings)

    $records = @(Get-CimInstance -ClassName Win32_Process -ErrorAction SilentlyContinue)
    if ($records.Count -eq 0) {
        return @()
    }

    $configMarker = [System.IO.Path]::GetFullPath($ConfigFile).TrimEnd('\').ToLowerInvariant()
    $backendMarker = [System.IO.Path]::GetFullPath($BackendDir).TrimEnd('\').ToLowerInvariant()
    $clientMarker = [System.IO.Path]::GetFullPath($ClientDir).TrimEnd('\').ToLowerInvariant()
    $backendPortToken = "--port $($Settings.BackendPort)".ToLowerInvariant()
    $backendPortEqualsToken = "--port=$($Settings.BackendPort)".ToLowerInvariant()
    $frontendPortToken = "--port $($Settings.FrontendPort)".ToLowerInvariant()
    $frontendPortEqualsToken = "--port=$($Settings.FrontendPort)".ToLowerInvariant()
    $matchedIds = [System.Collections.Generic.HashSet[int]]::new()

    foreach ($record in $records) {
        $commandLine = ([string]$record.CommandLine).Replace('/', '\').ToLowerInvariant()
        $isBackend = $commandLine.Contains($configMarker) -or
            ($commandLine.Contains('server.cli') -and
                ($commandLine.Contains($backendMarker) -or
                    $commandLine.Contains($backendPortToken) -or
                    $commandLine.Contains($backendPortEqualsToken)))
        $hasPreviewCommand = $commandLine.Contains('vite preview') -or
            ($commandLine.Contains('npm') -and $commandLine.Contains('run preview'))
        $isFrontend = $hasPreviewCommand -and
            ($commandLine.Contains($clientMarker) -or
                $commandLine.Contains($frontendPortToken) -or
                $commandLine.Contains($frontendPortEqualsToken))

        if ($isBackend -or $isFrontend) {
            [void]$matchedIds.Add([int]$record.ProcessId)
        }
    }

    # Include children such as the node process created by npm.cmd. This keeps
    # the cleanup scoped to a recognized ADSMOD process tree.
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($record in $records) {
            $processId = [int]$record.ProcessId
            $parentProcessId = [int]$record.ParentProcessId
            if (-not $matchedIds.Contains($processId) -and $matchedIds.Contains($parentProcessId)) {
                [void]$matchedIds.Add($processId)
                $changed = $true
            }
        }
    }

    return @(
        $records |
            Where-Object { $matchedIds.Contains([int]$_.ProcessId) } |
            Sort-Object @{ Expression = { [int]$_.ProcessId }; Descending = $true }
    )
}

function Stop-OwnedProcess {
    param(
        [System.Diagnostics.Process]$Process,
        [Parameter(Mandatory)][string]$Name
    )

    if ($null -eq $Process) {
        return
    }

    try {
        $Process.Refresh()
        if ($Process.HasExited) {
            return
        }
        Write-Step "Stopping ADSMOD $Name (PID $($Process.Id))"
        $Process.Kill($true)
        if (-not $Process.WaitForExit(5000)) {
            throw "ADSMOD $Name did not exit within five seconds."
        }
    }
    catch [System.InvalidOperationException] {
        # The process can exit between Refresh and Kill. That is already a
        # successful stop for this launcher-owned process.
        return
    }
}

function Stop-AllApplicationProcesses {
    if (-not (Confirm-DestructiveAction 'kill all recognized ADSMOD application processes')) {
        return
    }

    $settings = Import-Settings
    $records = @(Get-ApplicationProcessRecords -Settings $settings)
    $processIds = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($record in $records) {
        [void]$processIds.Add([int]$record.ProcessId)
    }
    foreach ($ownedProcess in @($script:FrontendProcess, $script:BackendProcess)) {
        if ($null -eq $ownedProcess) {
            continue
        }
        try {
            $ownedProcess.Refresh()
            if (-not $ownedProcess.HasExited) {
                [void]$processIds.Add([int]$ownedProcess.Id)
            }
        }
        catch [System.InvalidOperationException] {
            # The process exited while the inventory was being collected.
        }
    }

    if ($processIds.Count -eq 0) {
        Write-Warn "No recognized ADSMOD application processes were found."
        return
    }

    $stopped = 0
    Write-Step "Killing $($processIds.Count) recognized ADSMOD application process(es)"
    foreach ($processId in @($processIds | Sort-Object -Descending)) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $process) {
            continue
        }
        try {
            Write-Host "[STEP] Stopping ADSMOD process $($process.ProcessName) (PID $processId)" -ForegroundColor Cyan
            $process.Kill($true)
            if (-not $process.WaitForExit(5000)) {
                Write-Warn "ADSMOD process PID $processId did not exit within five seconds."
                continue
            }
            $stopped++
        }
        catch [System.InvalidOperationException] {
            $stopped++
        }
        catch {
            Write-Warn "Could not stop ADSMOD process PID ${processId}: $($_.Exception.Message)"
        }
    }

    $script:FrontendProcess = $null
    $script:BackendProcess = $null
    if ($stopped -gt 0) {
        Write-Ok "Stopped $stopped recognized ADSMOD application process(es)."
    }
    else {
        Write-Warn "No recognized ADSMOD application processes could be stopped."
    }
}

function Stop-Application {
    $stopped = $false
    if ($null -ne $script:FrontendProcess) {
        Stop-OwnedProcess -Process $script:FrontendProcess -Name 'frontend'
        $script:FrontendProcess = $null
        $stopped = $true
    }
    if ($null -ne $script:BackendProcess) {
        Stop-OwnedProcess -Process $script:BackendProcess -Name 'backend'
        $script:BackendProcess = $null
        $stopped = $true
    }
    if ($stopped) {
        Write-Ok "ADSMOD processes started by this launcher session have stopped."
    }
    else {
        Write-Warn "This launcher session does not own any ADSMOD processes."
    }
}

function Start-Application {
    $settings = Import-Settings
    Set-RuntimeEnvironment
    $frontendBuild = Join-Path $ClientDir 'dist\browser\index.html'
    if (-not (Test-DependenciesReady -IgnoreFrontendBuild)) {
        Write-Step "Required application environments or frontend dependencies are missing or unusable; repairing the base installation."
        Sync-Dependencies -BuildFrontend -FeatureSet Base
    }
    elseif (-not (Test-FrontendBuildCurrent -BuildPath $frontendBuild)) {
        Write-Step "Frontend source or configuration is newer than the generated bundle; rebuilding the frontend."
        Sync-FrontendDependencies -BuildFrontend
    }
    else {
        Write-Ok "Application environments and frontend build are ready; skipped dependency installation."
    }
    Set-RuntimeEnvironment
    $backendPort = $settings.BackendPort
    $uiPort = $settings.FrontendPort
    Assert-PortAvailable -Port $backendPort
    Assert-PortAvailable -Port $uiPort
    $backendArguments = @('-m', 'server.cli', '--config', ('"{0}"' -f $ConfigFile))
    Write-Step "Starting ADSMOD backend"
    $script:BackendProcess = Start-Process -FilePath $VenvPython -ArgumentList $backendArguments -WorkingDirectory $RepoRoot -WindowStyle Normal -PassThru
    $healthUrl = "http://$($settings.Host):$($settings.BackendPort)/health/ready"
    Write-Step "Waiting for backend readiness at $healthUrl"
    try { Wait-ForHealth -Url $healthUrl -TimeoutSeconds 60 } catch { Stop-OwnedProcess -Process $script:BackendProcess -Name 'backend'; $script:BackendProcess = $null; throw }
    Write-Step "Starting frontend preview"
    $script:FrontendProcess = Start-Process -FilePath $NpmCmd -ArgumentList @('run', 'preview', '--', '--host', $settings.Host, '--port', $settings.FrontendPort) -WorkingDirectory $ClientDir -WindowStyle Hidden -PassThru
    $frontendUrl = "http://$($settings.Host):$($settings.FrontendPort)"
    try { Wait-ForHealth -Url $frontendUrl -TimeoutSeconds 60 } catch { Stop-OwnedProcess -Process $script:FrontendProcess -Name 'frontend'; $script:FrontendProcess = $null; Stop-OwnedProcess -Process $script:BackendProcess -Name 'backend'; $script:BackendProcess = $null; throw }
    Start-Process -FilePath 'explorer.exe' -ArgumentList $frontendUrl
    Write-Host ""
    Write-Ok "ADSMOD started successfully."
    Write-Host "Backend: $healthUrl (PID $($script:BackendProcess.Id))" -ForegroundColor Green
    Write-Host "Frontend: $frontendUrl (PID $($script:FrontendProcess.Id))" -ForegroundColor Green
}

function Install-UpdateDependencies {
    Initialize-Runtimes
    Write-Ok "Portable runtimes ready."
    $installationType = Read-InstallationType
    $featureSet = Read-FeatureSet
    Sync-Dependencies -BuildFrontend -RuntimesReady -InstallationType $installationType -FeatureSet $featureSet
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

function Read-FeatureSet {
    Write-Host "  [1] Base - core ADSMOD functionality only"
    Write-Host "  [2] ML   - base application plus optional machine learning dependencies"
    $selection = (Read-Host "  Select feature set [1-2]").Trim()
    switch ($selection) { '1' { return 'Base' }; '2' { return 'ML' }; default { throw "Invalid feature set. Enter 1 for Base or 2 for ML." } }
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

function Confirm-DestructiveAction([string]$Description) {
    if (-not $script:LauncherInteractive) {
        throw "The destructive action '$Description' requires an interactive console; no files were changed."
    }
    $confirmation = ([string](Read-Host "Continue to $($Description)? [y/N]")).Trim()
    if ($confirmation -notmatch '^(?i:y|yes)$') {
        Write-Info "Operation cancelled. No changes were made."
        return $false
    }
    return $true
}

function Remove-Logs {
    Import-Settings | Out-Null
    if (-not (Confirm-DestructiveAction 'remove application log files')) { return }
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

function Get-RepositoryDirectories {
    param([Parameter(Mandatory)][string]$Root)
    $pending = [Collections.Generic.Stack[string]]::new()
    [void]$pending.Push($Root)
    while ($pending.Count -gt 0) {
        $current = $pending.Pop()
        $enumerationErrors = @()
        $children = @(Get-ChildItem -LiteralPath $current -Directory -Force -ErrorAction SilentlyContinue -ErrorVariable enumerationErrors)
        foreach ($errorRecord in $enumerationErrors) {
            $message = "$current ($($errorRecord.Exception.Message))"
            [void]$script:CacheScanErrors.Add($message)
            Write-Warn "Unable to inspect repository directory '$current': $($errorRecord.Exception.Message)"
        }
        foreach ($child in $children) {
            $fullPath = [System.IO.Path]::GetFullPath($child.FullName)
            if ($child.Name -in @('.git', 'node_modules', '.venv', 'runtimes')) { continue }
            if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { continue }
            $fullPath
            [void]$pending.Push($fullPath)
        }
    }
}

function Get-LegacyCachePaths {
    $canonicalPrefix = [System.IO.Path]::GetFullPath($RuntimeCacheDir).TrimEnd('\') + '\'
    $appPrefix = [System.IO.Path]::GetFullPath($AppDir).TrimEnd('\') + '\'
    $legacyNames = @(
        '__pycache__',
        '.pytest_cache',
        '.ruff_cache',
        '.mypy_cache',
        '.uv-cache',
        '.angular',
        '.startup-temp',
        '.cache'
    )
    $candidates = [Collections.Generic.List[string]]::new()
    $searchRoots = @($RepoRoot, $AppDir)
    foreach ($searchRoot in $searchRoots) {
        if (-not (Test-Path -LiteralPath $searchRoot -PathType Container)) { continue }
        foreach ($fullPath in @(Get-RepositoryDirectories -Root $searchRoot)) {
            if ($fullPath.StartsWith($canonicalPrefix, [StringComparison]::OrdinalIgnoreCase)) { continue }
            $name = [System.IO.Path]::GetFileName($fullPath)
            $isLegacyName = $name -in $legacyNames -or $name -like '.pytest-*'
            $isApplicationPath = $fullPath.StartsWith($appPrefix, [StringComparison]::OrdinalIgnoreCase)
            $isLegacyCacheRoot = $name -eq 'cache' -and $isApplicationPath
            $isLegacyCoverageRoot = $name -in @('coverage', 'htmlcov') -and $isApplicationPath
            if ($isLegacyName -or $isLegacyCacheRoot -or $isLegacyCoverageRoot) {
                [void]$candidates.Add($fullPath)
            }
        }
    }
    return @($candidates | Sort-Object @{ Expression = { $_.Length }; Descending = $true }, @{ Expression = { $_.ToUpperInvariant() }; Descending = $false } -Unique)
}

function Get-LegacyCacheFiles {
    $canonicalPrefix = [System.IO.Path]::GetFullPath($RuntimeCacheDir).TrimEnd('\') + '\'
    $files = [Collections.Generic.List[string]]::new()
    if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) { return @() }
    foreach ($directory in @($RepoRoot) + @(Get-RepositoryDirectories -Root $RepoRoot)) {
        $fullDirectory = [System.IO.Path]::GetFullPath($directory)
        if ($fullDirectory.StartsWith($canonicalPrefix, [StringComparison]::OrdinalIgnoreCase)) { continue }
        $entries = @(Get-ChildItem -LiteralPath $fullDirectory -File -Force -ErrorAction SilentlyContinue)
        foreach ($entry in $entries) {
            if ($entry.Name -eq '.coverage' -or $entry.Name -like '.coverage.*' -or $entry.Extension -in @('.pyc', '.pyo')) {
                [void]$files.Add([System.IO.Path]::GetFullPath($entry.FullName))
            }
        }
    }
    return @($files | Sort-Object -Unique)
}

function Clear-Cache {
    if (-not (Confirm-DestructiveAction 'clear runtime and test-tool caches')) { return }
    Write-Step "Clearing runtime and test-tool caches"
    $script:CacheScanErrors.Clear()
    $failures = [Collections.Generic.List[string]]::new()
    try {
        [void](Remove-LauncherPath -Path $RuntimeCacheDir -KeepRoot -PreserveNames @('.gitkeep', 'CACHEDIR.TAG') -Strict -Activity 'ADSMOD: clear canonical cache')
    } catch {
        [void]$failures.Add("${RuntimeCacheDir}: $($_.Exception.Message)")
    }
    foreach ($legacyPath in @(Get-LegacyCachePaths)) {
        try {
            [void](Remove-LauncherPath -Path $legacyPath -Strict -Activity "ADSMOD: remove legacy cache $([IO.Path]::GetFileName($legacyPath))")
        } catch {
            [void]$failures.Add("${legacyPath}: $($_.Exception.Message)")
        }
    }
    foreach ($legacyFile in @(Get-LegacyCacheFiles)) {
        try {
            [void](Remove-LauncherPath -Path $legacyFile -Strict -Activity "ADSMOD: remove legacy cache artifact $([IO.Path]::GetFileName($legacyFile))")
        } catch {
            [void]$failures.Add("${legacyFile}: $($_.Exception.Message)")
        }
    }
    foreach ($scanError in @($script:CacheScanErrors | Sort-Object -Unique)) {
        [void]$failures.Add("Cache discovery: $scanError")
    }
    Ensure-CacheLayout
    if ($failures.Count -gt 0) {
        throw "Cache cleanup was incomplete:`n - $($failures -join "`n - ")"
    }
    Write-Ok "Cache cleanup completed under $RuntimeCacheDir."
}

function Uninstall-Application {
    if (-not (Confirm-DestructiveAction 'remove local application runtimes and build artifacts')) { return }
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
        (Join-Path $RepoRoot '.venv'),
        (Join-Path $ClientDir 'node_modules'),
        (Join-Path $ClientDir '.angular'),
        (Join-Path $ClientDir 'dist')
    )
    foreach ($path in $paths) {
        Remove-RepoPath $path
    }
    foreach ($legacyPath in @(Get-LegacyCachePaths)) {
        Remove-RepoPath $legacyPath
    }
    foreach ($legacyFile in @(Get-LegacyCacheFiles)) {
        Remove-RepoPath $legacyFile
    }
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
    if (-not (Confirm-DestructiveAction 'remove all saved training checkpoints')) { return }
    Clear-CheckpointFiles
}

function Remove-All-Data {
    Import-Settings | Out-Null
    if (-not (Confirm-DestructiveAction 'remove all local user-generated data')) { return }

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
    Clear-LauncherProgress
    Write-Host ""
    Write-Host "Press any key to return to the menu..." -ForegroundColor DarkGray
    if (-not $script:LauncherInteractive) { return }
    [Console]::ReadKey($true) | Out-Null
}

function Get-MainMenuEntries {
    return @(
        [pscustomobject]@{ Section = 'APPLICATION'; Label = 'Launch application'; Hint = 'Start the local web workspace'; Key = 'Launch'; Destructive = $false }
        [pscustomobject]@{ Section = 'APPLICATION'; Label = 'Stop application'; Hint = 'Stop ADSMOD processes started by this session'; Key = 'Stop'; Destructive = $false }
        [pscustomobject]@{ Section = 'APPLICATION'; Label = 'Kill all application processes'; Hint = 'Stop every recognized ADSMOD process, including previous sessions'; Key = 'KillAll'; Destructive = $true }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Install / update dependencies'; Hint = 'Refresh local runtimes and packages'; Key = 'Install'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Rebuild frontend'; Hint = 'Install frontend packages and rebuild the bundle'; Key = 'Rebuild'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Initialize database'; Hint = 'Create or upgrade the Alembic-managed data store'; Key = 'Database'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Run test suite'; Hint = 'Execute the repository checks'; Key = 'Tests'; Destructive = $false }
        [pscustomobject]@{ Section = 'SOURCE CONTROL'; Label = 'Check for updates'; Hint = 'Report whether origin/main has a newer version'; Key = 'Check'; Destructive = $false }
        [pscustomobject]@{ Section = 'SOURCE CONTROL'; Label = 'Update application'; Hint = 'Pull origin/main (clean main branch required)'; Key = 'Update'; Destructive = $false }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Remove logs'; Hint = 'Delete generated log files'; Key = 'Logs'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Clear cache'; Hint = 'Remove runtime and test-tool caches'; Key = 'Cache'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Remove checkpoints'; Hint = 'Delete saved training checkpoints'; Key = 'Checkpoints'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Remove all data'; Hint = 'Delete local database and user-generated files'; Key = 'AllData'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Uninstall application'; Hint = 'Remove local runtimes and build artifacts'; Key = 'Uninstall'; Destructive = $true }
        [pscustomobject]@{ Section = 'EXIT'; Label = 'Exit'; Hint = 'Close this console'; Key = 'Exit'; Destructive = $false }
    )
}

function Write-MenuItem {
    param(
        [Parameter(Mandatory)][int]$Number,
        [Parameter(Mandatory)][int]$NumberWidth,
        [Parameter(Mandatory)][int]$LabelWidth,
        [Parameter(Mandatory)][pscustomobject]$Entry
    )
    $color = if ($Entry.Destructive) { 'Yellow' } elseif ($Entry.Key -eq 'Exit') { 'DarkGray' } else { 'White' }
    $line = ("  {0,$NumberWidth}. {1,-$LabelWidth}  {2}" -f $Number, $Entry.Label, $Entry.Hint)
    Write-Host $line -ForegroundColor $color
}

function Show-MainMenu {
    Clear-LauncherProgress
    if ($script:LauncherInteractive) { try { Clear-Host } catch { } }

    $entries = @(Get-MainMenuEntries)
    for ($index = 0; $index -lt $entries.Count; $index++) {
        $entries[$index] = [pscustomobject]@{
            Number = $index + 1
            Section = $entries[$index].Section
            Label = $entries[$index].Label
            Hint = $entries[$index].Hint
            Key = $entries[$index].Key
            Destructive = $entries[$index].Destructive
        }
    }
    $numberWidth = [string]$entries.Count
    $numberWidth = $numberWidth.Length
    $labelWidth = ($entries | ForEach-Object { $_.Label.Length } | Measure-Object -Maximum).Maximum

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
    $lastSection = $null
    foreach ($entry in $entries) {
        if ($entry.Section -ne $lastSection) {
            if ($null -ne $lastSection) { Write-Host "" }
            Write-Host ("  {0}" -f $entry.Section) -ForegroundColor DarkCyan
            if ($entry.Section -eq 'EXIT') { Write-Host $subtleRule -ForegroundColor DarkGray }
            $lastSection = $entry.Section
        }
        Write-MenuItem -Number $entry.Number -NumberWidth $numberWidth -LabelWidth $labelWidth -Entry $entry
    }
    Write-Host $rule -ForegroundColor DarkCyan
    return $entries
}

$exitMenu = $false
while (-not $exitMenu) {
    $entries = @(Show-MainMenu)
    $maxOption = $entries.Count
    $selection = (Read-Host "  Select an option (1-$maxOption)").Trim()

    if ($selection -notmatch "^[1-9][0-9]*$" -or [int]$selection -lt 1 -or [int]$selection -gt $maxOption) {
        Write-Warn "Please select a number from 1 to $maxOption."
        Wait-ForMenu
        continue
    }

    $entry = $entries[[int]$selection - 1]
    if ($entry.Key -eq 'Exit') { $exitMenu = $true; continue }

    try {
        Invoke-TrackedLauncherAction -Name $entry.Label -Action {
            switch ($entry.Key) {
                'Launch' {
                    Start-Application
                }
                'Stop' { Stop-Application }
                'KillAll' { Stop-AllApplicationProcesses }
                'Install' { Install-UpdateDependencies }
                'Rebuild' { Rebuild-Frontend }
                'Database' { Initialize-Database }
                'Tests' { Invoke-TestSuite }
                'Check' { Check-For-Updates }
                'Update' { Update-Application }
                'Logs' { Remove-Logs }
                'Cache' { Clear-Cache }
                'Checkpoints' { Remove-Checkpoints }
                'AllData' { Remove-All-Data }
                'Uninstall' { Uninstall-Application }
            }
        }
        Wait-ForMenu
    } catch {
        Write-Fatal $_.Exception.Message
        if ($selection -eq '1') {
            exit 1
        }
        Wait-ForMenu
    }
}
Clear-LauncherProgress
