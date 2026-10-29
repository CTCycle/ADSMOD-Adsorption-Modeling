# Copyright © 2023 Thomas Virdis
# Licensed under the MIT License.

[CmdletBinding()]
param([switch]$DryRun)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$AppDir = Join-Path $RepoRoot "app"
$BackendDir = Join-Path $AppDir "server"
$ClientDir = Join-Path $AppDir "client"
$TestsDir = Join-Path $AppDir "tests"
$DefaultDataDir = if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) { Join-Path $env:LOCALAPPDATA "ADSMOD" } else { Join-Path $HOME 'AppData\Local\ADSMOD' }
$DataDir = $DefaultDataDir
$LogDir = Join-Path $DataDir "logs"
$CheckpointsDir = Join-Path $DataDir "checkpoints"
$script:DatabasePath = $null
$script:EmbeddedDatabase = $true
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
$script:LauncherInteractive = -not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected
$script:BackendProcess = $null

Import-Module (Join-Path (Join-Path $RepoRoot 'app') 'scripts\launcher_cleanup.psm1') -Force -ErrorAction Stop

$PythonVersion = "3.14.7"
$PythonExe = Join-Path $PythonDir "python.exe"
$PythonPth = Join-Path $PythonDir "python314._pth"
$UvExe = Join-Path $UvDir "uv.exe"
$NodeExe = Join-Path $NodeDir "node.exe"
$NpmCmd = Join-Path $NodeDir "npm.cmd"
$VenvDir = Join-Path $BackendDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$BackendDependencyStateFile = Join-Path $VenvDir ".adsmod-dependency-state.json"
$FrontendDependencyStateFile = Join-Path $ClientDir "node_modules\.adsmod-dependency-state.json"
$FrontendBuildStateFile = Join-Path $ClientDir "dist\.adsmod-build-state.json"
$DependencyStateSchemaVersion = 1
$BuildStateSchemaVersion = 1

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

function Write-Info([string]$Message) {
    Clear-LauncherProgress
    Write-Host "[INFO] $Message" -ForegroundColor Gray
}

function Write-Warn([string]$Message) {
    Clear-LauncherProgress
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-LaunchTiming {
    param(
        [Parameter(Mandatory)][string]$Phase,
        [Parameter(Mandatory)][long]$ElapsedMilliseconds
    )
    Write-Host ('[TIME] {0,-42} {1,8} ms' -f "$($Phase):", $ElapsedMilliseconds) -ForegroundColor Gray
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
        $actionResult = & $Action
        if ($actionResult -is [bool] -and -not $actionResult) {
            Write-Warn "$Name cancelled."
            return
        }
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

function Remove-RepoPath([string]$Path) {
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $repoPrefix = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\') + '\'
    if (-not $fullPath.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the repository: $fullPath"
    }
    $result = Invoke-AdsmodScopedCleanup -Path $fullPath -ScopeRoot $RepoRoot -Kind Directory -Category "ADSMOD runtime path: $([IO.Path]::GetFileName($fullPath))"
    foreach ($path in $result.Skipped) { Write-Warn "Skipping '$path'" }
    foreach ($path in $result.Failed) { Write-Warn "Failed to remove '$path'" }
    return $result.Failed.Count -eq 0 -and $result.Skipped.Count -eq 0
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
    $result = Invoke-AdsmodScopedCleanup -Path $actualCachePath -ScopeRoot $RepoRoot -Kind Contents -PreserveNames @('.gitkeep', 'CACHEDIR.TAG') -Category 'ADSMOD uv cache'
    foreach ($path in $result.Skipped) { Write-Warn "Skipping '$path'" }
    foreach ($path in $result.Failed) { Write-Warn "Failed to remove '$path'" }
    $unexpectedSkips = @($result.Skipped | Where-Object { $_ -notmatch '\(preserved sentinel\)$' })
    if ($result.Failed.Count -gt 0 -or $unexpectedSkips.Count -gt 0) { throw "uv cache cleanup was incomplete ($($result.Failed.Count) failed, $($unexpectedSkips.Count) skipped)." }
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
    $version.Trim()
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
        [ValidateRange(1, 600)][int]$TimeoutSeconds = 60,
        [System.Diagnostics.Process]$Process = $null
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $activity = "ADSMOD: wait for health $Url"
    $progressId = Start-LauncherProgress -Activity $activity -Status "Waiting up to $TimeoutSeconds seconds"
    try {
        do {
            if ($null -ne $Process) {
                try {
                    $Process.Refresh()
                    if ($Process.HasExited) {
                        throw "Process PID $($Process.Id) exited before $Url became ready with exit code $($Process.ExitCode)."
                    }
                }
                catch [System.InvalidOperationException] {
                    throw "Process for $Url exited before it became ready."
                }
            }
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
            Start-Sleep -Milliseconds 250
        } while ([DateTime]::UtcNow -lt $deadline)
        throw "No healthy response from $Url within $TimeoutSeconds seconds."
    } finally {
        Complete-LauncherProgress $progressId
    }
}

function Import-Settings {
    if (-not (Test-PortableRuntimeFilesReady)) { Initialize-Runtimes }
    Set-RuntimeEnvironment
    Push-Location $RepoRoot
    try {
        $runtimeOutput = @(& $UvExe run --project $BackendDir --python $PythonExe python -m server.runtime_config --json 2>&1)
        Assert-LastExitCode "loading environment runtime settings"
    } finally {
        Pop-Location
    }
    $runtimeLine = @($runtimeOutput | ForEach-Object { [string]$_ } | Where-Object { $_.StartsWith('ADSMOD_RUNTIME:') } | Select-Object -Last 1)
    if ($runtimeLine.Count -ne 1) { throw "Environment runtime settings did not return a valid local endpoint." }
    try { $runtime = $runtimeLine[0].Substring('ADSMOD_RUNTIME:'.Length) | ConvertFrom-Json } catch { throw "Environment runtime settings returned invalid data." }
    $runtimeHost = [string]$runtime.host
    $backendPort = [int]$runtime.backend_port
    $frontendPort = [int]$runtime.frontend_port
    if ($runtimeHost -notin @('127.0.0.1', 'localhost', '::1') -or $backendPort -lt 1024 -or $backendPort -gt 65535 -or $frontendPort -lt 1024 -or $frontendPort -gt 65535 -or $backendPort -eq $frontendPort) {
        throw "Environment runtime settings contain an invalid local endpoint."
    }
    $backendUrl = [string]$runtime.backend_url
    $frontendUrl = [string]$runtime.frontend_url
    if (-not $backendUrl -or -not $frontendUrl) { throw "Environment runtime settings returned incomplete endpoint URLs." }
    $script:DataDir = [System.IO.Path]::GetFullPath([string]$runtime.storage_root)
    $script:LogDir = Join-Path $script:DataDir "logs"
    $script:CheckpointsDir = Join-Path $script:DataDir "checkpoints"
    $script:DatabasePath = if ($runtime.database_path) { [System.IO.Path]::GetFullPath([string]$runtime.database_path) } else { $null }
    $script:EmbeddedDatabase = ([string]$runtime.database_backend -eq 'sqlite')
    [Environment]::SetEnvironmentVariable('ADSMOD_PYTHON', $VenvPython, 'Process')
    return [pscustomobject]@{ Host = $runtimeHost; BackendPort = $backendPort; FrontendPort = $frontendPort; BackendUrl = $backendUrl; FrontendUrl = $frontendUrl; StorageRoot = $script:DataDir; DatabasePath = $script:DatabasePath; EmbeddedDatabase = $script:EmbeddedDatabase }
}

function Set-RuntimeEnvironment {
    Ensure-CacheLayout
    $env:UV_CACHE_DIR = $UvCacheDir
    Remove-Item Env:UV_NO_CACHE -ErrorAction SilentlyContinue
    $env:PIP_CACHE_DIR = $PipCacheDir
    $env:NPM_CONFIG_CACHE = $NpmCacheDir
    $env:npm_config_cache = $NpmCacheDir
    $env:XDG_CACHE_HOME = $RuntimeCacheDir
    $env:UV_PROJECT_ENVIRONMENT = $VenvDir
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
    $pythonNeedsInstall = -not (Test-Path -LiteralPath $PythonExe)
    if (-not $pythonNeedsInstall) {
        try {
            $existingPythonVersion = Get-PythonVersion -PythonExe $PythonExe
        } catch {
            $existingPythonVersion = $null
        }
        if ($existingPythonVersion -ne $PythonVersion) {
            Write-Warn "Replacing portable Python $existingPythonVersion with $PythonVersion."
            $pythonStageDir = Join-Path $RuntimesDir ".python-$PythonVersion-staging"
            $pythonBackupDir = Join-Path $RuntimesDir ".python-$existingPythonVersion-backup"
            if (Test-Path -LiteralPath $pythonStageDir -PathType Container) {
                if (-not (Remove-RepoPath $pythonStageDir)) {
                    throw "Could not clear the staged portable Python runtime at $pythonStageDir."
                }
            }
            if (Test-Path -LiteralPath $pythonBackupDir -PathType Container) {
                if (-not (Remove-RepoPath $pythonBackupDir)) {
                    throw "Could not clear the portable Python backup at $pythonBackupDir."
                }
            }
            New-Item -ItemType Directory -Path $pythonStageDir -Force | Out-Null
            try {
                Download-AndExtract `
                    -Url $PythonUrl `
                    -ArchivePath (Join-Path $pythonStageDir $PythonArchive) `
                    -DestinationPath $pythonStageDir
                $stagedPth = Join-Path $pythonStageDir "python314._pth"
                Patch-PythonPth -Path $stagedPth
                $stagedPythonExe = Join-Path $pythonStageDir "python.exe"
                $stagedPythonVersion = Get-PythonVersion -PythonExe $stagedPythonExe
                if ($stagedPythonVersion -ne $PythonVersion) {
                    throw "Downloaded portable Python version mismatch. Expected $PythonVersion, detected $stagedPythonVersion."
                }
                Move-Item -LiteralPath $PythonDir -Destination $pythonBackupDir
                try {
                    Move-Item -LiteralPath $pythonStageDir -Destination $PythonDir
                } catch {
                    Move-Item -LiteralPath $pythonBackupDir -Destination $PythonDir -Force
                    throw
                }
                if ((Test-Path -LiteralPath $pythonBackupDir -PathType Container) -and
                    -not (Remove-RepoPath $pythonBackupDir)) {
                    Write-Warn "The previous portable Python runtime could not be removed from $pythonBackupDir."
                }
            } catch {
                if (Test-Path -LiteralPath $pythonStageDir -PathType Container) {
                    [void](Remove-RepoPath $pythonStageDir)
                }
                throw
            }
            $pythonNeedsInstall = $false
        }
    }
    if ($pythonNeedsInstall) {
        Download-AndExtract `
            -Url $PythonUrl `
            -ArchivePath (Join-Path $PythonDir $PythonArchive) `
            -DestinationPath $PythonDir
    }
    Patch-PythonPth -Path $PythonPth
    $detectedPython = Get-PythonVersion -PythonExe $PythonExe
    if ($detectedPython -ne $PythonVersion) {
        throw "Portable Python version mismatch. Expected $PythonVersion, detected $detectedPython."
    }
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
}

function Get-RepositoryRelativePath([string]$Path) {
    return ([System.IO.Path]::GetRelativePath($RepoRoot, [System.IO.Path]::GetFullPath($Path))).Replace('\', '/')
}

function Get-InputFingerprint {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string[]]$Paths,
        [hashtable]$AdditionalEntries = @{}
    )

    $files = [Collections.Generic.List[pscustomobject]]::new()
    $filePaths = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $missing = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($path in $Paths) {
        $fullPath = [System.IO.Path]::GetFullPath($path)
        if (-not (Test-Path -LiteralPath $fullPath)) {
            [void]$missing.Add((Get-RepositoryRelativePath $fullPath))
            continue
        }
        $item = Get-Item -LiteralPath $fullPath -Force
        $items = if ($item.PSIsContainer) {
            @(Get-ChildItem -LiteralPath $fullPath -File -Recurse -Force -ErrorAction Stop)
        } else {
            @($item)
        }
        foreach ($file in $items) {
            $relativePath = Get-RepositoryRelativePath $file.FullName
            if ($filePaths.Add($relativePath)) {
                [void]$files.Add([pscustomobject]@{ FullPath = $file.FullName; RelativePath = $relativePath })
            }
        }
    }

    $hasher = [System.Security.Cryptography.IncrementalHash]::CreateHash([System.Security.Cryptography.HashAlgorithmName]::SHA256)
    $encoding = [System.Text.UTF8Encoding]::new($false)
    try {
        foreach ($key in @($AdditionalEntries.Keys | Sort-Object)) {
            $metadata = "META|$key=$($AdditionalEntries[$key])`n"
            $hasher.AppendData($encoding.GetBytes($metadata))
        }
        foreach ($relativePath in @($missing | Sort-Object)) {
            $hasher.AppendData($encoding.GetBytes("MISSING|$relativePath`n"))
        }
        foreach ($file in @($files | Sort-Object @{ Expression = { $_.RelativePath.ToUpperInvariant() } }, RelativePath)) {
            $hasher.AppendData($encoding.GetBytes("FILE|$($file.RelativePath)|`n"))
            $stream = [System.IO.File]::OpenRead($file.FullPath)
            try {
                $buffer = [byte[]]::new(65536)
                while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    $hasher.AppendData($buffer, 0, $read)
                }
            } finally {
                $stream.Dispose()
            }
            $hasher.AppendData($encoding.GetBytes("`nEND|$($file.RelativePath)`n"))
        }
        return ([Convert]::ToHexString($hasher.GetHashAndReset())).ToLowerInvariant()
    } finally {
        $hasher.Dispose()
    }
}

function Get-BackendDependencyFingerprint {
    return Get-InputFingerprint -Paths @(
        (Join-Path $BackendDir 'pyproject.toml'),
        (Join-Path $BackendDir 'uv.lock')
    ) -AdditionalEntries @{ 'launcher-python-version' = $PythonVersion }
}

function Get-FrontendDependencyFingerprint {
    return Get-InputFingerprint -Paths @(
        (Join-Path $ClientDir 'package.json'),
        (Join-Path $ClientDir 'package-lock.json')
    ) -AdditionalEntries @{ 'launcher-node-version' = $NodeVersion }
}

function Get-FrontendBuildFingerprint {
    return Get-InputFingerprint -Paths @(
        (Join-Path $ClientDir 'package.json'),
        (Join-Path $ClientDir 'package-lock.json'),
        (Join-Path $ClientDir 'angular.json'),
        (Join-Path $ClientDir 'tsconfig.json'),
        (Join-Path $ClientDir 'tsconfig.app.json'),
        (Join-Path $ClientDir 'index.html'),
        (Join-Path $ClientDir 'src'),
        (Join-Path $ClientDir 'public'),
        (Join-Path $ClientDir 'scripts\verify-angular-migration.mjs')
    ) -AdditionalEntries @{ 'launcher-node-version' = $NodeVersion }
}

function Read-StateFile([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Get-StateValue {
    param([object]$State, [Parameter(Mandatory)][string]$Name)
    if ($null -eq $State) {
        return $null
    }
    $property = $State.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Write-StateFile {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][pscustomobject]$State
    )
    $directory = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $temporaryPath = "$Path.tmp"
    try {
        $json = $State | ConvertTo-Json -Depth 5
        [System.IO.File]::WriteAllText($temporaryPath, $json, [System.Text.UTF8Encoding]::new($false))
        Move-Item -LiteralPath $temporaryPath -Destination $Path -Force | Out-Null
    } finally {
        Remove-Item -LiteralPath $temporaryPath -Force -ErrorAction SilentlyContinue
    }
}

function Get-RecordedBackendProfile {
    $state = Read-StateFile -Path $BackendDependencyStateFile
    $installationType = 'Standard'
    $featureSet = 'Base'
    if ($null -ne $state) {
        $recordedInstallationType = [string](Get-StateValue -State $state -Name 'installationType')
        $recordedFeatureSet = [string](Get-StateValue -State $state -Name 'featureSet')
        if ($recordedInstallationType -in @('Standard', 'Development')) {
            $installationType = $recordedInstallationType
        }
        if ($recordedFeatureSet -in @('Base', 'ML')) {
            $featureSet = $recordedFeatureSet
        }
    }
    return [pscustomobject]@{ InstallationType = $installationType; FeatureSet = $featureSet }
}

function Test-PortableRuntimeFilesReady {
    return (Test-Path -LiteralPath $PythonExe -PathType Leaf) -and
        (Test-Path -LiteralPath $UvExe -PathType Leaf) -and
        (Test-Path -LiteralPath $NodeExe -PathType Leaf) -and
        (Test-Path -LiteralPath $NpmCmd -PathType Leaf)
}

function Test-BackendDependenciesCurrent {
    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf) -or
        -not (Test-Path -LiteralPath (Join-Path $BackendDir 'pyproject.toml') -PathType Leaf) -or
        -not (Test-Path -LiteralPath (Join-Path $BackendDir 'uv.lock') -PathType Leaf)) {
        return $false
    }
    $state = Read-StateFile -Path $BackendDependencyStateFile
    try {
        if ($null -eq $state -or [int](Get-StateValue -State $state -Name 'schemaVersion') -ne $DependencyStateSchemaVersion -or
            [string](Get-StateValue -State $state -Name 'pythonVersion') -ne $PythonVersion -or
            [string](Get-StateValue -State $state -Name 'fingerprint') -ne (Get-BackendDependencyFingerprint)) {
            return $false
        }
        return $true
    } catch {
        return $false
    }
}

function Test-FrontendDependenciesCurrent {
    $runner = Join-Path $ClientDir 'node_modules\@angular\cli\bin\ng.js'
    if (-not (Test-Path -LiteralPath $runner -PathType Leaf) -or
        -not (Test-Path -LiteralPath (Join-Path $ClientDir 'package.json') -PathType Leaf) -or
        -not (Test-Path -LiteralPath (Join-Path $ClientDir 'package-lock.json') -PathType Leaf)) {
        return $false
    }
    $state = Read-StateFile -Path $FrontendDependencyStateFile
    try {
        if ($null -eq $state -or [int](Get-StateValue -State $state -Name 'schemaVersion') -ne $DependencyStateSchemaVersion -or
            [string](Get-StateValue -State $state -Name 'nodeVersion') -ne $NodeVersion -or
            [string](Get-StateValue -State $state -Name 'fingerprint') -ne (Get-FrontendDependencyFingerprint)) {
            return $false
        }
        return $true
    } catch {
        return $false
    }
}

function Test-FrontendBuildCurrent {
    $buildPath = Join-Path $ClientDir 'dist\browser\index.html'
    if (-not (Test-Path -LiteralPath $buildPath -PathType Leaf)) {
        return $false
    }
    $state = Read-StateFile -Path $FrontendBuildStateFile
    try {
        if ($null -eq $state -or [int](Get-StateValue -State $state -Name 'schemaVersion') -ne $BuildStateSchemaVersion) {
            return $false
        }
        return [string](Get-StateValue -State $state -Name 'fingerprint') -eq (Get-FrontendBuildFingerprint)
    } catch {
        return $false
    }
}

function Sync-BackendDependencies {
    param(
        [switch]$RuntimesReady,
        [ValidateSet('Standard', 'Development')][string]$InstallationType = 'Standard',
        [ValidateSet('Base', 'ML')][string]$FeatureSet = 'Base'
    )
    if (-not $RuntimesReady) {
        Initialize-Runtimes
        Set-RuntimeEnvironment
    }
    $venvNeedsRebuild = -not (Test-Path -LiteralPath $VenvPython)
    if (-not $venvNeedsRebuild) {
        try {
            $venvNeedsRebuild = (Get-PythonVersion -PythonExe $VenvPython) -ne $PythonVersion
        } catch {
            $venvNeedsRebuild = $true
        }
    }
    if ($venvNeedsRebuild) {
        Write-Step "Recreating backend virtual environment for Python $PythonVersion"
        & $UvExe venv --clear --python $PythonExe $VenvDir
        Assert-LastExitCode "backend virtual-environment recreation"
    }
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
    if ($InstallationType -eq 'Development') {
        Write-Step "Installing Playwright Chromium browser"
        & $VenvPython -m playwright install chromium
        Assert-LastExitCode "Playwright Chromium installation"
        Write-Ok "Playwright Chromium is ready under $PlaywrightCacheDir."
    }
    Write-StateFile -Path $BackendDependencyStateFile -State ([pscustomobject][ordered]@{
        schemaVersion = $DependencyStateSchemaVersion
        fingerprint = Get-BackendDependencyFingerprint
        pythonVersion = $PythonVersion
        installationType = $InstallationType
        featureSet = $FeatureSet
    })
    Write-Ok "Python dependencies are ready ($InstallationType / $FeatureSet)."
}

function Install-FrontendDependencies {
    Write-Step "Installing frontend dependencies"
    Push-Location $ClientDir
    try {
        $lockfile = Join-Path $ClientDir 'package-lock.json'
        if (-not (Test-Path -LiteralPath $lockfile -PathType Leaf)) {
            throw "Missing frontend lockfile: $lockfile"
        }
        & $NpmCmd ci
        Assert-LastExitCode "npm dependency installation"
    } finally {
        Pop-Location
    }
    Write-StateFile -Path $FrontendDependencyStateFile -State ([pscustomobject][ordered]@{
        schemaVersion = $DependencyStateSchemaVersion
        fingerprint = Get-FrontendDependencyFingerprint
        nodeVersion = $NodeVersion
    })
    Write-Ok "Frontend dependencies are ready."
}

function Build-Frontend {
    $buildPath = Join-Path $ClientDir 'dist\browser\index.html'
    Remove-Item -LiteralPath $FrontendBuildStateFile -Force -ErrorAction SilentlyContinue
    Write-Step "Building frontend"
    Push-Location $ClientDir
    try {
        & $NpmCmd run build
        Assert-LastExitCode "frontend build"
    } finally {
        Pop-Location
    }
    if (-not (Test-Path -LiteralPath $buildPath -PathType Leaf)) {
        throw "Frontend build completed without producing $buildPath."
    }
    Write-StateFile -Path $FrontendBuildStateFile -State ([pscustomobject][ordered]@{
        schemaVersion = $BuildStateSchemaVersion
        fingerprint = Get-FrontendBuildFingerprint
    })
    Write-Ok "Frontend build is current."
}

function Sync-Dependencies {
    param(
        [switch]$BuildFrontend,
        [switch]$RuntimesReady,
        [ValidateSet('Standard', 'Development')][string]$InstallationType = 'Standard',
        [ValidateSet('Base', 'ML')][string]$FeatureSet = 'Base'
    )
    Import-Settings | Out-Null
    if (-not $RuntimesReady) { Initialize-Runtimes }
    Set-RuntimeEnvironment
    Sync-BackendDependencies -RuntimesReady -InstallationType $InstallationType -FeatureSet $FeatureSet
    Install-FrontendDependencies
    if ($BuildFrontend) { Build-Frontend }
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
        $processName = 'unknown'
        $processStartTime = $null
        $ownershipResolved = $false
        if ($null -ne $process) {
            try {
                $process.Refresh()
                $processName = [string]$process.ProcessName
                $processStartTime = [string]$process.StartTime.ToUniversalTime().Ticks
                $ownershipResolved = -not [string]::IsNullOrWhiteSpace($processName) -and
                    -not [string]::IsNullOrWhiteSpace($processStartTime)
            } catch { }
        }
        [pscustomobject]@{
            Id = [int]$processId
            Name = $processName
            ProcessStartTime = $processStartTime
            OwnershipResolved = $ownershipResolved
        }
    }
}

function Test-ActiveTcpListener([int]$Port) {
    try {
        $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
        return @($listeners | Where-Object { [int]$_.Port -eq $Port }).Count -gt 0
    }
    catch {
        return $false
    }
}

function Get-PortConflicts {
    param([Parameter(Mandatory)][int[]]$Ports)

    foreach ($port in @($Ports | Sort-Object -Unique)) {
        $owners = @(Get-ListeningProcess -Port $port)
        if ($owners.Count -gt 0) {
            foreach ($owner in $owners) {
                [pscustomobject]@{
                    Port = $port
                    Id = $owner.Id
                    Name = $owner.Name
                    ProcessStartTime = $owner.ProcessStartTime
                    OwnershipResolved = $owner.OwnershipResolved
                }
            }
            continue
        }
        if (Test-ActiveTcpListener -Port $port) {
            [pscustomobject]@{
                Port = $port
                Id = $null
                Name = 'unresolved owner'
                ProcessStartTime = $null
                OwnershipResolved = $false
            }
        }
    }
}

function Write-PortConflictSummary {
    param([Parameter(Mandatory)][pscustomobject[]]$Conflicts)

    Write-Host ""
    Write-Host "Configured ADSMOD ports are already in use:" -ForegroundColor Yellow
    foreach ($conflict in @($Conflicts | Sort-Object Port, Id)) {
        if ($conflict.OwnershipResolved) {
            Write-Host "  Port $($conflict.Port): PID $($conflict.Id) ($($conflict.Name))" -ForegroundColor Yellow
        } else {
            Write-Host "  Port $($conflict.Port): owner could not be resolved" -ForegroundColor Red
        }
    }
}

function Resolve-PortConflicts {
    param([pscustomobject[]]$Conflicts = @())

    $conflicts = @($Conflicts)
    if ($conflicts.Count -eq 0) {
        return $true
    }

    Write-PortConflictSummary -Conflicts $conflicts
    if (@($conflicts | Where-Object { -not $_.OwnershipResolved }).Count -gt 0) {
        Write-Warn "Launch cancelled because at least one listener owner could not be resolved. Stop the owning application and retry; ADSMOD will not terminate an unowned listener."
        return $false
    }
    if (-not $script:LauncherInteractive) {
        Write-Warn "Launch cancelled because this console is non-interactive. No conflicting process was terminated; stop the listed process(es) and retry."
        return $false
    }

    $processIds = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($conflict in $conflicts) {
        [void]$processIds.Add([int]$conflict.Id)
    }
    $processSummary = if ($processIds.Count -eq 1) { 'process listed above' } else { 'processes listed above' }
    $confirmation = ([string](Read-Host "Terminate the $($processIds.Count) $processSummary and continue? [y/N]")).Trim()
    if ($confirmation -notmatch '^(?i:y|yes)$') {
        Write-Warn "Launch cancelled. No conflicting process was terminated."
        return $false
    }

    $ports = @($conflicts | Select-Object -ExpandProperty Port | Sort-Object -Unique)
    $currentConflicts = @(Get-PortConflicts -Ports $ports)
    foreach ($port in $ports) {
        $originalForPort = @($conflicts | Where-Object { $_.Port -eq $port })
        $currentForPort = @($currentConflicts | Where-Object { $_.Port -eq $port })
        if ($currentForPort.Count -eq 0) {
            continue
        }
        if (@($currentForPort | Where-Object { -not $_.OwnershipResolved }).Count -gt 0) {
            Write-PortConflictSummary -Conflicts $currentForPort
            Write-Warn "Launch cancelled because port $port changed to an unresolved listener before termination."
            return $false
        }
        $originalIdentities = @($originalForPort | ForEach-Object { "$($_.Id)|$($_.Name)|$($_.ProcessStartTime)" } | Sort-Object)
        $currentIdentities = @($currentForPort | ForEach-Object { "$($_.Id)|$($_.Name)|$($_.ProcessStartTime)" } | Sort-Object)
        if ($originalIdentities.Count -ne $currentIdentities.Count -or
            (Compare-Object -ReferenceObject $originalIdentities -DifferenceObject $currentIdentities)) {
            Write-PortConflictSummary -Conflicts $currentForPort
            Write-Warn "Launch cancelled because ownership of port $port changed before termination."
            return $false
        }
    }

    $approved = [Collections.Generic.List[pscustomobject]]::new()
    $approvedIds = [System.Collections.Generic.HashSet[int]]::new()
    foreach ($current in $currentConflicts) {
        if ($approvedIds.Add([int]$current.Id)) {
            [void]$approved.Add($current)
        }
    }
    foreach ($approvedConflict in $approved) {
        $lastCheck = @(Get-PortConflicts -Ports @([int]$approvedConflict.Port))
        if ($lastCheck.Count -eq 0) {
            continue
        }
        if (@($lastCheck | Where-Object { -not $_.OwnershipResolved }).Count -gt 0) {
            Write-PortConflictSummary -Conflicts $lastCheck
            Write-Warn "Launch cancelled because the listener owner could not be resolved immediately before termination."
            return $false
        }
        $matching = @($lastCheck | Where-Object {
            [int]$_.Id -eq [int]$approvedConflict.Id -and
            [string]$_.Name -eq [string]$approvedConflict.Name -and
            [string]$_.ProcessStartTime -eq [string]$approvedConflict.ProcessStartTime
        })
        if ($matching.Count -eq 0) {
            Write-PortConflictSummary -Conflicts $lastCheck
            Write-Warn "Launch cancelled because port $($approvedConflict.Port) changed owner before termination."
            return $false
        }
        $process = Get-Process -Id ([int]$approvedConflict.Id) -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($null -eq $process) {
            continue
        }
        try {
            $process.Refresh()
            if ($process.HasExited) {
                continue
            }
            $currentName = [string]$process.ProcessName
            $currentStartTime = [string]$process.StartTime.ToUniversalTime().Ticks
            if ($currentName -ne [string]$approvedConflict.Name -or
                $currentStartTime -ne [string]$approvedConflict.ProcessStartTime) {
                Write-PortConflictSummary -Conflicts $lastCheck
                Write-Warn "Launch cancelled because the approved PID $($approvedConflict.Id) changed process identity before termination."
                return $false
            }
            Write-Step "Stopping approved port-conflict process $($approvedConflict.Name) (PID $($approvedConflict.Id))"
            $process.Kill($true)
            if (-not $process.WaitForExit(5000)) {
                throw "Process did not exit within five seconds."
            }
        }
        catch [System.InvalidOperationException] {
            # The approved process exited during the termination race.
        }
        catch {
            Write-Warn "Could not stop approved port-conflict PID $($approvedConflict.Id): $($_.Exception.Message)"
            $failureCheck = @(Get-PortConflicts -Ports $ports)
            if ($failureCheck.Count -gt 0) { Write-PortConflictSummary -Conflicts $failureCheck }
            return $false
        }
    }

    $remaining = @(Get-PortConflicts -Ports $ports)
    if ($remaining.Count -gt 0) {
        Write-PortConflictSummary -Conflicts $remaining
        Write-Warn "Launch cancelled because one or more configured ports remain occupied."
        return $false
    }
    Write-Ok "Configured ADSMOD ports are available."
    return $true
}

function Get-ApplicationProcessRecords {
    param([Parameter(Mandatory)][pscustomobject]$Settings)

    $records = @(Get-CimInstance -ClassName Win32_Process -ErrorAction SilentlyContinue)
    if ($records.Count -eq 0) {
        return @()
    }

    $backendMarker = [System.IO.Path]::GetFullPath($BackendDir).TrimEnd('\').ToLowerInvariant()
    $backendPortToken = "--port $($Settings.BackendPort)".ToLowerInvariant()
    $backendPortEqualsToken = "--port=$($Settings.BackendPort)".ToLowerInvariant()
    $matchedIds = [System.Collections.Generic.HashSet[int]]::new()

    foreach ($record in $records) {
        $commandLine = ([string]$record.CommandLine).Replace('/', '\').ToLowerInvariant()
        $isBackend = ($commandLine.Contains('server.cli') -and
                ($commandLine.Contains($backendMarker) -or
                    $commandLine.Contains($backendPortToken) -or
                    $commandLine.Contains($backendPortEqualsToken)))
        if ($isBackend) {
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
    foreach ($ownedProcess in @($script:BackendProcess)) {
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

    $script:BackendProcess = $null
    if ($stopped -gt 0) {
        Write-Ok "Stopped $stopped recognized ADSMOD application process(es)."
    }
    else {
        Write-Warn "No recognized ADSMOD application processes could be stopped."
    }
}

function Start-Application {
    $launchStopwatch = [Diagnostics.Stopwatch]::StartNew()

    $phaseStarted = $launchStopwatch.ElapsedMilliseconds
    try {
        $settings = Import-Settings
        $configuredPorts = @($settings.BackendPort)
        $portConflicts = @(Get-PortConflicts -Ports $configuredPorts)
        $portsReady = Resolve-PortConflicts -Conflicts $portConflicts
    } finally {
        Write-LaunchTiming -Phase 'Configuration and port preflight' -ElapsedMilliseconds ($launchStopwatch.ElapsedMilliseconds - $phaseStarted)
    }
    if (-not $portsReady) { return $false }

    $phaseStarted = $launchStopwatch.ElapsedMilliseconds
    try {
        if (-not (Test-PortableRuntimeFilesReady)) {
            Initialize-Runtimes
        }
        Set-RuntimeEnvironment
    } finally {
        Write-LaunchTiming -Phase 'Portable runtime setup' -ElapsedMilliseconds ($launchStopwatch.ElapsedMilliseconds - $phaseStarted)
    }

    $phaseStarted = $launchStopwatch.ElapsedMilliseconds
    try {
        if (-not (Test-BackendDependenciesCurrent)) {
            $profile = Get-RecordedBackendProfile
            Write-Step "Backend dependency state is missing or stale; repairing the recorded $($profile.InstallationType) / $($profile.FeatureSet) profile."
            Sync-BackendDependencies -RuntimesReady -InstallationType $profile.InstallationType -FeatureSet $profile.FeatureSet
        }
    } finally {
        Write-LaunchTiming -Phase 'Backend dependency check and setup' -ElapsedMilliseconds ($launchStopwatch.ElapsedMilliseconds - $phaseStarted)
    }

    $phaseStarted = $launchStopwatch.ElapsedMilliseconds
    try {
        if (-not (Test-FrontendDependenciesCurrent)) {
            Write-Step "Frontend dependency state is missing or stale; installing the locked package set."
            Install-FrontendDependencies
        }
    } finally {
        Write-LaunchTiming -Phase 'Frontend dependency check and setup' -ElapsedMilliseconds ($launchStopwatch.ElapsedMilliseconds - $phaseStarted)
    }

    $phaseStarted = $launchStopwatch.ElapsedMilliseconds
    try {
        if (-not (Test-FrontendBuildCurrent)) {
            Build-Frontend
        } else {
            Write-Ok "Application environments and frontend build are ready; skipped dependency installation and Angular build."
        }
    } finally {
        Write-LaunchTiming -Phase 'Frontend production build check' -ElapsedMilliseconds ($launchStopwatch.ElapsedMilliseconds - $phaseStarted)
    }

    $healthUrl = "$($settings.BackendUrl)/health/ready"
    Write-Step "Starting ADSMOD (backend and built frontend on $($settings.BackendUrl))"
    $phaseStarted = $launchStopwatch.ElapsedMilliseconds
    try {
        $script:BackendProcess = Start-Process -FilePath $VenvPython -ArgumentList @('-m', 'server.cli') -WorkingDirectory $RepoRoot -NoNewWindow -PassThru
    } finally {
        Write-LaunchTiming -Phase 'Backend process start' -ElapsedMilliseconds ($launchStopwatch.ElapsedMilliseconds - $phaseStarted)
    }
    try {
        $phaseStarted = $launchStopwatch.ElapsedMilliseconds
        try {
            try { Wait-ForHealth -Url $healthUrl -TimeoutSeconds 60 -Process $script:BackendProcess } catch { Stop-OwnedProcess -Process $script:BackendProcess -Name 'backend'; throw }
        } finally {
            Write-LaunchTiming -Phase 'Backend readiness check' -ElapsedMilliseconds ($launchStopwatch.ElapsedMilliseconds - $phaseStarted)
        }
        $launchStopwatch.Stop()
        Write-LaunchTiming -Phase 'Total time to application readiness' -ElapsedMilliseconds $launchStopwatch.ElapsedMilliseconds
        Start-Process -FilePath 'explorer.exe' -ArgumentList $settings.BackendUrl
        Write-Host ""
        Write-Ok "ADSMOD is running at $($settings.BackendUrl) (backend PID $($script:BackendProcess.Id))."
        Write-Host "Press Ctrl+C in this console to stop ADSMOD." -ForegroundColor Cyan
        while (-not $script:BackendProcess.WaitForExit(500)) { }
        Write-Warn "ADSMOD backend exited with code $($script:BackendProcess.ExitCode)."
    }
    finally {
        # Ctrl+C reaches the backend through the shared console, so give it time
        # to run its normal shutdown before the process tree is forced down.
        if ($null -ne $script:BackendProcess -and -not $script:BackendProcess.HasExited) {
            [void]$script:BackendProcess.WaitForExit(10000)
        }
        Stop-OwnedProcess -Process $script:BackendProcess -Name 'backend'
        $script:BackendProcess = $null
    }
    return $true
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
    Build-Frontend
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
        & $UvExe run --project $BackendDir --python $PythonExe python app/scripts/initialize_database.py
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

function Read-DotEnvStorageRoot {
    $dotenvPath = Join-Path (Join-Path $RepoRoot 'settings') '.env'
    if (-not (Test-Path -LiteralPath $dotenvPath -PathType Leaf)) { return $null }
    $storageRoot = $null
    foreach ($line in Get-Content -LiteralPath $dotenvPath -ErrorAction Stop) {
        $text = ([string]$line).Trim()
        if (-not $text -or $text.StartsWith('#') -or $text.StartsWith(';') -or -not $text.Contains('=')) { continue }
        $parts = $text.Split('=', 2)
        if ($parts[0].Trim() -ne 'ADSMOD_STORAGE_ROOT') { continue }
        $value = $parts[1].Trim()
        if ($value.Length -ge 2 -and (($value[0] -eq '"' -and $value[$value.Length - 1] -eq '"') -or ($value[0] -eq "'" -and $value[$value.Length - 1] -eq "'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        elseif ($value.Contains(' #')) {
            $value = $value.Split(' #', 2)[0].TrimEnd()
        }
        $storageRoot = $value
    }
    return $storageRoot
}

function Resolve-MaintenancePath {
    param([Parameter(Mandatory)][string]$Value, [Parameter(Mandatory)][string]$BasePath)
    $expanded = [Environment]::ExpandEnvironmentVariables($Value)
    $expanded = [regex]::Replace($expanded, '%([^%]+)%', [System.Text.RegularExpressions.MatchEvaluator]{
        param($match)
        $replacement = [Environment]::GetEnvironmentVariable($match.Groups[1].Value)
        if ($null -eq $replacement) { return $match.Value }
        return $replacement
    })
    $expanded = [regex]::Replace($expanded, '\$(?:\{(?<braced>[^}]+)\}|(?<simple>[A-Za-z_][A-Za-z0-9_]*))', [System.Text.RegularExpressions.MatchEvaluator]{
        param($match)
        $name = if ($match.Groups['braced'].Success) { $match.Groups['braced'].Value } else { $match.Groups['simple'].Value }
        $replacement = [Environment]::GetEnvironmentVariable($name)
        if ($null -eq $replacement) { return $match.Value }
        return $replacement
    })
    if ($expanded -eq '~' -or $expanded.StartsWith('~\') -or $expanded.StartsWith('~/')) {
        $expanded = Join-Path $HOME $expanded.Substring(1).TrimStart('\', '/')
    }
    if (-not [System.IO.Path]::IsPathRooted($expanded)) { $expanded = Join-Path $BasePath $expanded }
    return [System.IO.Path]::GetFullPath($expanded)
}

function Resolve-MaintenanceSettings {
    $processStorageRoot = [Environment]::GetEnvironmentVariable('ADSMOD_STORAGE_ROOT', 'Process')
    $configuredRoot = if ($null -ne $processStorageRoot) { $processStorageRoot } else { Read-DotEnvStorageRoot }
    if ([string]::IsNullOrWhiteSpace($configuredRoot)) {
        if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
            $configuredRoot = Join-Path $env:LOCALAPPDATA 'ADSMOD'
        }
        else {
            $configuredRoot = Join-Path $HOME 'AppData\Local\ADSMOD'
        }
    }

    $script:DataDir = Resolve-MaintenancePath -Value $configuredRoot -BasePath $RepoRoot
    $script:LogDir = Join-Path $script:DataDir 'logs'
    $script:CheckpointsDir = Join-Path $script:DataDir 'checkpoints'
    $script:MaintenanceDatabaseBackend = 'sqlite'
    $script:MaintenanceDatabasePath = Resolve-MaintenancePath -Value 'data\database.db' -BasePath $script:DataDir

    $databaseConfigPath = Join-Path $script:DataDir 'database.json'
    if (Test-Path -LiteralPath $databaseConfigPath -PathType Leaf) {
        try {
            $databaseConfig = Get-Content -LiteralPath $databaseConfigPath -Raw -ErrorAction Stop | ConvertFrom-Json -ErrorAction Stop
            $script:MaintenanceDatabaseBackend = [string]$databaseConfig.backend
            if ($script:MaintenanceDatabaseBackend -eq 'sqlite') {
                $sqlitePath = [string]$databaseConfig.connection.sqlite_path
                if ([string]::IsNullOrWhiteSpace($sqlitePath) -or $sqlitePath -eq ':memory:') {
                    $script:MaintenanceDatabaseBackend = 'unknown'
                    $script:MaintenanceDatabasePath = $null
                }
                else {
                    $script:MaintenanceDatabasePath = Resolve-MaintenancePath -Value $sqlitePath -BasePath $script:DataDir
                }
            }
            elseif ($script:MaintenanceDatabaseBackend -ne 'postgres') {
                $script:MaintenanceDatabaseBackend = 'unknown'
                $script:MaintenanceDatabasePath = $null
            }
            else {
                $script:MaintenanceDatabasePath = $null
            }
        }
        catch {
            $script:MaintenanceDatabaseBackend = 'unknown'
            $script:MaintenanceDatabasePath = $null
            Write-Warn "Database configuration could not be read; its database file will be preserved: $($_.Exception.Message)"
        }
    }
}

function Get-MaintenanceProcessBlockers {
    try {
        $processes = @(Get-CimInstance -ClassName Win32_Process -ErrorAction Stop)
    }
    catch {
        throw "Could not verify whether ADSMOD or its tests are running. Cleanup was stopped for safety: $($_.Exception.Message)"
    }

    $repoMarker = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\').Replace('/', '\').ToLowerInvariant()
    $knownExecutableRoots = @(
        [System.IO.Path]::GetFullPath($VenvDir).TrimEnd('\'),
        [System.IO.Path]::GetFullPath($PythonDir).TrimEnd('\'),
        [System.IO.Path]::GetFullPath($NodeDir).TrimEnd('\'),
        [System.IO.Path]::GetFullPath($UvDir).TrimEnd('\'),
        [System.IO.Path]::GetFullPath($PlaywrightCacheDir).TrimEnd('\')
    ) | ForEach-Object { $_.Replace('/', '\').ToLowerInvariant() }
    $markers = @('start_on_windows.ps1', 'server.cli', 'ng.js', 'dev-serve.mjs', 'pytest', 'run_tests.bat', 'playwright', 'vitest', 'npm-cli.js', 'uv.exe')
    $ownedRuntimeProcessNames = @('python.exe', 'pythonw.exe', 'node.exe', 'uv.exe', 'pytest.exe', 'ng.exe', 'npm.exe', 'chrome.exe', 'chromium.exe')
    $blockers = [Collections.Generic.List[object]]::new()
    foreach ($process in $processes) {
        if ([int]$process.ProcessId -eq $PID) { continue }
        $commandLine = ([string]$process.CommandLine).Replace('/', '\').ToLowerInvariant()
        $executablePath = ([string]$process.ExecutablePath).Replace('/', '\').ToLowerInvariant()
        $mentionsRepository = $commandLine.Contains($repoMarker) -or
            $commandLine.Contains('app\server') -or
            $commandLine.Contains('app\client') -or
            $commandLine.Contains('app\tests')
        $usesOwnedRuntime = $false
        foreach ($root in $knownExecutableRoots) {
            if ($executablePath.StartsWith($root + '\', [StringComparison]::OrdinalIgnoreCase)) { $usesOwnedRuntime = $true; break }
        }
        $hasLifecycleMarker = $false
        foreach ($marker in $markers) {
            if ($commandLine.Contains($marker)) { $hasLifecycleMarker = $true; break }
        }
        $ownedRuntimeProcess = $usesOwnedRuntime -and ([string]$process.Name).ToLowerInvariant() -in $ownedRuntimeProcessNames
        if ($ownedRuntimeProcess -or ($hasLifecycleMarker -and ($mentionsRepository -or $usesOwnedRuntime))) {
            [void]$blockers.Add([pscustomobject]@{ Id = [int]$process.ProcessId; Name = [string]$process.Name; CommandLine = [string]$process.CommandLine })
        }
    }
    return @($blockers)
}

function Assert-MaintenanceProcessesStopped {
    $blockers = @(Get-MaintenanceProcessBlockers)
    if ($blockers.Count -eq 0) { return }
    $processSummary = @($blockers | ForEach-Object { "PID $($_.Id) $($_.Name): $($_.CommandLine)" }) -join "`n - "
    throw "ADSMOD or a repository test process is running. Stop it before cleanup:`n - $processSummary"
}

function Write-CleanupPreview {
    param([Parameter(Mandatory)][pscustomobject]$Plan, [Parameter(Mandatory)][pscustomobject]$Preview)
    Write-Host "Cleanup inventory ($($Plan.Operation))" -ForegroundColor Cyan
    if ($Plan.Targets.Count -eq 0) { Write-Host '  No recognized targets found.' -ForegroundColor Gray }
    foreach ($target in $Plan.Targets) {
        $state = if ($target.Exists) { 'target' } else { 'missing' }
        Write-Host ("  [{0}] {1}: {2}" -f $state, $target.Category, $target.Path) -ForegroundColor Gray
    }
    if ($Plan.Operation -in @('AllData', 'Checkpoints')) {
        foreach ($path in $Preview.Removed) { Write-Host "    affected: $path" -ForegroundColor DarkGray }
    }
    foreach ($message in $Plan.Skipped) { Write-Warn "Skipped: $message" }
    foreach ($message in $Plan.Errors) { Write-Warn "Invalid target: $message" }
    foreach ($message in $Preview.Skipped) { Write-Warn "Skipped: $message" }
    foreach ($message in $Preview.Failed) { Write-Warn "Failed to inventory: $message" }
}

function Write-CleanupResult {
    param([Parameter(Mandatory)][pscustomobject]$Result)
    $verb = if ($Result.DryRun) { 'would clean' } else { 'cleaned' }
    if (-not $Result.DryRun) {
        foreach ($location in $Result.Locations) {
            Write-Host ("  [{0}] {1}: {2} ({3} item(s))" -f $verb, $location.Category, $location.Path, $location.Count) -ForegroundColor Gray
        }
    }
    foreach ($message in $Result.Skipped) { Write-Warn "Skipped: $message" }
    foreach ($message in $Result.Failed) { Write-Warn "Failed: $message" }
    Write-Host ("Cleanup summary: {0} {1}, {2} skipped, {3} failed." -f $Result.Removed.Count, $verb, $Result.Skipped.Count, $Result.Failed.Count) -ForegroundColor Cyan
}

function Invoke-MaintenanceCleanup {
    param(
        [Parameter(Mandatory)][ValidateSet('Cache', 'Uninstall', 'Logs', 'Checkpoints', 'AllData')][string]$Operation,
        [Parameter(Mandatory)][string]$Description
    )

    Resolve-MaintenanceSettings
    $plan = Get-AdsmodCleanupPlan -RepoRoot $RepoRoot -DataRoot $script:DataDir -DatabasePath $script:MaintenanceDatabasePath -DatabaseBackend $script:MaintenanceDatabaseBackend -Operation $Operation
    $preview = Invoke-AdsmodCleanupPlan -Plan $plan -DryRun -DetailedPreview:($Operation -in @('AllData', 'Checkpoints'))
    Write-CleanupPreview -Plan $plan -Preview $preview
    if ($plan.Errors.Count -gt 0) { throw "Cleanup was refused because one or more target paths failed validation." }
    if ($preview.Failed.Count -gt 0) { throw "Cleanup inventory was incomplete; no files were changed." }

    if ($DryRun) {
        Write-Info "Dry run completed. No files were changed."
        Write-CleanupResult -Result $preview
        return $true
    }

    Assert-MaintenanceProcessesStopped
    if (-not (Confirm-DestructiveAction $Description)) { return $false }

    Assert-MaintenanceProcessesStopped
    $result = Invoke-AdsmodCleanupPlan -Plan $plan
    Write-CleanupResult -Result $result
    if ($result.Failed.Count -gt 0) { throw "Cleanup was incomplete; see failed paths above." }
    return $true
}

function Remove-Logs {
    Write-Step 'Removing ADSMOD log files'
    [void](Invoke-MaintenanceCleanup -Operation Logs -Description 'remove ADSMOD log files')
}

function Clear-Cache {
    Write-Step 'Clearing ADSMOD runtime, backend, frontend, and test-tool caches'
    [void](Invoke-MaintenanceCleanup -Operation Cache -Description 'clear ADSMOD caches and generated frontend build output')
}

function Uninstall-Application {
    Write-Step 'Removing ADSMOD runtimes, dependencies, build output, caches, and logs'
    if (-not (Invoke-MaintenanceCleanup -Operation Uninstall -Description 'uninstall ADSMOD runtimes, dependencies, generated output, and logs')) { return }
    if (-not $DryRun) { Write-Ok 'ADSMOD software and generated logs removed. Database configuration, database, checkpoints, and training artifacts were preserved.' }
}

function Remove-Checkpoints {
    Write-Step 'Removing saved ADSMOD checkpoints'
    [void](Invoke-MaintenanceCleanup -Operation Checkpoints -Description 'remove saved ADSMOD checkpoints')
}

function Remove-All-Data {
    Write-Step 'Removing all local ADSMOD application data'
    if (-not (Invoke-MaintenanceCleanup -Operation AllData -Description 'remove all listed local ADSMOD data')) { return }
    if (-not $DryRun) {
        if ($script:MaintenanceDatabaseBackend -eq 'postgres') {
            Write-Ok 'Local ADSMOD data removed. The configured PostgreSQL database was left untouched.'
        }
        else {
            Write-Ok 'Local ADSMOD data removed. Settings return to defaults when a new database is initialized.'
        }
    }
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
        [pscustomobject]@{ Section = 'APPLICATION'; Label = 'Launch application'; Hint = 'Run ADSMOD here; Ctrl+C stops it'; Key = 'Launch'; Destructive = $false }
        [pscustomobject]@{ Section = 'APPLICATION'; Label = 'Kill all application processes'; Hint = 'Stop every recognized ADSMOD process, including previous sessions'; Key = 'KillAll'; Destructive = $true }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Install / update dependencies'; Hint = 'Refresh local runtimes and packages'; Key = 'Install'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Rebuild frontend'; Hint = 'Rebuild the generated frontend bundle'; Key = 'Rebuild'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Initialize database'; Hint = 'Create or upgrade the Alembic-managed data store'; Key = 'Database'; Destructive = $false }
        [pscustomobject]@{ Section = 'SETUP & VALIDATION'; Label = 'Run test suite'; Hint = 'Execute the repository checks'; Key = 'Tests'; Destructive = $false }
        [pscustomobject]@{ Section = 'SOURCE CONTROL'; Label = 'Check for updates'; Hint = 'Report whether origin/main has a newer version'; Key = 'Check'; Destructive = $false }
        [pscustomobject]@{ Section = 'SOURCE CONTROL'; Label = 'Update application'; Hint = 'Pull origin/main (clean main branch required)'; Key = 'Update'; Destructive = $false }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Remove logs'; Hint = 'Delete generated ADSMOD logs under the configured data root'; Key = 'Logs'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Clear cache'; Hint = 'Remove backend, frontend, runtime, and test-tool caches'; Key = 'Cache'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Remove checkpoints'; Hint = 'Delete saved training checkpoints after listing their location'; Key = 'Checkpoints'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Remove all data'; Hint = 'List and delete local database and user-generated files'; Key = 'AllData'; Destructive = $true }
        [pscustomobject]@{ Section = 'DATA & MAINTENANCE'; Label = 'Uninstall application'; Hint = 'Remove runtimes, dependencies, build output, and logs; preserve data'; Key = 'Uninstall'; Destructive = $true }
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
    if ($DryRun -and $entry.Key -notin @('Logs', 'Cache', 'Checkpoints', 'AllData', 'Uninstall')) {
        Write-Warn '-DryRun previews cleanup actions only. No other launcher action was run.'
        Wait-ForMenu
        continue
    }

    try {
        Invoke-TrackedLauncherAction -Name $entry.Label -Action {
            switch ($entry.Key) {
                'Launch' { [void](Start-Application) }
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
