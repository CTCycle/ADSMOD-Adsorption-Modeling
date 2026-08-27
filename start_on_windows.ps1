[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "app\scripts\ensure_environment.ps1")

$RepoRoot = $PSScriptRoot
$AppDir = Join-Path $RepoRoot "app"
$ServerDir = Join-Path $AppDir "server"
$BackendDir = Join-Path $AppDir "backend"
$ClientDir = Join-Path $AppDir "client"
$TestsDir = Join-Path $AppDir "tests"
$DefaultResourcesDir = Join-Path $AppDir "resources"
$ResourcesDir = $DefaultResourcesDir
$LogDir = Join-Path $ResourcesDir "logs"
$CheckpointsDir = Join-Path $ResourcesDir "checkpoints"
$SettingsDir = Join-Path $RepoRoot "settings"
$ConfigFile = Join-Path $ResourcesDir "adsmod.json"
$RuntimesDir = Join-Path $RepoRoot "runtimes"
$StartupTempDir = Join-Path $ServerDir ".startup-temp"
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

$PythonVersion = "3.14.2"
$PythonExe = Join-Path $PythonDir "python.exe"
$PythonPth = Join-Path $PythonDir "python314._pth"
$UvExe = Join-Path $UvDir "uv.exe"
$NodeExe = Join-Path $NodeDir "node.exe"
$NpmCmd = Join-Path $NodeDir "npm.cmd"
$VenvPython = Join-Path $ServerDir ".venv\Scripts\python.exe"
$UvCacheDir = $RuntimeCacheDir
$EnvFile = Join-Path $SettingsDir ".env"
$EnvExample = Join-Path $SettingsDir ".env.example"

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
    Write-Host "[STEP] $Message" -ForegroundColor Cyan
}

function Write-Ok([string]$Message) {
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Fatal([string]$Message) {
    Write-Host "[FATAL] $Message" -ForegroundColor Red
}

function Assert-LastExitCode([string]$Operation) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

function Resolve-ResourcesDirectory([string]$ConfiguredPath) {
    if ([string]::IsNullOrWhiteSpace($ConfiguredPath)) {
        return [System.IO.Path]::GetFullPath($DefaultResourcesDir)
    }

    $expandedPath = [Environment]::ExpandEnvironmentVariables($ConfiguredPath.Trim())
    if (-not [System.IO.Path]::IsPathRooted($expandedPath)) {
        $expandedPath = Join-Path $RepoRoot $expandedPath
    }
    return [System.IO.Path]::GetFullPath($expandedPath)
}

function Set-ConfiguredResourcePaths([string]$ConfiguredPath) {
    $script:ResourcesDir = Resolve-ResourcesDirectory $ConfiguredPath
    $script:LogDir = Join-Path $script:ResourcesDir "logs"
    $script:CheckpointsDir = Join-Path $script:ResourcesDir "checkpoints"
    $script:ConfigFile = Join-Path $script:ResourcesDir "adsmod.json"
    [Environment]::SetEnvironmentVariable(
        'ADSMOD_RESOURCES_DIR',
        $script:ResourcesDir,
        'Process'
    )
}

function Remove-RepoPath([string]$Path) {
    $repoPrefix = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\') + '\'
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the repository: $fullPath"
    }
    try {
        if (Test-Path -LiteralPath $fullPath) {
            Remove-Item -LiteralPath $fullPath -Recurse -Force -ErrorAction Stop
            return $true
        }
        return $true
    } catch {
        Write-Warn "Skipping locked or inaccessible path '$fullPath': $($_.Exception.Message)"
        return $false
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

        $items = @(Get-ChildItem -LiteralPath $fullPath -Recurse -Force -ErrorAction SilentlyContinue |
            Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true })
        foreach ($item in $items) {
            [void](Remove-RepoPath $item.FullName)
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
        if (Test-Path -LiteralPath $fullPath) {
            Remove-Item -LiteralPath $fullPath -Recurse -Force -ErrorAction Stop
        }
        return $true
    } catch {
        Write-Warn "Skipping locked or inaccessible user-data path '$fullPath': $($_.Exception.Message)"
        return $false
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

    foreach ($item in @(Get-ChildItem -LiteralPath $fullPath -Force -ErrorAction SilentlyContinue)) {
        if ($item.Name -eq '.gitkeep') {
            continue
        }
        [void](Remove-ResourcePath $item.FullName)
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
    $ProgressPreference = 'SilentlyContinue'
    New-Item -ItemType Directory -Path $DestinationPath -Force | Out-Null
    Invoke-WebRequest -Uri $Url -OutFile $ArchivePath
    try {
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $DestinationPath -Force
    } finally {
        Remove-Item -LiteralPath $ArchivePath -Force -ErrorAction SilentlyContinue
    }
    if ($FlattenSingleDirectory) {
        $children = @(Get-ChildItem -LiteralPath $DestinationPath -Force)
        if ($children.Count -eq 1 -and $children[0].PSIsContainer) {
            $nestedRoot = $children[0].FullName
            Get-ChildItem -LiteralPath $nestedRoot -Force | Move-Item -Destination $DestinationPath -Force
            Remove-Item -LiteralPath $nestedRoot -Force -ErrorAction SilentlyContinue
        }
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
    do {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
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
}

function Import-Settings {
    if (Ensure-EnvironmentFile -EnvFile $EnvFile -EnvExample $EnvExample) {
        Write-Ok "Created settings/.env from settings/.env.example."
    }

    $configuredResourcePath = $null
    foreach ($line in Get-Content -LiteralPath $EnvFile) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or $trimmed.StartsWith(';')) {
            continue
        }
        $separator = $trimmed.IndexOf('=')
        if ($separator -lt 1) {
            continue
        }
        $key = $trimmed.Substring(0, $separator).Trim()
        if ($key -eq 'ADSMOD_RESOURCES_DIR') {
            $configuredResourcePath = $trimmed.Substring($separator + 1).Trim()
            if (($configuredResourcePath.StartsWith('"') -and $configuredResourcePath.EndsWith('"')) -or
                ($configuredResourcePath.StartsWith("'") -and $configuredResourcePath.EndsWith("'"))) {
                $configuredResourcePath = $configuredResourcePath.Substring(1, $configuredResourcePath.Length - 2)
            }
            break
        }
    }
    if ($null -eq $configuredResourcePath) {
        $configuredResourcePath = [Environment]::GetEnvironmentVariable('ADSMOD_RESOURCES_DIR', 'Process')
    }
    Set-ConfiguredResourcePaths $configuredResourcePath

    if (-not (Test-Path -LiteralPath $ConfigFile)) {
        throw "Missing canonical configuration: $ConfigFile"
    }
    $canonical = Get-Content -LiteralPath $ConfigFile -Raw | ConvertFrom-Json
    if (-not $canonical.runtime) {
        throw "Canonical configuration is missing the runtime section: $ConfigFile"
    }
    $defaults = [ordered]@{
        BACKEND_HOST = [string]$canonical.runtime.host
        BACKEND_PORT = [string]$canonical.runtime.core_port
        UI_HOST = [string]$canonical.runtime.host
        UI_PORT = [string]$canonical.runtime.frontend_port
        BACKEND_LOGS_VISIBLE = "true"
    }

    foreach ($line in Get-Content -LiteralPath $EnvFile) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or $trimmed.StartsWith(';')) {
            continue
        }
        $separator = $trimmed.IndexOf('=')
        if ($separator -lt 1) {
            continue
        }
        $key = $trimmed.Substring(0, $separator).Trim()
        $value = $trimmed.Substring($separator + 1).Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if ($defaults.Contains($key)) {
            $defaults[$key] = $value
            [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        } elseif ($key -notin @('RELOAD', 'MPLBACKEND', 'KERAS_BACKEND', 'VITE_API_BASE_URL', 'ADSMOD_RESOURCES_DIR')) {
            throw "Unsupported setting '$key'. Runtime hosts and ports belong in the selected resource directory's adsmod.json."
        } else {
            [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        }
    }

    Set-ConfiguredResourcePaths $configuredResourcePath

    if ($defaults.BACKEND_LOGS_VISIBLE -notmatch '^(true|false)$') {
        throw "BACKEND_LOGS_VISIBLE must be true or false."
    }

    return $defaults
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
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $ServerDir ".venv"
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
        [switch]$BuildFrontend,
        [switch]$AllowExistingEnvironmentFallback
    )

    Write-Step "Installing frontend dependencies"
    Push-Location $ClientDir
    try {
        try {
            if (Test-Path -LiteralPath (Join-Path $ClientDir 'package-lock.json')) {
                & $NpmCmd ci
            } else {
                & $NpmCmd install
            }
            Assert-LastExitCode "npm dependency installation"
        } catch {
            if (-not $AllowExistingEnvironmentFallback -or -not (Test-Path -LiteralPath (Join-Path $ClientDir 'node_modules'))) {
                throw
            }
            Write-Warn "npm dependency installation could not update the existing node_modules tree; reusing it for startup. Run Install / update dependencies after resolving the filesystem lock."
        }

        if ($BuildFrontend) {
            Write-Step "Building frontend"
            try {
                & $NpmCmd run build
                Assert-LastExitCode "frontend build"
            } catch {
                $existingFrontend = Join-Path $ClientDir "dist\browser\index.html"
                if (-not $AllowExistingEnvironmentFallback -or -not (Test-Path -LiteralPath $existingFrontend)) {
                    throw
                }
                Write-Warn "Frontend rebuild could not update the existing checkout; reusing the existing dist bundle for startup. Run Install / update dependencies after resolving the filesystem lock."
            }
        }
    } finally {
        Pop-Location
    }
    Write-Ok "Frontend dependencies are ready."
}

function Sync-Dependencies {
    param(
        [switch]$BuildFrontend,
        [switch]$AllowExistingEnvironmentFallback,
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
    Push-Location $ServerDir
    try {
        $arguments = @('sync', '--all-packages', '--python', $PythonExe)
        if ($InstallationType -eq 'Development') {
            $arguments += '--group', 'dev'
        }
        else {
            $arguments += '--no-dev'
        }
        try {
            & $UvExe @arguments
            Assert-LastExitCode "uv sync"
        } catch {
            if (-not $AllowExistingEnvironmentFallback -or -not (Test-Path -LiteralPath $VenvPython)) {
                throw
            }
            Write-Warn "uv sync could not write its temporary files; reusing the existing backend environment for startup. Run Install / update dependencies after resolving the filesystem permission issue."
        }
    } finally {
        Pop-Location
    }
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "Backend virtual-environment Python was not created at $VenvPython."
    }
    Write-Ok "Python dependencies are ready."

    Sync-FrontendDependencies `
        -BuildFrontend:$BuildFrontend `
        -AllowExistingEnvironmentFallback:$AllowExistingEnvironmentFallback
}

function Test-DependenciesReady {
    $frontendPackage = Join-Path $ClientDir 'package.json'
    $frontendLock = Join-Path $ClientDir 'package-lock.json'
    $frontendModules = Join-Path $ClientDir 'node_modules'
    $frontendInstallState = Join-Path $frontendModules '.package-lock.json'
    $frontendRunner = Join-Path $frontendModules '@angular/cli/bin/ng.js'
    $frontendBuild = Join-Path $ClientDir 'dist\browser\index.html'
    $backendEntrypoint = Join-Path $AppDir 'server/app.py'

    if (-not (Test-Path -LiteralPath $PythonExe) -or
        -not (Test-Path -LiteralPath $UvExe) -or
        -not (Test-Path -LiteralPath $NodeExe) -or
        -not (Test-Path -LiteralPath $NpmCmd) -or
        -not (Test-Path -LiteralPath $VenvPython) -or
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
    & $VenvPython -c 'import fastapi, uvicorn' *> $null
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
        Sync-Dependencies -BuildFrontend -AllowExistingEnvironmentFallback
    } else {
        Write-Ok "Application environments are ready; skipped dependency installation."
    }
    Set-RuntimeEnvironment

    $backendPort = [int]$settings.BACKEND_PORT
    $uiPort = [int]$settings.UI_PORT
    Stop-ListenerOnPort -Port $backendPort
    Stop-ListenerOnPort -Port $uiPort

    $backendArguments = @(
        '-m', 'uvicorn', 'app.server.app:app',
        '--host', $settings.BACKEND_HOST,
        '--port', $settings.BACKEND_PORT
    )
    Write-Step "Starting backend"
    $backendProcess = $null
    if ($settings.BACKEND_LOGS_VISIBLE -eq 'true') {
        $launchCommand = 'start "Backend" cmd /c ""{0}" -m uvicorn app.server.app:app --host {1} --port {2}"' -f `
            $VenvPython, $settings.BACKEND_HOST, $settings.BACKEND_PORT
        Push-Location $RepoRoot
        try {
            & cmd.exe /d /c $launchCommand
            Assert-LastExitCode "visible backend launch"
        } finally {
            Pop-Location
        }
    } else {
        $backendProcess = Start-Process -FilePath $VenvPython `
            -ArgumentList $backendArguments `
            -WorkingDirectory $RepoRoot `
            -WindowStyle Hidden `
            -PassThru
    }

    $healthUrl = "http://$($settings.BACKEND_HOST):$($settings.BACKEND_PORT)/api/health"
    Write-Step "Waiting for backend health at $healthUrl"
    try {
        Wait-ForHealth -Url $healthUrl -TimeoutSeconds 60
    } catch {
        if ($backendProcess -and -not $backendProcess.HasExited) {
            Stop-Process -Id $backendProcess.Id -Force
        }
        throw
    }
    $backendPid = Get-ListenerPid -Port $backendPort

    Write-Step "Starting frontend preview"
    $frontendProcess = Start-Process -FilePath $NpmCmd `
        -ArgumentList @('run', 'preview', '--', '--host', $settings.UI_HOST, '--port', $settings.UI_PORT) `
        -WorkingDirectory $ClientDir `
        -WindowStyle Hidden `
        -PassThru

    $frontendUrl = "http://$($settings.UI_HOST):$($settings.UI_PORT)"
    try {
        Wait-ForHealth -Url $frontendUrl -TimeoutSeconds 60
    } catch {
        if (-not $frontendProcess.HasExited) { Stop-Process -Id $frontendProcess.Id -Force }
        throw
    }
    $frontendPid = Get-ListenerPid -Port $uiPort

    Start-Process $frontendUrl
    Write-Host ""
    Write-Ok "ADSMOD started successfully."
    Write-Host "Backend: $healthUrl (PID $backendPid)" -ForegroundColor Green
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
        & $UvExe run --project app/server --python $PythonExe python app/scripts/initialize_database.py
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
            ForEach-Object {
                try {
                    Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Stop
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
        @(Get-ChildItem -LiteralPath $ServerDir -Directory -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in $legacyCacheNames })
        @(Get-ChildItem -LiteralPath $BackendDir -Directory -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -in $legacyCacheNames })
    )
    $legacySearchRoots = @(
        (Join-Path $AppDir 'scripts'),
        (Join-Path $ServerDir 'core_service'),
        (Join-Path $ServerDir 'ml_service'),
        (Join-Path $ServerDir 'shared'),
        (Join-Path $ServerDir 'migrations'),
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
        Sort-Object @{ Expression = { $_.FullName.Length }; Descending = $true } -Unique)
    foreach ($legacyCacheDirectory in $legacyCacheDirectories) {
        Remove-RepoDirectoryContents $legacyCacheDirectory.FullName
        [void](Remove-RepoPath $legacyCacheDirectory.FullName)
    }

    $legacyToolCache = Join-Path $ClientDir '.angular'
    Remove-RepoDirectoryContents $legacyToolCache
    [void](Remove-RepoPath $legacyToolCache)

    Write-Ok "Cache cleanup completed; locked or inaccessible items were skipped."
}

function Uninstall-Application {
    Write-Step "Removing local application runtimes and build artifacts"
    $runtimeContents = @()
    if (Test-Path -LiteralPath $RuntimesDir) {
        $runtimeContents = @(Get-ChildItem -LiteralPath $RuntimesDir -Force |
            Where-Object { $_.Name -ne '.gitkeep' } |
            Select-Object -ExpandProperty FullName)
    }
    $paths = @($runtimeContents) + @(
        (Join-Path $ServerDir '.venv'),
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
        Sort-Object FullName -Descending |
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
        return [System.IO.Path]::GetFullPath((Join-Path $ResourcesDir 'database.db'))
    }

    $expandedPath = [Environment]::ExpandEnvironmentVariables($configuredPath.Trim())
    if ([System.IO.Path]::IsPathRooted($expandedPath)) {
        return [System.IO.Path]::GetFullPath($expandedPath)
    }

    $repositoryPath = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $expandedPath))
    $defaultDatabasePath = [System.IO.Path]::GetFullPath((Join-Path $DefaultResourcesDir 'database.db'))
    if ($repositoryPath -eq $defaultDatabasePath) {
        return [System.IO.Path]::GetFullPath((Join-Path $ResourcesDir 'database.db'))
    }
    return $repositoryPath
}

function Remove-DatabaseFiles {
    $databasePath = Get-ConfiguredDatabasePath
    if ($null -eq $databasePath) {
        Write-Warn "An external database is configured; local data was cleared, but the external database was not modified."
        return
    }

    $protectedPaths = @(
        [System.IO.Path]::GetFullPath($ConfigFile),
        [System.IO.Path]::GetFullPath((Join-Path $ResourcesDir 'adsmod.schema.json'))
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
    $confirmation = (Read-Host "Type REMOVE CHECKPOINTS to continue").Trim()
    if ($confirmation -cne 'REMOVE CHECKPOINTS') {
        Write-Warn "Remove Checkpoints cancelled."
        return
    }
    Clear-CheckpointFiles
}

function Remove-All-Data {
    Import-Settings | Out-Null
    Write-Warn "This removes the local database, uploaded dataset records, saved checkpoints, and generated logs."
    $confirmation = (Read-Host "Type REMOVE ALL DATA to continue").Trim()
    if ($confirmation -cne 'REMOVE ALL DATA') {
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

    Push-Location $RepoRoot
    try {
        & git @Arguments
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($exitCode -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $exitCode."
    }
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
        Write-Step "Switching to main"
        Invoke-Git -Arguments @('switch', 'main')
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
    Write-MenuItem -Number '7' -Label 'Update' -Hint 'Switch to main and pull the latest application version'
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
    } catch {
        Write-Fatal $_.Exception.Message
        if ($selection -eq '1') {
            exit 1
        }
        Wait-ForMenu
    }
}
