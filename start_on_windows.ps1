[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$AppDir = Join-Path $RepoRoot "app"
$ServerDir = Join-Path $AppDir "server"
$ClientDir = Join-Path $AppDir "client"
$TestsDir = Join-Path $AppDir "tests"
$LogDir = Join-Path $AppDir "resources\logs"
$SettingsDir = Join-Path $RepoRoot "settings"
$RuntimesDir = Join-Path $RepoRoot "runtimes"
$StartupTempDir = Join-Path $ServerDir ".startup-temp"
$PythonDir = Join-Path $RuntimesDir "python"
$UvDir = Join-Path $RuntimesDir "uv"
$NodeDir = Join-Path $RuntimesDir "nodejs"

$PythonVersion = "3.14.2"
$PythonExe = Join-Path $PythonDir "python.exe"
$PythonPth = Join-Path $PythonDir "python314._pth"
$UvExe = Join-Path $UvDir "uv.exe"
$NodeExe = Join-Path $NodeDir "node.exe"
$NpmCmd = Join-Path $NodeDir "npm.cmd"
$VenvPython = Join-Path $ServerDir ".venv\Scripts\python.exe"
$UvCacheDir = Join-Path $StartupTempDir "uv-cache"
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

function Remove-RepoPath([string]$Path) {
    $repoPrefix = [System.IO.Path]::GetFullPath($RepoRoot).TrimEnd('\') + '\'
    $fullPath = [System.IO.Path]::GetFullPath($Path)
    if (-not $fullPath.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the repository: $fullPath"
    }
    if (Test-Path -LiteralPath $fullPath) {
        Remove-Item -LiteralPath $fullPath -Recurse -Force
    }
}

function Remove-UvCache {
    $expectedCachePath = [System.IO.Path]::GetFullPath((Join-Path $ServerDir ".startup-temp\uv-cache"))
    $actualCachePath = [System.IO.Path]::GetFullPath($UvCacheDir)
    if ($actualCachePath -ne $expectedCachePath) {
        throw "Refusing to remove an unexpected uv cache path: $actualCachePath"
    }
    if (Test-Path -LiteralPath $actualCachePath) {
        Remove-Item -LiteralPath $actualCachePath -Recurse -Force
    }
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
            Remove-Item -LiteralPath $nestedRoot -Force
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
    if ($content -match '(?m)^#import site\s*$') {
        $content = $content -replace '(?m)^#import site\s*$', 'import site'
        Set-Content -LiteralPath $Path -Value $content -Encoding utf8
    }
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
    $defaults = [ordered]@{
        FASTAPI_HOST = "127.0.0.1"
        FASTAPI_PORT = "6045"
        CORE_SERVICE_HOST = "127.0.0.1"
        CORE_SERVICE_PORT = "8000"
        ML_SERVICE_HOST = "127.0.0.1"
        ML_SERVICE_PORT = "8001"
        UI_HOST = "127.0.0.1"
        UI_PORT = "5173"
        OPTIONAL_DEPENDENCIES = "false"
        BACKEND_LOGS_VISIBLE = "true"
        ALWAYS_REBUILD = "true"
    }

    if (-not (Test-Path -LiteralPath $EnvFile)) {
        if (-not (Test-Path -LiteralPath $EnvExample)) {
            throw "Missing environment template: $EnvExample"
        }
        Copy-Item -LiteralPath $EnvExample -Destination $EnvFile
        Write-Ok "Created settings/.env from settings/.env.example."
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
        [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        if ($defaults.Contains($key)) {
            $defaults[$key] = $value
        }
    }

    if ($defaults.BACKEND_LOGS_VISIBLE -notmatch '^(true|false)$') {
        throw "BACKEND_LOGS_VISIBLE must be true or false."
    }
    if ($defaults.ALWAYS_REBUILD -notmatch '^(true|false)$') {
        throw "ALWAYS_REBUILD must be true or false."
    }

    if (-not $defaults.CORE_SERVICE_HOST) { $defaults.CORE_SERVICE_HOST = $defaults.FASTAPI_HOST }
    if (-not $defaults.CORE_SERVICE_PORT) { $defaults.CORE_SERVICE_PORT = $defaults.FASTAPI_PORT }
    return $defaults
}

function Set-RuntimeEnvironment {
    New-Item -ItemType Directory -Path $StartupTempDir -Force | Out-Null
    $env:UV_CACHE_DIR = $UvCacheDir
    $env:UV_NO_CACHE = "1"
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $ServerDir ".venv"
    $env:TEMP = $StartupTempDir
    $env:TMP = $StartupTempDir
    Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue
    $env:PATH = "$NodeDir;$UvDir;$env:PATH"
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

    Write-Step "Ensuring portable Node.js $NodeVersion"
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
    Set-RuntimeEnvironment
}

function Sync-Dependencies {
    param(
        [switch]$BuildFrontend,
        [switch]$AllowExistingEnvironmentFallback
    )

    $settings = Import-Settings
    Initialize-Runtimes
    Set-RuntimeEnvironment

    Write-Step "Syncing Python dependencies"
    Push-Location $ServerDir
    try {
        $arguments = @('sync', '--all-packages', '--python', $PythonExe, '--no-cache')
        if ($settings.OPTIONAL_DEPENDENCIES -eq 'true') {
            $arguments += '--all-extras'
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
    Sync-Dependencies -BuildFrontend:($settings.ALWAYS_REBUILD -eq 'true') -AllowExistingEnvironmentFallback
    Set-RuntimeEnvironment

    $backendPort = [int]$settings.FASTAPI_PORT
    $uiPort = [int]$settings.UI_PORT
    Stop-ListenerOnPort -Port $backendPort
    Stop-ListenerOnPort -Port $uiPort

    $backendArguments = @(
        '-m', 'uvicorn', 'app.server.app:app',
        '--host', $settings.FASTAPI_HOST,
        '--port', $settings.FASTAPI_PORT
    )
    Write-Step "Starting backend"
    $backendProcess = $null
    if ($settings.BACKEND_LOGS_VISIBLE -eq 'true') {
        $launchCommand = 'start "Backend" cmd /c ""{0}" -m uvicorn app.server.app:app --host {1} --port {2}"' -f `
            $VenvPython, $settings.FASTAPI_HOST, $settings.FASTAPI_PORT
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

    $healthUrl = "http://$($settings.FASTAPI_HOST):$($settings.FASTAPI_PORT)/api/health"
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

function Install-OrUpdate {
    Sync-Dependencies -BuildFrontend
    Remove-UvCache
    Write-Ok "Dependencies installed and frontend built successfully."
}

function Initialize-Database {
    Import-Settings | Out-Null
    Initialize-Runtimes
    Set-RuntimeEnvironment
    Write-Step "Initializing database"
    Push-Location $RepoRoot
    try {
        & $UvExe run --project app/server --python $PythonExe python app/scripts/initialize_database.py `
            --drop-existing --seed-catalogs --force-reseed-catalogs
        Assert-LastExitCode "database initialization"
    } finally {
        Pop-Location
    }
    Write-Ok "Database initialized successfully."
}

function Invoke-TestSuite {
    $testScript = Join-Path $TestsDir "run_tests.bat"
    if (-not (Test-Path -LiteralPath $testScript)) {
        throw "Missing test runner: $testScript"
    }
    Write-Step "Running test suite"
    & $testScript
    Assert-LastExitCode "test suite"
    Write-Ok "Test suite completed successfully."
}

function Remove-Logs {
    Write-Step "Removing log files"
    if (Test-Path -LiteralPath $LogDir) {
        Get-ChildItem -LiteralPath $LogDir -Filter '*.log' -File | Remove-Item -Force
    }
    Write-Ok "Log files removed."
}

function Clear-Cache {
    Write-Step "Clearing Python and uv caches"
    Get-ChildItem -LiteralPath $RepoRoot -Directory -Filter '__pycache__' -Recurse -Force -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending |
        ForEach-Object { Remove-RepoPath $_.FullName }
    Remove-UvCache
    Write-Ok "Caches cleared."
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
        (Join-Path $ClientDir 'dist'),
        (Join-Path $ClientDir 'package-lock.json'),
        (Join-Path $ServerDir 'uv.lock'),
        (Join-Path $RepoRoot 'uv.lock')
    )
    foreach ($path in $paths) {
        Remove-RepoPath $path
    }
    Get-ChildItem -LiteralPath $RepoRoot -Directory -Filter '__pycache__' -Recurse -Force -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending |
        ForEach-Object { Remove-RepoPath $_.FullName }
    Write-Ok "Application artifacts removed. Settings, resources, database, and user data were preserved."
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
    Write-Host "  WORKSPACE" -ForegroundColor DarkCyan
    Write-MenuItem -Number '1' -Label 'Launch application' -Hint 'Start the local web workspace'
    Write-MenuItem -Number '2' -Label 'Install / update dependencies' -Hint 'Refresh local runtimes and packages'
    Write-MenuItem -Number '3' -Label 'Initialize database' -Hint 'Prepare the local data store'
    Write-MenuItem -Number '4' -Label 'Run test suite' -Hint 'Execute the repository checks'
    Write-Host ""
    Write-Host "  MAINTENANCE" -ForegroundColor DarkCyan
    Write-MenuItem -Number '5' -Label 'Remove logs' -Hint 'Delete generated log files' -NumberColor Yellow
    Write-MenuItem -Number '6' -Label 'Clear cache' -Hint 'Remove Python and uv caches' -NumberColor Yellow
    Write-MenuItem -Number '7' -Label 'Uninstall application' -Hint 'Remove local runtimes and build artifacts' -NumberColor Yellow
    Write-Host ""
    Write-Host $subtleRule -ForegroundColor DarkGray
    Write-MenuItem -Number '8' -Label 'Exit' -Hint 'Close this console' -NumberColor DarkGray
    Write-Host $rule -ForegroundColor DarkCyan
}

$exitMenu = $false
while (-not $exitMenu) {
    Show-MainMenu
    $selection = Read-Host "  Select an option [1-8]"

    if ($selection -notmatch '^[1-8]$') {
        Write-Warn "Please select a number from 1 to 8."
        Wait-ForMenu
        continue
    }

    try {
        switch ($selection) {
            '1' {
                Start-Application
                exit 0
            }
            '2' { Install-OrUpdate; Wait-ForMenu }
            '3' { Initialize-Database; Wait-ForMenu }
            '4' { Invoke-TestSuite; Wait-ForMenu }
            '5' { Remove-Logs; Wait-ForMenu }
            '6' { Clear-Cache; Wait-ForMenu }
            '7' { Uninstall-Application; Wait-ForMenu }
            '8' { $exitMenu = $true }
        }
    } catch {
        Write-Fatal $_.Exception.Message
        if ($selection -eq '1') {
            exit 1
        }
        Wait-ForMenu
    }
}
