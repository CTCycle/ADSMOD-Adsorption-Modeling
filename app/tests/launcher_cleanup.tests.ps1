[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$modulePath = Join-Path (Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts') 'launcher_cleanup.psm1'
Import-Module $modulePath -Force -ErrorAction Stop

function Assert-True {
    param([Parameter(Mandatory)][bool]$Condition, [Parameter(Mandatory)][string]$Message)
    if (-not $Condition) { throw "Assertion failed: $Message" }
}

function New-TestFile {
    param([Parameter(Mandatory)][string]$Path, [string]$Content = 'test')
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    Set-Content -LiteralPath $Path -Value $Content -NoNewline
}

function New-DataFixture {
    param([Parameter(Mandatory)][string]$Repository, [string]$DataRoot = '')
    if (-not $DataRoot) { $DataRoot = Join-Path $Repository 'data' }
    New-Item -ItemType Directory -Path $Repository -Force | Out-Null
    New-TestFile (Join-Path $DataRoot 'database.json') '{"backend":"sqlite","connection":{"sqlite_path":"external/adsmod.sqlite"}}'
    New-TestFile (Join-Path $DataRoot 'logs\ADSMOD.log') 'log'
    New-TestFile (Join-Path $DataRoot 'checkpoints\model.ckpt') 'checkpoint'
    New-TestFile (Join-Path $DataRoot 'training\manifest.json') '{}'
    New-TestFile (Join-Path $DataRoot 'training\training-manifest.tmp') '{}'
    New-TestFile (Join-Path $Repository 'runtimes\.gitkeep') ''
    New-TestFile (Join-Path $Repository 'runtimes\python\python.exe') 'portable python runtime'
    New-TestFile (Join-Path $Repository 'runtimes\python\Lib\__pycache__\runtime.pyc') 'portable runtime bytecode'
    New-TestFile (Join-Path $Repository 'runtimes\cache\CACHEDIR.TAG') 'sentinel'
    New-TestFile (Join-Path $Repository 'runtimes\cache\pytest\state.bin') 'cache'
    New-TestFile (Join-Path $Repository '.ruff_cache\state') 'cache'
    New-TestFile (Join-Path $Repository 'app\server\package\__pycache__\module.pyc') 'bytecode'
    New-TestFile (Join-Path $Repository 'app\server\models\training\__pycache__\trainer.pyc') 'bytecode'
    New-TestFile (Join-Path $Repository 'data\__pycache__\user-state.pyc') 'persistent data'
    New-TestFile (Join-Path $Repository 'app\tests\cache\pytest-tmp-old\state') 'temp'
    New-TestFile (Join-Path $Repository 'app\client\.angular\cache\state') 'angular'
    New-TestFile (Join-Path $Repository 'app\client\node_modules\.cache\vitest\state') 'vitest'
    New-TestFile (Join-Path $Repository 'app\client\out-tsc\app\main.js') 'compiler output'
    New-TestFile (Join-Path $Repository 'app\client\coverage\lcov-report\index.html') 'coverage'
    New-TestFile (Join-Path $Repository 'app\client\dist\browser\index.html') 'build'
    New-TestFile (Join-Path $Repository 'settings\.env-stale.tmp') 'interrupted config update'
    New-TestFile (Join-Path $Repository 'assets\QA\core-visual-artifacts\keep.png') 'evidence'
    New-TestFile (Join-Path $Repository 'app\client\assets\QA\validation\keep.txt') 'evidence'
    New-TestFile (Join-Path $DataRoot '.database-stale.tmp') 'temporary config'
    New-TestFile (Join-Path $Repository 'runtimes\uv\uv.zip') 'interrupted download'
    New-TestFile (Join-Path $Repository 'runtimes\.python-3.14-staging\python-3.14.zip') 'staging'
    return $DataRoot
}

$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("adsmod-cleanup-tests-{0}" -f [guid]::NewGuid().ToString('N'))
try {
    $repo = Join-Path $testRoot 'repo'
    $data = New-DataFixture -Repository $repo
    $outside = Join-Path $testRoot 'outside'
    New-TestFile (Join-Path $outside '.ruff_cache\outside.txt') 'outside'

    $linkPath = Join-Path $repo 'app\server\linked-cache'
    try {
        New-Item -ItemType Junction -Path $linkPath -Target (Join-Path $outside '.ruff_cache') -ErrorAction Stop | Out-Null
    }
    catch {
        $linkPath = $null
    }

    Push-Location $testRoot
    try {
        $cachePlan = Get-AdsmodCleanupPlan -RepoRoot $repo -DataRoot $data -Operation Cache
    }
    finally { Pop-Location }

    $cacheTargetPaths = @($cachePlan.Targets | ForEach-Object { $_.Path })
    Assert-True ($cacheTargetPaths -contains (Join-Path $repo '.ruff_cache')) 'root hidden Ruff cache is discovered'
    Assert-True ($cacheTargetPaths -contains (Join-Path $repo 'app\server\package\__pycache__')) 'deep Python bytecode cache is discovered'
    Assert-True ($cacheTargetPaths -contains (Join-Path $repo 'app\server\models\training\__pycache__')) 'Python caches in the training source package are discovered'
    Assert-True ($cacheTargetPaths -contains (Join-Path $repo 'runtimes\python\Lib\__pycache__')) 'Python caches inside the portable runtime are discovered'
    Assert-True (-not ($cacheTargetPaths -contains (Join-Path $repo 'data\__pycache__'))) 'cache-like directories under the repository data root are excluded'
    Assert-True ($cacheTargetPaths -contains (Join-Path $repo 'app\client\node_modules\.cache')) 'frontend package cache is discovered'
    Assert-True ($cacheTargetPaths -contains (Join-Path $repo 'app\client\dist')) 'frontend build output is discovered'
    Assert-True ($cacheTargetPaths -contains (Join-Path $repo 'app\client\out-tsc')) 'frontend TypeScript output is discovered'
    Assert-True ($cacheTargetPaths -contains (Join-Path $repo 'app\client\coverage')) 'frontend coverage output is discovered'
    Assert-True ($cacheTargetPaths -contains (Join-Path $repo 'settings\.env-stale.tmp')) 'stale settings temporary file is discovered'
    Assert-True ($cacheTargetPaths -contains (Join-Path $data 'training\training-manifest.tmp')) 'stale training manifest temporary file is discovered'
    Assert-True (@($cachePlan.Targets | Where-Object { $_.Path -eq (Join-Path $repo 'app\client\node_modules\.vite') -and -not $_.Exists }).Count -gt 0) 'missing frontend cache paths are safe no-op targets'
    Assert-True (-not (@($cacheTargetPaths | Where-Object { $_ -match '[\\/]assets[\\/]QA([\\/]|$)' }).Count -gt 0)) 'QA assets are excluded from launcher cleanup'
    if ($linkPath) {
        Assert-True (@($cachePlan.Skipped | Where-Object { $_.Contains($linkPath) }).Count -gt 0) 'junction is reported and not traversed'
    }

    $dryRun = Invoke-AdsmodCleanupPlan -Plan $cachePlan -DryRun -DetailedPreview
    Assert-True (Test-Path (Join-Path $repo '.ruff_cache\state')) 'dry-run leaves caches intact'
    Assert-True (Test-Path (Join-Path $data 'database.json')) 'dry-run leaves persistent configuration intact'
    Assert-True (@($dryRun.Removed | Where-Object { $_ -eq (Join-Path $repo 'app\client\dist\browser\index.html') }).Count -gt 0) 'dry-run inventories nested build output'

    $lockedPath = Join-Path $repo 'app\server\locked\__pycache__\locked.pyc'
    New-TestFile $lockedPath 'locked bytecode'
    $lockedPlan = Get-AdsmodCleanupPlan -RepoRoot $repo -DataRoot $data -Operation Cache
    $lockedStream = [System.IO.File]::Open($lockedPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    try {
        $lockedResult = Invoke-AdsmodCleanupPlan -Plan $lockedPlan
        Assert-True (($lockedResult.Failed.Count -gt 0) -or (-not (Test-Path $lockedPath))) 'locked file is either removed safely or reported as a failure'
    }
    finally { $lockedStream.Dispose() }
    if (Test-Path $lockedPath) {
        $retryPlan = Get-AdsmodCleanupPlan -RepoRoot $repo -DataRoot $data -Operation Cache
        $retryResult = Invoke-AdsmodCleanupPlan -Plan $retryPlan
        Assert-True ($retryResult.Failed.Count -eq 0) 'failed locked-file cleanup can be safely retried'
    }

    $restrictedPath = Join-Path $repo 'app\server\restricted\__pycache__\blocked.pyc'
    New-TestFile $restrictedPath 'restricted bytecode'
    $restrictedRoot = Split-Path -Parent $restrictedPath
    $restrictedPlan = Get-AdsmodCleanupPlan -RepoRoot $repo -DataRoot $data -Operation Cache
    $originalAcl = $null
    $permissionReady = $false
    try {
        $originalAcl = Get-Acl -LiteralPath $restrictedRoot -ErrorAction Stop
        $principal = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        $deny = [System.Security.AccessControl.FileSystemAccessRule]::new(
            $principal,
            ([System.Security.AccessControl.FileSystemRights]::Delete -bor [System.Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles),
            ([System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor [System.Security.AccessControl.InheritanceFlags]::ObjectInherit),
            [System.Security.AccessControl.PropagationFlags]::None,
            [System.Security.AccessControl.AccessControlType]::Deny
        )
        $restrictedAcl = Get-Acl -LiteralPath $restrictedRoot
        [void]$restrictedAcl.AddAccessRule($deny)
        Set-Acl -LiteralPath $restrictedRoot -AclObject $restrictedAcl -ErrorAction Stop
        $permissionReady = $true
    }
    catch {
        Write-Host "Permission ACL check skipped on this account: $($_.Exception.Message)" -ForegroundColor DarkYellow
    }
    if ($permissionReady) {
        try {
        $permissionResult = Invoke-AdsmodCleanupPlan -Plan $restrictedPlan
            $pathWasReported = @($permissionResult.Failed + $permissionResult.Skipped | Where-Object { $_.Contains($restrictedRoot) }).Count -gt 0
            Assert-True ($pathWasReported -or (-not (Test-Path $restrictedPath))) 'permission denial is reported without aborting cleanup'
        }
        finally {
            if ($null -ne $originalAcl) { Set-Acl -LiteralPath $restrictedRoot -AclObject $originalAcl -ErrorAction SilentlyContinue }
        }
    }
    if (Test-Path $restrictedPath) {
        $permissionRetry = Get-AdsmodCleanupPlan -RepoRoot $repo -DataRoot $data -Operation Cache
        $permissionRetryResult = Invoke-AdsmodCleanupPlan -Plan $permissionRetry
        Assert-True ($permissionRetryResult.Failed.Count -eq 0) 'permission-denied cleanup succeeds after access is restored'
    }

    $cacheResult = Invoke-AdsmodCleanupPlan -Plan $cachePlan
    Assert-True ($cacheResult.Failed.Count -eq 0) 'cache cleanup completes without an exception'
    Assert-True (-not (Test-Path (Join-Path $repo '.ruff_cache'))) 'root Ruff cache is removed'
    Assert-True (-not (Test-Path (Join-Path $repo 'app\client\dist'))) 'generated frontend build is removed'
    Assert-True (-not (Test-Path (Join-Path $repo 'app\client\out-tsc'))) 'generated frontend TypeScript output is removed'
    Assert-True (-not (Test-Path (Join-Path $repo 'app\client\coverage'))) 'generated frontend coverage output is removed'
    Assert-True (-not (Test-Path (Join-Path $repo 'runtimes\python\Lib\__pycache__'))) 'runtime cache cleanup removes portable Python bytecode'
    Assert-True (Test-Path (Join-Path $repo 'runtimes\python\python.exe')) 'runtime cache cleanup preserves the portable Python executable'
    Assert-True (-not (Test-Path (Join-Path $repo 'settings\.env-stale.tmp'))) 'stale settings temporary file is removed'
    Assert-True (-not (Test-Path (Join-Path $data 'training\training-manifest.tmp'))) 'stale training manifest temporary file is removed'
    Assert-True (Test-Path (Join-Path $repo 'runtimes\cache\CACHEDIR.TAG')) 'canonical cache sentinel is preserved'
    Assert-True (Test-Path (Join-Path $repo 'data\__pycache__\user-state.pyc')) 'persistent repository data is preserved even when it contains cache-like files'
    Assert-True (Test-Path (Join-Path $data 'database.json')) 'cache cleanup preserves database configuration'
    Assert-True (Test-Path (Join-Path $data 'checkpoints\model.ckpt')) 'cache cleanup preserves checkpoints'
    Assert-True (Test-Path (Join-Path $data 'logs\ADSMOD.log')) 'cache cleanup preserves logs'
    Assert-True (Test-Path (Join-Path $repo 'assets\QA\core-visual-artifacts\keep.png')) 'cache cleanup preserves QA output'
    Assert-True (Test-Path (Join-Path $outside '.ruff_cache\outside.txt')) 'linked external content is preserved'

    $repeatPlan = Get-AdsmodCleanupPlan -RepoRoot $repo -DataRoot $data -Operation Cache
    $repeatResult = Invoke-AdsmodCleanupPlan -Plan $repeatPlan
    Assert-True ($repeatResult.Failed.Count -eq 0) 'repeated cache cleanup is safe'

    foreach ($relative in @('runtimes\python\python.exe', 'runtimes\nodejs\node.exe', 'app\server\.venv\Scripts\python.exe', '.venv\Scripts\python.exe', 'app\client\node_modules\angular.bin')) {
        New-TestFile (Join-Path $repo $relative) 'runtime'
    }
    New-TestFile (Join-Path $repo 'runtimes\orphaned-installer\leftover.tmp') 'orphaned launcher output'
    $preservedSqlite = Join-Path $data 'external\adsmod.sqlite'
    New-TestFile $preservedSqlite 'persistent sqlite database'
    $uninstallPlan = Get-AdsmodCleanupPlan -RepoRoot $repo -DataRoot $data -Operation Uninstall
    $uninstallResult = Invoke-AdsmodCleanupPlan -Plan $uninstallPlan
    Assert-True ($uninstallResult.Failed.Count -eq 0) 'standard uninstall completes in the disposable repository'
    Assert-True (-not (Test-Path (Join-Path $repo 'runtimes\python'))) 'uninstall removes portable runtimes'
    Assert-True (-not (Test-Path (Join-Path $repo 'runtimes\orphaned-installer'))) 'uninstall removes orphaned launcher files under the owned runtimes directory'
    Assert-True (Test-Path (Join-Path $repo 'runtimes\.gitkeep')) 'uninstall preserves the runtimes repository sentinel'
    Assert-True (-not (Test-Path (Join-Path $repo 'app\client\node_modules'))) 'uninstall removes frontend dependencies'
    Assert-True (-not (Test-Path (Join-Path $data 'logs\ADSMOD.log'))) 'uninstall removes application logs'
    Assert-True (Test-Path (Join-Path $data 'database.json')) 'uninstall preserves database configuration'
    Assert-True (Test-Path $preservedSqlite) 'uninstall preserves the configured SQLite database'
    Assert-True (Test-Path (Join-Path $data 'checkpoints\model.ckpt')) 'uninstall preserves checkpoints'
    Assert-True (Test-Path (Join-Path $data 'training\manifest.json')) 'uninstall preserves training artifacts'

    $completeRepo = Join-Path $testRoot 'complete-repo'
    $completeData = New-DataFixture -Repository $completeRepo
    $sqlitePath = Join-Path $completeRepo 'external\adsmod.sqlite'
    New-TestFile $sqlitePath 'sqlite'
    New-TestFile "$sqlitePath-wal" 'wal'
    New-TestFile "$sqlitePath-shm" 'shm'
    New-TestFile "$sqlitePath-journal" 'journal'
    $completePlan = Get-AdsmodCleanupPlan -RepoRoot $completeRepo -DataRoot $completeData -DatabasePath $sqlitePath -DatabaseBackend sqlite -Operation AllData
    $completePreview = Invoke-AdsmodCleanupPlan -Plan $completePlan -DryRun -DetailedPreview
    Assert-True (@($completePreview.Removed | Where-Object { $_ -eq $sqlitePath }).Count -gt 0) 'complete-removal preview lists the configured SQLite database'
    Assert-True (Test-Path (Join-Path $completeData 'database.json')) 'complete-removal dry-run preserves files'
    $completeResult = Invoke-AdsmodCleanupPlan -Plan $completePlan
    Assert-True ($completeResult.Failed.Count -eq 0) 'complete local data removal succeeds in the disposable data root'
    Assert-True (-not (Test-Path (Join-Path $completeData 'database.json'))) 'complete removal deletes database configuration'
    Assert-True (-not (Test-Path $sqlitePath)) 'complete removal deletes selected SQLite database'
    Assert-True (-not (Test-Path "$sqlitePath-wal")) 'complete removal deletes SQLite WAL sidecar'
    Assert-True (-not (Test-Path "$sqlitePath-shm")) 'complete removal deletes SQLite shared-memory sidecar'
    Assert-True (-not (Test-Path "$sqlitePath-journal")) 'complete removal deletes SQLite rollback-journal sidecar'
    Assert-True (-not (Test-Path (Join-Path $completeData 'checkpoints'))) 'complete removal deletes checkpoints'
    Assert-True (-not (Test-Path (Join-Path $completeData 'training'))) 'complete removal deletes training artifacts'
    Assert-True (-not (Test-Path (Join-Path $completeData 'logs\ADSMOD.log'))) 'complete removal deletes application logs'
    Assert-True (Test-Path (Join-Path $completeRepo 'assets\QA\core-visual-artifacts\keep.png')) 'complete removal leaves QA assets alone'

    $postgresRepo = Join-Path $testRoot 'postgres-repo'
    $postgresData = Join-Path $postgresRepo 'data'
    New-DataFixture -Repository $postgresRepo -DataRoot $postgresData | Out-Null
    $remoteMarker = Join-Path $testRoot 'external-postgres-data.marker'
    New-TestFile $remoteMarker 'external database'
    $postgresPlan = Get-AdsmodCleanupPlan -RepoRoot $postgresRepo -DataRoot $postgresData -DatabaseBackend postgres -Operation AllData
    $postgresResult = Invoke-AdsmodCleanupPlan -Plan $postgresPlan
    Assert-True ($postgresResult.Failed.Count -eq 0) 'local data cleanup succeeds for PostgreSQL configuration'
    Assert-True (Test-Path $remoteMarker) 'external PostgreSQL data is untouched'

    $launcherRepo = Join-Path $testRoot 'launcher-repo'
    $launcherData = Join-Path $testRoot 'launcher-data'
    $launcherScript = Join-Path $launcherRepo 'start_on_windows.ps1'
    $launcherModule = Join-Path $launcherRepo 'app\scripts\launcher_cleanup.psm1'
    New-Item -ItemType Directory -Path (Split-Path -Parent $launcherModule) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'start_on_windows.ps1') -Destination $launcherScript
    Copy-Item -LiteralPath (Join-Path (Split-Path -Parent $PSScriptRoot) 'scripts\launcher_cleanup.psm1') -Destination $launcherModule
    New-DataFixture -Repository $launcherRepo -DataRoot $launcherData | Out-Null
    $launcherSqlite = Join-Path $launcherData 'external\adsmod.sqlite'
    New-TestFile $launcherSqlite 'sqlite'
    $previousStorageRoot = [Environment]::GetEnvironmentVariable('ADSMOD_STORAGE_ROOT', 'Process')
    try {
        [Environment]::SetEnvironmentVariable('ADSMOD_STORAGE_ROOT', $launcherData, 'Process')
        Push-Location $testRoot
        try {
            $menuInput = "9`n10`n11`n12`n13`n14`n"
            $launcherOutput = @($menuInput | & pwsh -NoLogo -NoProfile -File $launcherScript -DryRun 2>&1)
            $launcherExitCode = $LASTEXITCODE
        }
        finally { Pop-Location }
    }
    finally { [Environment]::SetEnvironmentVariable('ADSMOD_STORAGE_ROOT', $previousStorageRoot, 'Process') }
    if ($launcherExitCode -ne 0) { throw "Disposable launcher menu DryRun exited with code $launcherExitCode`n$($launcherOutput | Select-Object -Last 30 | Out-String)" }
    foreach ($operation in @('Logs', 'Cache', 'Checkpoints', 'AllData', 'Uninstall')) {
        Assert-True (@($launcherOutput | Where-Object { ([string]$_).Contains("Cleanup inventory ($operation)") }).Count -gt 0) "launcher DryRun routes the $operation menu action"
    }
    Assert-True (@($launcherOutput | Where-Object { ([string]$_).Contains('Cleanup summary:') }).Count -eq 5) 'launcher DryRun completes all five cleanup menu actions'
    Assert-True (Test-Path (Join-Path $launcherData 'database.json')) 'launcher DryRun does not delete data configuration'
    Assert-True (Test-Path (Join-Path $launcherData 'checkpoints\model.ckpt')) 'launcher DryRun does not delete checkpoints'
    Assert-True (Test-Path $launcherSqlite) 'launcher DryRun does not delete the configured SQLite database'

    $nestedDataRepo = Join-Path $testRoot 'nested-data-repo'
    $nestedDataRoot = Join-Path $nestedDataRepo 'app\client\dist'
    New-DataFixture -Repository $nestedDataRepo -DataRoot $nestedDataRoot | Out-Null
    $nestedDataPlan = Get-AdsmodCleanupPlan -RepoRoot $nestedDataRepo -DataRoot $nestedDataRoot -Operation Cache
    Assert-True (-not (@($nestedDataPlan.Targets | Where-Object { $_.Path -eq $nestedDataRoot }).Count -gt 0)) 'cache cleanup refuses a generated-output target containing configured user data'
    $nestedDataResult = Invoke-AdsmodCleanupPlan -Plan $nestedDataPlan
    Assert-True ($nestedDataResult.Failed.Count -eq 0) 'cache cleanup handles data nested in a build-output path'
    Assert-True (Test-Path (Join-Path $nestedDataRoot 'database.json')) 'configured data nested below a generated-output path is preserved'

    $qaDataRepo = Join-Path $testRoot 'qa-data-repo'
    $qaDataRoot = Join-Path $qaDataRepo 'assets\QA\configured-data'
    New-DataFixture -Repository $qaDataRepo -DataRoot $qaDataRoot | Out-Null
    $qaDataPlan = Get-AdsmodCleanupPlan -RepoRoot $qaDataRepo -DataRoot $qaDataRoot -Operation AllData
    Assert-True (@($qaDataPlan.Targets | Where-Object { $_.Path -match '[\\/]assets[\\/]QA([\\/]|$)' }).Count -eq 0) 'complete data removal also refuses targets inside a QA directory'
    $qaDataResult = Invoke-AdsmodCleanupPlan -Plan $qaDataPlan
    Assert-True ($qaDataResult.Failed.Count -eq 0) 'QA-rooted data cleanup is safely skipped'
    Assert-True (Test-Path (Join-Path $qaDataRoot 'database.json')) 'QA-rooted configured data is preserved'

    $emptyRootPlan = Get-AdsmodCleanupPlan -RepoRoot '' -DataRoot $data -Operation Cache
    Assert-True ($emptyRootPlan.Errors.Count -gt 0) 'empty cleanup roots are refused without resolving to the current directory'

    Write-Host 'Launcher cleanup tests passed.' -ForegroundColor Green
}
finally {
    if (Test-Path -LiteralPath $testRoot) {
        Remove-Item -LiteralPath $testRoot -Force -Recurse -ErrorAction SilentlyContinue
    }
}
