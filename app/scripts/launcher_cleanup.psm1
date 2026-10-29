Set-StrictMode -Version Latest

$script:CacheDirectoryNames = @(
    '__pycache__',
    '.pytest_cache',
    '.ruff_cache',
    '.mypy_cache',
    '.angular',
    '.vite',
    '.cache',
    '.npm',
    'htmlcov'
)

function Get-FullPathSafe {
    param([Parameter(Mandatory)][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { throw 'An empty filesystem path is never a valid cleanup target.' }
    return [System.IO.Path]::GetFullPath($Path)
}

function Test-PathWithin {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Root,
        [switch]$AllowRoot
    )

    $fullPath = Get-FullPathSafe $Path
    $fullRoot = (Get-FullPathSafe $Root).TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    if ($AllowRoot -and $fullPath.Equals($fullRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }
    $prefix = $fullRoot + [System.IO.Path]::DirectorySeparatorChar
    return $fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-IsFilesystemRoot {
    param([Parameter(Mandatory)][string]$Path)
    $fullPath = Get-FullPathSafe $Path
    $root = [System.IO.Path]::GetPathRoot($fullPath)
    return $fullPath.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar).Equals(
        $root.TrimEnd([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar),
        [System.StringComparison]::OrdinalIgnoreCase
    )
}

function Get-ReparseComponent {
    param([Parameter(Mandatory)][string]$Path)
    $cursor = Get-FullPathSafe $Path
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            try {
                $item = Get-Item -LiteralPath $cursor -Force -ErrorAction Stop
                if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) { return $cursor }
            }
            catch {
                return "$cursor ($($_.Exception.Message))"
            }
        }
        $parent = [System.IO.Path]::GetDirectoryName($cursor)
        if (-not $parent -or $parent.Equals($cursor, [System.StringComparison]::OrdinalIgnoreCase)) { break }
        $cursor = $parent
    }
    return $null
}

function New-CleanupPlan {
    param([Parameter(Mandatory)][ValidateSet('Cache', 'Uninstall', 'Logs', 'Checkpoints', 'AllData')][string]$Operation)
    return [ordered]@{
        Operation = $Operation
        Targets = [Collections.Generic.List[object]]::new()
        Skipped = [Collections.Generic.List[string]]::new()
        Errors = [Collections.Generic.List[string]]::new()
    }
}

function Add-CleanupTarget {
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary]$Plan,
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$ScopeRoot,
        [Parameter(Mandatory)][ValidateSet('File', 'Directory', 'Contents')][string]$Kind,
        [Parameter(Mandatory)][string]$Category,
        [string[]]$PreserveNames = @(),
        [switch]$AllowExternalExact,
        [switch]$KnownSafeAncestry
    )

    try {
        $fullPath = Get-FullPathSafe $Path
        $scope = Get-FullPathSafe $ScopeRoot
        if (Test-IsFilesystemRoot $fullPath) {
            [void]$Plan.Skipped.Add("$fullPath (filesystem roots are never cleanup targets)")
            return
        }
        if ([regex]::Split($fullPath, '[\\/]') -contains 'QA') {
            [void]$Plan.Skipped.Add("$fullPath (QA directories are excluded from launcher cleanup)")
            return
        }
        if ($Plan.Contains('DataRoot') -and $Plan.Operation -in @('Cache', 'Uninstall') -and
            (Test-PathWithin -Path $Plan.DataRoot -Root $fullPath -AllowRoot)) {
            [void]$Plan.Skipped.Add("$fullPath (target contains the configured storage root; persistent data is protected)")
            return
        }
        if ($KnownSafeAncestry) {
            if (Test-Path -LiteralPath $fullPath) {
                $attributes = [System.IO.File]::GetAttributes($fullPath)
                if ($attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                    [void]$Plan.Skipped.Add("$fullPath (target is a reparse point; not followed)")
                    return
                }
            }
        }
        else {
            $reparseComponent = Get-ReparseComponent $fullPath
            if ($reparseComponent) {
                [void]$Plan.Skipped.Add("$fullPath (path crosses a reparse point at '$reparseComponent')")
                return
            }
        }
        $inScope = Test-PathWithin -Path $fullPath -Root $scope -AllowRoot:($Kind -eq 'Contents')
        if (-not $inScope -and -not $AllowExternalExact) {
            [void]$Plan.Errors.Add("Refused out-of-scope cleanup path: $fullPath")
            return
        }
        if ($AllowExternalExact -and -not $fullPath.Equals($Path, [System.StringComparison]::OrdinalIgnoreCase)) {
            # The caller supplies only an exact configured database path; never a wildcard or parent.
            if ([System.IO.Path]::GetFileName($fullPath) -ne [System.IO.Path]::GetFileName($Path)) {
                [void]$Plan.Errors.Add("Refused non-exact external cleanup path: $Path")
                return
            }
        }

        $key = $fullPath.ToUpperInvariant()
        if (@($Plan.Targets | Where-Object { $_.Path.ToUpperInvariant() -eq $key }).Count -gt 0) { return }
        $exists = Test-Path -LiteralPath $fullPath
        if ($exists) {
            $item = Get-Item -LiteralPath $fullPath -Force -ErrorAction Stop
            if (($Kind -eq 'File' -and $item.PSIsContainer) -or ($Kind -in @('Directory', 'Contents') -and -not $item.PSIsContainer)) {
                [void]$Plan.Errors.Add("Refused cleanup target with unexpected item type: $fullPath")
                return
            }
        }
        [void]$Plan.Targets.Add([pscustomobject]@{
            Path = $fullPath
            ScopeRoot = $scope
            Kind = $Kind
            Category = $Category
            PreserveNames = @($PreserveNames)
            Exists = [bool]$exists
            AllowExternalExact = [bool]$AllowExternalExact
            KnownSafeAncestry = [bool]$KnownSafeAncestry
        })
    }
    catch {
        [void]$Plan.Errors.Add("Could not validate cleanup target '$Path': $($_.Exception.Message)")
    }
}

function Add-CacheDirectoryCandidates {
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary]$Plan,
        [Parameter(Mandatory)][string]$RepoRoot,
        [Parameter(Mandatory)][string]$DataRoot
    )

    $repo = Get-FullPathSafe $RepoRoot
    $data = Get-FullPathSafe $DataRoot
    $skipRootNames = @('.git', 'node_modules', 'QA')
    $queue = [Collections.Generic.Queue[string]]::new()
    $queue.Enqueue($repo)
    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        if (-not (Test-PathWithin -Path $current -Root $repo -AllowRoot)) {
            [void]$Plan.Skipped.Add("$current (directory scan escaped the repository scope)")
            continue
        }
        try { $currentAttributes = [System.IO.File]::GetAttributes($current) }
        catch {
            [void]$Plan.Skipped.Add("$current (directory attributes unavailable: $($_.Exception.Message))")
            continue
        }
        if ($currentAttributes -band [System.IO.FileAttributes]::ReparsePoint) {
            [void]$Plan.Skipped.Add("$current (directory became a reparse point; not followed)")
            continue
        }
        $children = @()
        try {
            $children = @([System.IO.Directory]::EnumerateDirectories($current))
        }
        catch {
            [void]$Plan.Skipped.Add("$current (directory scan failed: $($_.Exception.Message))")
            continue
        }

        try {
            foreach ($filePath in @([System.IO.Directory]::EnumerateFiles($current))) {
                $name = [System.IO.Path]::GetFileName($filePath)
                if ($name -notmatch '^\.coverage($|\.)' -and $name -notin @('coverage.xml') -and $name -notmatch '\.py[co]$') { continue }
                try {
                    $attributes = [System.IO.File]::GetAttributes($filePath)
                    if ($attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                        [void]$Plan.Skipped.Add("$filePath (reparse point not followed)")
                        continue
                    }
                }
                catch {
                    [void]$Plan.Skipped.Add("$filePath (attributes unavailable: $($_.Exception.Message))")
                    continue
                }
                Add-CleanupTarget -Plan $Plan -Path $filePath -ScopeRoot $repo -Kind File -Category 'Generated bytecode or coverage output' -KnownSafeAncestry
            }
        }
        catch {
            [void]$Plan.Skipped.Add("$current (file scan failed: $($_.Exception.Message))")
        }

        foreach ($childPath in $children) {
            $path = Get-FullPathSafe $childPath
            $name = [System.IO.Path]::GetFileName($path.TrimEnd('\', '/'))
            if ($name -in $skipRootNames) { continue }
            if ($path.Equals((Join-Path (Join-Path $repo 'runtimes') 'cache'), [StringComparison]::OrdinalIgnoreCase)) { continue }
            if ($Plan.Operation -eq 'Uninstall' -and $path.Equals((Join-Path $repo 'runtimes'), [StringComparison]::OrdinalIgnoreCase)) { continue }
            try { $attributes = [System.IO.File]::GetAttributes($path) }
            catch {
                [void]$Plan.Skipped.Add("$path (attributes unavailable: $($_.Exception.Message))")
                continue
            }
            if ($attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                [void]$Plan.Skipped.Add("$path (reparse point not followed)")
                continue
            }
            if (-not $data.Equals($repo, [System.StringComparison]::OrdinalIgnoreCase) -and
                (Test-PathWithin -Path $path -Root $data -AllowRoot)) { continue }
            if ($path.Equals((Join-Path $repo 'data'), [System.StringComparison]::OrdinalIgnoreCase) -or
                (Test-PathWithin -Path $path -Root (Join-Path $repo 'data'))) { continue }
            if ($data.Equals($repo, [System.StringComparison]::OrdinalIgnoreCase) -and
                [System.IO.Path]::GetDirectoryName($path).Equals($repo, [StringComparison]::OrdinalIgnoreCase) -and
                $name -in @('logs', 'checkpoints', 'training')) { continue }

            if ($name -in @('dist', 'build')) {
                if (Test-PathWithin -Path $path -Root (Join-Path $repo 'app\server')) {
                    Add-CleanupTarget -Plan $Plan -Path $path -ScopeRoot $repo -Kind Directory -Category 'Generated backend build output' -KnownSafeAncestry
                }
                continue
            }
            $isCacheName = $name -in $script:CacheDirectoryNames -or
                $name -like 'pytest-tmp*' -or
                $name -like 'pytest-cache*' -or
                $name -like '.pytest-*'
            $isTestCache = $name -eq 'cache' -and (Test-PathWithin -Path $path -Root (Join-Path $repo 'app\tests'))
            $isBackendBuildOutput = (Test-PathWithin -Path $path -Root (Join-Path $repo 'app\server')) -and
                ($name -in @('build', 'dist') -or $name -like '*.egg-info')
            if ($isCacheName -or $isTestCache -or $isBackendBuildOutput) {
                Add-CleanupTarget -Plan $Plan -Path $path -ScopeRoot $repo -Kind Directory -Category 'Python, test, or framework cache' -KnownSafeAncestry
                continue
            }
            $queue.Enqueue($path)
        }
    }
}

function Add-DatabaseTemporaryTargets {
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary]$Plan,
        [Parameter(Mandatory)][string]$DataRoot,
        [switch]$IncludeTrainingTemporary
    )
    if (-not (Test-Path -LiteralPath $DataRoot -PathType Container)) { return }
    try {
        foreach ($temporary in @(Get-ChildItem -LiteralPath $DataRoot -Filter '.database-*.tmp' -File -Force -ErrorAction Stop)) {
            Add-CleanupTarget -Plan $Plan -Path $temporary.FullName -ScopeRoot $DataRoot -Kind File -Category 'Stale database configuration temporary file'
        }
    }
    catch {
        [void]$Plan.Skipped.Add("$DataRoot (temporary database file scan failed: $($_.Exception.Message))")
    }

    if ($IncludeTrainingTemporary) {
        $trainingTemporary = Join-Path (Join-Path (Get-FullPathSafe $DataRoot) 'training') 'training-manifest.tmp'
        Add-CleanupTarget -Plan $Plan -Path $trainingTemporary -ScopeRoot $DataRoot -Kind File -Category 'Stale training manifest temporary file'
    }
}

function Add-SettingsTemporaryTargets {
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary]$Plan,
        [Parameter(Mandatory)][string]$RepoRoot
    )

    $settingsRoot = Join-Path (Get-FullPathSafe $RepoRoot) 'settings'
    if (-not (Test-Path -LiteralPath $settingsRoot -PathType Container)) { return }
    $reparseComponent = Get-ReparseComponent $settingsRoot
    if ($reparseComponent) {
        [void]$Plan.Skipped.Add("$settingsRoot (settings temporary scan crosses a reparse point at '$reparseComponent')")
        return
    }
    try {
        foreach ($file in @(Get-ChildItem -LiteralPath $settingsRoot -Filter '.env-*.tmp' -File -Force -ErrorAction Stop)) {
            Add-CleanupTarget -Plan $Plan -Path $file.FullName -ScopeRoot $RepoRoot -Kind File -Category 'Stale settings temporary file' -KnownSafeAncestry
        }
    }
    catch {
        [void]$Plan.Skipped.Add("$settingsRoot (settings temporary scan failed: $($_.Exception.Message))")
    }
}

function Add-InstallerTemporaryTargets {
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary]$Plan,
        [Parameter(Mandatory)][string]$RepoRoot
    )
    $repo = Get-FullPathSafe $RepoRoot
    $runtimes = Join-Path $repo 'runtimes'
    $reparseComponent = Get-ReparseComponent $runtimes
    if ($reparseComponent) {
        [void]$Plan.Skipped.Add("$runtimes (runtime directory crosses a reparse point at '$reparseComponent')")
        return
    }
    foreach ($directory in @('python', 'uv', 'nodejs')) {
        $runtimePath = Join-Path $runtimes $directory
        if (-not (Test-Path -LiteralPath $runtimePath -PathType Container)) { continue }
        $runtimeReparse = Get-ReparseComponent $runtimePath
        if ($runtimeReparse) {
            [void]$Plan.Skipped.Add("$runtimePath (runtime scan crosses a reparse point at '$runtimeReparse')")
            continue
        }
        try {
            foreach ($file in @(Get-ChildItem -LiteralPath $runtimePath -File -Force -ErrorAction Stop)) {
                $isInstallerArchive = ($directory -eq 'uv' -and $file.Name -eq 'uv.zip') -or
                    ($directory -eq 'python' -and $file.Name -like 'python-*.zip') -or
                    ($directory -eq 'nodejs' -and $file.Name -like 'node-v*.zip')
                if ($isInstallerArchive) {
                    Add-CleanupTarget -Plan $Plan -Path $file.FullName -ScopeRoot $repo -Kind File -Category 'Interrupted runtime installer archive' -KnownSafeAncestry
                }
            }
        }
        catch {
            [void]$Plan.Skipped.Add("$runtimePath (installer archive scan failed: $($_.Exception.Message))")
        }
    }
    if (Test-Path -LiteralPath $runtimes -PathType Container) {
        try {
            foreach ($directory in @(Get-ChildItem -LiteralPath $runtimes -Directory -Force -ErrorAction Stop | Where-Object { $_.Name -match '^\.python-.+-(staging|backup)$' })) {
                Add-CleanupTarget -Plan $Plan -Path $directory.FullName -ScopeRoot $repo -Kind Directory -Category 'Interrupted Python runtime installation' -KnownSafeAncestry
            }
        }
        catch {
            [void]$Plan.Skipped.Add("$runtimes (staging directory scan failed: $($_.Exception.Message))")
        }
    }
}

function Add-LogsTargets {
    param(
        [Parameter(Mandatory)][System.Collections.IDictionary]$Plan,
        [Parameter(Mandatory)][string]$DataRoot
    )
    $logRoot = Join-Path (Get-FullPathSafe $DataRoot) 'logs'
    if (-not (Test-Path -LiteralPath $logRoot -PathType Container)) {
        Add-CleanupTarget -Plan $Plan -Path $logRoot -ScopeRoot $DataRoot -Kind Directory -Category 'Application logs'
        return
    }
    $reparseComponent = Get-ReparseComponent $logRoot
    if ($reparseComponent) {
        [void]$Plan.Skipped.Add("$logRoot (path crosses a reparse point at '$reparseComponent')")
        return
    }

    $queue = [Collections.Generic.Queue[string]]::new()
    $queue.Enqueue($logRoot)
    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        try {
            foreach ($item in @(Get-ChildItem -LiteralPath $current -Force -ErrorAction Stop)) {
                if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                    [void]$Plan.Skipped.Add("$($item.FullName) (reparse point not followed)")
                    continue
                }
                if ($item.PSIsContainer) { $queue.Enqueue($item.FullName); continue }
                if ($item.Name -like '*.log') {
                    Add-CleanupTarget -Plan $Plan -Path $item.FullName -ScopeRoot $DataRoot -Kind File -Category 'Application log'
                }
            }
        }
        catch {
            [void]$Plan.Skipped.Add("$current (log scan failed: $($_.Exception.Message))")
        }
    }
}

function Get-AdsmodCleanupPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][AllowEmptyString()][string]$RepoRoot,
        [Parameter(Mandatory)][AllowEmptyString()][string]$DataRoot,
        [string]$DatabasePath,
        [ValidateSet('sqlite', 'postgres', 'unknown')][string]$DatabaseBackend = 'unknown',
        [Parameter(Mandatory)][ValidateSet('Cache', 'Uninstall', 'Logs', 'Checkpoints', 'AllData')][string]$Operation
    )

    $plan = New-CleanupPlan -Operation $Operation
    if ([string]::IsNullOrWhiteSpace($RepoRoot) -or [string]::IsNullOrWhiteSpace($DataRoot)) {
        [void]$plan.Errors.Add('Repository and storage roots must both be explicit, non-empty paths.')
        return [pscustomobject]$plan
    }
    try {
        $repo = Get-FullPathSafe $RepoRoot
        $data = Get-FullPathSafe $DataRoot
    }
    catch {
        [void]$plan.Errors.Add("Cleanup roots could not be resolved: $($_.Exception.Message)")
        return [pscustomobject]$plan
    }
    $plan.DataRoot = $data
    if (Test-IsFilesystemRoot $repo) {
        [void]$plan.Errors.Add("Repository root is unsafe: $repo")
        return [pscustomobject]$plan
    }
    if (Test-IsFilesystemRoot $data) {
        [void]$plan.Errors.Add("Storage root is unsafe: $data")
        return [pscustomobject]$plan
    }
    if (-not (Test-Path -LiteralPath $repo -PathType Container)) {
        [void]$plan.Errors.Add("Repository root is missing or is not a directory: $repo")
        return [pscustomobject]$plan
    }
    if ((Test-Path -LiteralPath $data) -and -not (Test-Path -LiteralPath $data -PathType Container)) {
        [void]$plan.Errors.Add("Storage root exists but is not a directory: $data")
        return [pscustomobject]$plan
    }
    foreach ($rootPath in @($repo, $data)) {
        $reparseComponent = Get-ReparseComponent $rootPath
        if ($reparseComponent) {
            [void]$plan.Errors.Add("Cleanup root crosses a reparse point: $rootPath ('$reparseComponent')")
            return [pscustomobject]$plan
        }
    }

    if ($Operation -in @('Cache', 'Uninstall')) {
        $runtimeCache = Join-Path $repo 'runtimes\cache'
        if ($Operation -eq 'Cache') {
            Add-CleanupTarget -Plan $plan -Path $runtimeCache -ScopeRoot $repo -Kind Contents -Category 'Launcher runtime and tool caches' -PreserveNames @('.gitkeep', 'CACHEDIR.TAG')
        } else {
            Add-CleanupTarget -Plan $plan -Path $runtimeCache -ScopeRoot $repo -Kind Directory -Category 'Launcher runtime and tool caches'
        }

        $client = Join-Path $repo 'app\client'
        foreach ($relative in @('.angular', 'node_modules\.cache', 'node_modules\.vite')) {
            Add-CleanupTarget -Plan $plan -Path (Join-Path $client $relative) -ScopeRoot $repo -Kind Directory -Category 'Frontend cache'
        }
        Add-CleanupTarget -Plan $plan -Path (Join-Path $client 'out-tsc') -ScopeRoot $repo -Kind Directory -Category 'Generated frontend compiler output'
        Add-CleanupTarget -Plan $plan -Path (Join-Path $client 'coverage') -ScopeRoot $repo -Kind Directory -Category 'Generated frontend coverage output'
        Add-CleanupTarget -Plan $plan -Path (Join-Path $client 'dist') -ScopeRoot $repo -Kind Directory -Category 'Generated frontend build'

        Add-CacheDirectoryCandidates -Plan $plan -RepoRoot $repo -DataRoot $data
        Add-DatabaseTemporaryTargets -Plan $plan -DataRoot $data -IncludeTrainingTemporary
        Add-SettingsTemporaryTargets -Plan $plan -RepoRoot $repo
        Add-InstallerTemporaryTargets -Plan $plan -RepoRoot $repo
    }

    if ($Operation -eq 'Uninstall') {
        Add-CleanupTarget -Plan $plan -Path (Join-Path $repo 'runtimes') -ScopeRoot $repo -Kind Contents -Category 'Launcher runtime files' -PreserveNames @('.gitkeep')
        foreach ($relative in @('app\server\.venv', '.venv', 'app\client\node_modules')) {
            Add-CleanupTarget -Plan $plan -Path (Join-Path $repo $relative) -ScopeRoot $repo -Kind Directory -Category 'Application runtime or dependency environment'
        }
        Add-LogsTargets -Plan $plan -DataRoot $data
    }
    elseif ($Operation -eq 'Logs') {
        Add-LogsTargets -Plan $plan -DataRoot $data
    }
    elseif ($Operation -eq 'Checkpoints') {
        Add-CleanupTarget -Plan $plan -Path (Join-Path $data 'checkpoints') -ScopeRoot $data -Kind Directory -Category 'Persistent training checkpoints'
    }
    elseif ($Operation -eq 'AllData') {
        Add-CleanupTarget -Plan $plan -Path (Join-Path $data 'database.json') -ScopeRoot $data -Kind File -Category 'Database connection configuration'
        Add-CleanupTarget -Plan $plan -Path (Join-Path $data 'checkpoints') -ScopeRoot $data -Kind Directory -Category 'Persistent training checkpoints'
        Add-CleanupTarget -Plan $plan -Path (Join-Path $data 'training') -ScopeRoot $data -Kind Directory -Category 'Persistent training artifacts'
        Add-LogsTargets -Plan $plan -DataRoot $data
        Add-DatabaseTemporaryTargets -Plan $plan -DataRoot $data
        if ($DatabaseBackend -eq 'sqlite' -and -not [string]::IsNullOrWhiteSpace($DatabasePath)) {
            try { $sqlitePath = Get-FullPathSafe $DatabasePath }
            catch {
                [void]$plan.Errors.Add("Configured SQLite path could not be resolved: $($_.Exception.Message)")
                $sqlitePath = $null
            }
            if ($sqlitePath -and (Test-IsFilesystemRoot $sqlitePath)) {
                [void]$plan.Skipped.Add("$sqlitePath (configured SQLite path is a filesystem root; database and sidecars are preserved)")
            }
            elseif ($sqlitePath) {
                foreach ($path in @($sqlitePath, "$sqlitePath-wal", "$sqlitePath-shm", "$sqlitePath-journal")) {
                    Add-CleanupTarget -Plan $plan -Path $path -ScopeRoot $data -Kind File -Category 'Configured SQLite database' -AllowExternalExact
                }
            }
        }
        elseif ($DatabaseBackend -eq 'postgres') {
            [void]$plan.Skipped.Add('Configured PostgreSQL database (external database is never modified)')
        }
        else {
            [void]$plan.Skipped.Add('SQLite database (backend or database path could not be resolved)')
        }
    }

    $targets = @($plan.Targets | Sort-Object @{ Expression = { $_.Path.Length }; Descending = $true }, @{ Expression = { $_.Path.ToUpperInvariant() }; Descending = $false })
    $plan.Targets = [Collections.Generic.List[object]]::new()
    foreach ($target in $targets) { [void]$plan.Targets.Add($target) }
    return [pscustomobject]$plan
}

function Get-RemovalItems {
    param(
        [Parameter(Mandatory)][pscustomobject]$Target,
        [Parameter(Mandatory)][AllowEmptyCollection()][System.Collections.Generic.List[string]]$Skipped
    )

    $path = Get-FullPathSafe $Target.Path
    if (-not (Test-Path -LiteralPath $path)) { return @() }
    if (-not $Target.KnownSafeAncestry) {
        $reparseComponent = Get-ReparseComponent $path
        if ($reparseComponent) {
            [void]$Skipped.Add("$path (path crosses a reparse point at '$reparseComponent')")
            return @()
        }
    }
    $rootItem = Get-Item -LiteralPath $path -Force -ErrorAction Stop
    if ($rootItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
        [void]$Skipped.Add("$path (target is a reparse point; not followed or removed)")
        return @()
    }
    if (($Target.Kind -eq 'File' -and $rootItem.PSIsContainer) -or
        ($Target.Kind -in @('Directory', 'Contents') -and -not $rootItem.PSIsContainer)) {
        [void]$Skipped.Add("$path (target changed type after inventory; not removed)")
        return @()
    }
    if (-not $rootItem.PSIsContainer) { return @([pscustomobject]@{ Path = $path; IsDirectory = $false }) }

    $items = [Collections.Generic.List[object]]::new()
    $protectedDirectories = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $queue = [Collections.Generic.Queue[string]]::new()
    $queue.Enqueue($path)
    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        if (-not (Test-PathWithin -Path $current -Root $Target.ScopeRoot -AllowRoot:($Target.Kind -eq 'Contents'))) {
            [void]$Skipped.Add("$current (directory scan escaped the validated scope)")
            continue
        }
        try { $currentAttributes = [System.IO.File]::GetAttributes($current) }
        catch {
            [void]$Skipped.Add("$current (directory attributes unavailable: $($_.Exception.Message))")
            continue
        }
        if ($currentAttributes -band [System.IO.FileAttributes]::ReparsePoint) {
            [void]$Skipped.Add("$current (directory became a reparse point; not followed)")
            continue
        }
        try {
            foreach ($child in @(Get-ChildItem -LiteralPath $current -Force -ErrorAction Stop)) {
                $childPath = Get-FullPathSafe $child.FullName
                if (-not (Test-PathWithin -Path $childPath -Root $Target.ScopeRoot)) {
                    [void]$Skipped.Add("$childPath (outside validated scope)")
                    continue
                }
                if ($child.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                    [void]$Skipped.Add("$childPath (reparse point; not followed or removed)")
                    continue
                }
                $isRootSentinel = $Target.Kind -eq 'Contents' -and
                    [System.IO.Path]::GetDirectoryName($childPath).Equals($path, [StringComparison]::OrdinalIgnoreCase) -and
                    $child.Name -in $Target.PreserveNames
                if ($isRootSentinel) {
                    [void]$Skipped.Add("$childPath (preserved sentinel)")
                    $ancestor = [System.IO.Path]::GetDirectoryName($childPath)
                    $rootPrefix = $path.TrimEnd('\') + '\'
                    while ($ancestor -and $ancestor.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
                        [void]$protectedDirectories.Add($ancestor)
                        $ancestor = [System.IO.Path]::GetDirectoryName($ancestor)
                    }
                    if ($Target.Kind -ne 'Contents') { [void]$protectedDirectories.Add($path) }
                    continue
                }
                [void]$items.Add([pscustomobject]@{ Path = $childPath; IsDirectory = [bool]$child.PSIsContainer })
                if ($child.PSIsContainer) { $queue.Enqueue($childPath) }
            }
        }
        catch {
            [void]$Skipped.Add("$current (enumeration failed: $($_.Exception.Message))")
        }
    }

    $ordered = @($items |
        Where-Object { -not $protectedDirectories.Contains($_.Path) } |
        Sort-Object @{ Expression = { $_.Path.Length }; Descending = $true }, @{ Expression = { $_.Path.ToUpperInvariant() }; Descending = $false })
    if ($Target.Kind -ne 'Contents') {
        $ordered += [pscustomobject]@{ Path = $path; IsDirectory = $true }
    }
    return $ordered
}

function Invoke-AdsmodCleanupPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][pscustomobject]$Plan,
        [switch]$DryRun,
        [switch]$DetailedPreview
    )

    $removed = [Collections.Generic.List[string]]::new()
    $locations = [Collections.Generic.List[object]]::new()
    $skipped = [Collections.Generic.List[string]]::new()
    $failed = [Collections.Generic.List[string]]::new()
    foreach ($message in $Plan.Skipped) { [void]$skipped.Add($message) }
    foreach ($message in $Plan.Errors) { [void]$failed.Add($message) }

    foreach ($target in $Plan.Targets) {
        if (-not $target.Exists) { continue }
        if ($DryRun -and -not $DetailedPreview) {
            [void]$removed.Add($target.Path)
            [void]$locations.Add([pscustomobject]@{ Path = $target.Path; Category = $target.Category; Count = $null })
            continue
        }
        try {
            $items = @(Get-RemovalItems -Target $target -Skipped $skipped)
        }
        catch {
            [void]$failed.Add("$($target.Path) ($($_.Exception.Message))")
            continue
        }
        $targetRemoved = 0
        foreach ($item in $items) {
            if ($DryRun) {
                [void]$removed.Add($item.Path)
                $targetRemoved++
                continue
            }
            try {
                if (-not $Target.AllowExternalExact -and -not (Test-PathWithin -Path $item.Path -Root $Target.ScopeRoot)) {
                    [void]$skipped.Add("$($item.Path) (outside validated scope)")
                    continue
                }
                $attributes = [System.IO.File]::GetAttributes($item.Path)
                if ($attributes -band [System.IO.FileAttributes]::ReparsePoint) {
                    [void]$skipped.Add("$($item.Path) (target became a reparse point; not followed or removed)")
                    continue
                }
                Remove-Item -LiteralPath $item.Path -Force -ErrorAction Stop
                [void]$removed.Add($item.Path)
                $targetRemoved++
            }
            catch {
                [void]$failed.Add("$($item.Path) ($($_.Exception.Message))")
            }
        }
        if ($targetRemoved -gt 0) {
            [void]$locations.Add([pscustomobject]@{ Path = $target.Path; Category = $target.Category; Count = $targetRemoved })
        }
    }

    return [pscustomobject]@{
        Operation = $Plan.Operation
        DryRun = [bool]$DryRun
        Removed = @($removed)
        Locations = @($locations)
        Skipped = @($skipped)
        Failed = @($failed)
    }
}

function Invoke-AdsmodScopedCleanup {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$ScopeRoot,
        [Parameter(Mandatory)][ValidateSet('File', 'Directory', 'Contents')][string]$Kind,
        [string[]]$PreserveNames = @(),
        [string]$Category = 'Launcher maintenance path'
    )
    $plan = New-CleanupPlan -Operation Cache
    Add-CleanupTarget -Plan $plan -Path $Path -ScopeRoot $ScopeRoot -Kind $Kind -Category $Category -PreserveNames $PreserveNames
    return Invoke-AdsmodCleanupPlan -Plan ([pscustomobject]$plan)
}

Export-ModuleMember -Function Get-AdsmodCleanupPlan, Invoke-AdsmodCleanupPlan, Invoke-AdsmodScopedCleanup
