[CmdletBinding()]
param(
  [string]$ClientRelativePath = "..\..\..\app\\client",
  [string]$OutputRelativePath = "..\..\windows"
)

$ErrorActionPreference = "Stop"

$clientDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $ClientRelativePath))
$releaseDir = Join-Path $clientDir "src-tauri\target\release"
$bundleDir = Join-Path $releaseDir "bundle"
$outputDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $OutputRelativePath))
$installersDir = Join-Path $outputDir "installers"
$portableDir = Join-Path $outputDir "portable"
$tauriConfigPath = Join-Path $clientDir "src-tauri\tauri.conf.json"
$tauriConfig = Get-Content -Path $tauriConfigPath -Raw | ConvertFrom-Json
$releaseVersion = [string]$tauriConfig.version

if (-not (Test-Path $bundleDir)) {
  throw "Bundle directory not found. Run 'release\\tauri\\build_with_tauri.bat' first. Missing: $bundleDir"
}

if (Test-Path $outputDir) {
  Remove-Item -Recurse -Force $outputDir
}

New-Item -ItemType Directory -Path $installersDir -Force | Out-Null
New-Item -ItemType Directory -Path $portableDir -Force | Out-Null

$installerArtifacts = @()

$nsisDir = Join-Path $bundleDir "nsis"
if (Test-Path $nsisDir) {
  $nsisFiles = Get-ChildItem -Path $nsisDir -Filter "*.exe" -File |
    Where-Object { $_.Name -match [regex]::Escape("_$releaseVersion`_") -or $_.Name -match [regex]::Escape("-$releaseVersion-") }
  foreach ($file in $nsisFiles) {
    Copy-Item -Path $file.FullName -Destination $installersDir -Force
    $installerArtifacts += Join-Path $installersDir $file.Name
  }
}

$msiDir = Join-Path $bundleDir "msi"
if (Test-Path $msiDir) {
  $msiFiles = Get-ChildItem -Path $msiDir -Filter "*.msi" -File |
    Where-Object { $_.Name -match [regex]::Escape("_$releaseVersion`_") -or $_.Name -match [regex]::Escape("-$releaseVersion-") }
  foreach ($file in $msiFiles) {
    Copy-Item -Path $file.FullName -Destination $installersDir -Force
    $installerArtifacts += Join-Path $installersDir $file.Name
  }
}

$portableExeCandidates = Get-ChildItem -Path $releaseDir -Filter "*.exe" -File |
  Where-Object { $_.Name -notmatch "(?i)(setup|installer|uninstall|updater)" }

if ($portableExeCandidates.Count -eq 0) {
  throw "Portable app executable not found under release directory: $releaseDir"
}

foreach ($file in $portableExeCandidates) {
  Copy-Item -Path $file.FullName -Destination $portableDir -Force
}

$legacyLayoutEntries = @(
  "app",
  "settings",
  "runtimes",
  "pyproject.toml",
  "uv.lock"
)

$currentLayoutEntries = @(
  "r\server",
  "r\settings",
  "r\runtimes",
  "r\pyproject.toml",
  "r\uv.lock"
)

$hasLegacyLayout = $true
foreach ($entry in $legacyLayoutEntries) {
  if (-not (Test-Path (Join-Path $releaseDir $entry))) {
    $hasLegacyLayout = $false
    break
  }
}

$hasCurrentLayout = $true
foreach ($entry in $currentLayoutEntries) {
  if (-not (Test-Path (Join-Path $releaseDir $entry))) {
    $hasCurrentLayout = $false
    break
  }
}

if (-not $hasLegacyLayout -and -not $hasCurrentLayout) {
  throw "Required portable payload layout not found under: $releaseDir"
}

if ($hasCurrentLayout) {
  Copy-Item -Path (Join-Path $releaseDir "r\server") -Destination (Join-Path $portableDir "app") -Recurse -Force
  Copy-Item -Path (Join-Path $releaseDir "r\settings") -Destination (Join-Path $portableDir "settings") -Recurse -Force
  Copy-Item -Path (Join-Path $releaseDir "r\runtimes") -Destination (Join-Path $portableDir "runtimes") -Recurse -Force
  Copy-Item -Path (Join-Path $releaseDir "r\pyproject.toml") -Destination (Join-Path $portableDir "pyproject.toml") -Force
  Copy-Item -Path (Join-Path $releaseDir "r\uv.lock") -Destination (Join-Path $portableDir "uv.lock") -Force
  if (Test-Path (Join-Path $releaseDir "r\resources")) {
    Copy-Item -Path (Join-Path $releaseDir "r\resources") -Destination (Join-Path $portableDir "app\resources") -Recurse -Force
  }
}

if ($hasLegacyLayout) {
  $portableResourceEntries = @(
    "app",
    "settings",
    "runtimes",
    "pyproject.toml",
    "uv.lock",
    "_up_",
    "resources"
  )
  foreach ($entry in $portableResourceEntries) {
    $sourcePath = Join-Path $releaseDir $entry
    if (Test-Path $sourcePath) {
      $destinationPath = Join-Path $portableDir $entry
      Copy-Item -Path $sourcePath -Destination $destinationPath -Recurse -Force
    }
  }
}

$requiredRuntimePayload = @(
  "runtimes\\uv\\uv.exe",
  "runtimes\\python\\python.exe"
)

foreach ($relativePath in $requiredRuntimePayload) {
  $payloadPath = Join-Path $portableDir $relativePath
  if (-not (Test-Path $payloadPath)) {
    throw "Required portable runtime file is missing after export: $payloadPath"
  }
}

$instructions = @"
ADSMOD desktop build output

1) Preferred for users:
   Open installers\ and run the setup executable (.exe) or .msi.

2) Portable executable:
   portable\ contains the app .exe and the required runtime resource payload.
   Keep the exported contents together in the same directory.

Generated from:
$bundleDir
"@
Set-Content -Path (Join-Path $outputDir "README.txt") -Value $instructions -Encoding ascii

Write-Host "[OK] Exported Windows artifacts to: $outputDir"
Write-Host "[INFO] Installers:"
if ($installerArtifacts.Count -eq 0) {
  Write-Host " - none found"
} else {
  $installerArtifacts | ForEach-Object { Write-Host " - $_" }
}
Write-Host "[INFO] Portable executables:"
$portableFiles = Get-ChildItem -Path $portableDir -Filter "*.exe" -File
if ($portableFiles.Count -eq 0) {
  Write-Host " - none found"
} else {
  $portableFiles | ForEach-Object { Write-Host " - $($_.FullName)" }
}
