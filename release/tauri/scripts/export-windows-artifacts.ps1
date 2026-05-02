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
  $nsisFiles = Get-ChildItem -Path $nsisDir -Filter "*.exe" -File
  foreach ($file in $nsisFiles) {
    Copy-Item -Path $file.FullName -Destination $installersDir -Force
    $installerArtifacts += Join-Path $installersDir $file.Name
  }
}

$msiDir = Join-Path $bundleDir "msi"
if (Test-Path $msiDir) {
  $msiFiles = Get-ChildItem -Path $msiDir -Filter "*.msi" -File
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

$requiredPortableEntries = @(
  "app",
  "settings",
  "runtimes",
  "pyproject.toml",
  "uv.lock"
)

foreach ($entry in $requiredPortableEntries) {
  $requiredPath = Join-Path $releaseDir $entry
  if (-not (Test-Path $requiredPath)) {
    throw "Required portable payload entry not found: $requiredPath"
  }
}

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
