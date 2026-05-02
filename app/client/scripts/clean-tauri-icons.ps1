[CmdletBinding()]
param(
  [string]$IconsPath = "..\src-tauri\icons"
)

$ErrorActionPreference = "Stop"

$iconsDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $IconsPath))

foreach ($mobileDir in @("android", "ios")) {
  $path = Join-Path $iconsDir $mobileDir
  if (Test-Path $path) {
    Remove-Item -Recurse -Force $path
    Write-Host "[OK] Removed generated mobile icon folder: $path"
  }
}

Write-Host "[DONE] Desktop icon set ready at: $iconsDir"
