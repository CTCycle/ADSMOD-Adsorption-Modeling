[CmdletBinding()]
param(
  [string]$ClientRelativePath = "..\..\..\ADSMOD\client",
  [string]$OutputRelativePath = "..\..\windows"
)

$ErrorActionPreference = "Stop"

$clientDir = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $ClientRelativePath))
$pathsToRemove = @(
  (Join-Path $clientDir "src-tauri\target\release"),
  [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot $OutputRelativePath))
)

foreach ($path in $pathsToRemove) {
  if (Test-Path $path) {
    Remove-Item -Recurse -Force $path
    Write-Host "[OK] Removed: $path"
  } else {
    Write-Host "[INFO] Not found: $path"
  }
}

Write-Host "[DONE] Build cleanup complete."
