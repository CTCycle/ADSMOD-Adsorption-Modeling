param(
    [Parameter(Mandatory)][string]$PythonPath,
    [Parameter(Mandatory)][string]$RepositoryRoot
)

$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = Join-Path $RepositoryRoot 'app'
Push-Location $RepositoryRoot
try {
    $output = @(& $PythonPath -m server.runtime_config --json)
    if ($LASTEXITCODE -ne 0) { throw 'ADSMOD runtime settings could not be loaded.' }
    $marker = @($output | Where-Object { [string]$_ -like 'ADSMOD_RUNTIME:*' } | Select-Object -Last 1)
    if ($marker.Count -ne 1) { throw 'ADSMOD runtime settings returned no runtime record.' }
    $runtime = $marker[0].Substring('ADSMOD_RUNTIME:'.Length) | ConvertFrom-Json
    Write-Output ('{0}|{1}|{2}' -f $runtime.host, $runtime.backend_port, $runtime.frontend_port)
} finally {
    Pop-Location
}
