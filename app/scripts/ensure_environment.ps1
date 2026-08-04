function Ensure-EnvironmentFile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$EnvFile,
        [Parameter(Mandatory)][string]$EnvExample
    )

    if (Test-Path -LiteralPath $EnvFile -PathType Container) {
        throw "Environment path exists but is not a file: $EnvFile"
    }
    if (Test-Path -LiteralPath $EnvFile -PathType Leaf) {
        return $false
    }
    if (-not (Test-Path -LiteralPath $EnvExample -PathType Leaf)) {
        throw "Missing environment template: $EnvExample"
    }

    Copy-Item -LiteralPath $EnvExample -Destination $EnvFile -ErrorAction Stop
    return $true
}
