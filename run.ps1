param([string]$Stage = 'all', [switch]$SkipBuild, [switch]$NoCmo)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $pipelineArgs = @('-m', 'popy.cli', $Stage)
    if ($SkipBuild) { $pipelineArgs += '--skip-build' }
    if ($NoCmo) { $pipelineArgs += '--no-cmo' }
    & python @pipelineArgs
    if ($LASTEXITCODE -ne 0) { throw "Conversion failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }
