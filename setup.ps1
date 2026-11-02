$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    New-Item -ItemType Directory -Path vendor,artifacts -Force | Out-Null
    if (-not (Test-Path -LiteralPath vendor/psd2live/.git)) {
        & git clone https://github.com/tsunehimatoi/psd2live vendor/psd2live
        if ($LASTEXITCODE -ne 0) { throw 'PSD2Live clone failed' }
        & git -C vendor/psd2live checkout bafabab1a360e208671d39adf5087c62f6cae1f2
    }
    if (-not (Test-Path -LiteralPath vendor/ayagami/.git)) {
        & git clone https://github.com/AyagamiDev/ayagami vendor/ayagami
        if ($LASTEXITCODE -ne 0) { throw 'Ayagami clone failed' }
        & git -C vendor/ayagami checkout 0d1d7aa3efe57b1b363b72e672e353c52a9c15d6
    }
    # A narrow exporter interoperability correction; Ayagami itself remains unmodified.
    & python -m popy.setup_exporter
    if ($LASTEXITCODE -ne 0) { throw 'Exporter setup failed' }
    & python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency setup failed' }
} finally { Pop-Location }
