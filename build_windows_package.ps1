$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment not found at $pythonExe"
}

Push-Location $projectRoot
try {
    $env:PYTHONNOUSERSITE = "1"
    $env:APPDATA = Join-Path $projectRoot ".build-appdata"
    $userSite = Join-Path $env:APPDATA "Python\Python312\site-packages"
    New-Item -ItemType Directory -Force -Path $userSite | Out-Null

    & $pythonExe -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }

    Remove-Item -Recurse -Force build, dist, release -ErrorAction SilentlyContinue

    & $pythonExe -m PyInstaller `
        --noconfirm `
        --clean `
        --onedir `
        --name TVLKForecastingTool `
        --collect-all streamlit `
        --collect-all altair `
        --collect-all pyarrow `
        --hidden-import statsmodels.tsa.holtwinters `
        --hidden-import pandas._libs.tslibs.timedeltas `
        --hidden-import pandas._libs.tslibs.np_datetime `
        --add-data "app.py;." `
        --add-data "data;data" `
        launcher.py
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller build failed."
    }

    $releaseRoot = Join-Path $projectRoot "release\TVLK Financial Forecasting Tool"
    $zipPath = Join-Path $projectRoot "release\TVLK Financial Forecasting Tool - Windows.zip"

    New-Item -ItemType Directory -Force -Path $releaseRoot | Out-Null
    Copy-Item -Recurse -Force (Join-Path $projectRoot "dist\TVLKForecastingTool\*") $releaseRoot
    Copy-Item -Force (Join-Path $projectRoot "Launch TVLK Forecasting Tool.bat") $releaseRoot
    Copy-Item -Force (Join-Path $projectRoot "README-Windows.txt") $releaseRoot

    if (Test-Path $zipPath) {
        Remove-Item -Force $zipPath
    }

    Compress-Archive -Path "$releaseRoot\*" -DestinationPath $zipPath -Force

    Write-Host ""
    Write-Host "Packaged app folder: $releaseRoot"
    Write-Host "Packaged zip:        $zipPath"
}
finally {
    Pop-Location
}
