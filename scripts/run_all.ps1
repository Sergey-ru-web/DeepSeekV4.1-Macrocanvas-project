$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install --quiet -e .
& ".venv\Scripts\python.exe" -m macrocanvas --config "configs\baseline.toml"
