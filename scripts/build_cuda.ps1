$ErrorActionPreference = 'Stop'
$vswhere = "${env:ProgramFiles(x86)}/Microsoft Visual Studio/Installer/vswhere.exe"
$vsPath = & $vswhere -latest -prerelease -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw 'Visual Studio C++ x64 build tools were not found' }
$vcvars = Join-Path $vsPath 'VC/Auxiliary/Build/vcvars64.bat'
# Import the compiler environment; no filesystem operations are performed by cmd.
$compilerEnv = & cmd.exe /c "call `"$vcvars`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'Loading the MSVC environment failed' }
$compilerEnv | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
# Some hosts inject both Path and PATH; vcvars updates uppercase PATH only.
$compilerPath = $compilerEnv | Where-Object { $_ -cmatch '^PATH=' } | Select-Object -First 1
if ($compilerPath) { $env:PATH = $compilerPath.Substring(5) }
$env:DISTUTILS_USE_SDK = '1'
$env:FDTDMESH_BUILD_CUDA = '1'
Push-Location "$PSScriptRoot/.."
try {
    & "$PSScriptRoot/../.venv/Scripts/python.exe" "$PSScriptRoot/../setup.py" build_ext --inplace
    if ($LASTEXITCODE -ne 0) { throw 'CUDA build failed' }
} finally {
    Pop-Location
    Remove-Item Env:FDTDMESH_BUILD_CUDA -ErrorAction SilentlyContinue
}
