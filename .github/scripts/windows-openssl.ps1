# Install OpenSSL via Chocolatey and set OPENSSL_DIR / OPENSSL_LIB_DIR for GitHub Actions.
$ErrorActionPreference = "Stop"

choco install openssl -y

$paths = @("C:\Program Files\OpenSSL-Win64", "C:\Program Files\OpenSSL")
$found = $null
foreach ($p in $paths) {
    if (Test-Path "$p\include") { $found = $p; break }
}
if (-not $found) {
    $dirs = Get-ChildItem "C:\Program Files" -Filter "OpenSSL*" -Directory -ErrorAction SilentlyContinue
    foreach ($d in $dirs) {
        if (Test-Path "$($d.FullName)\include") { $found = $d.FullName; break }
    }
}
if (-not $found) { throw "OpenSSL include dir not found. Checked: $($paths -join ', ')" }
"OPENSSL_DIR=$found" >> $env:GITHUB_ENV

$libSubdirs = @("lib\VC\x64\MD", "lib\VC\x64\MDd", "lib\VC\x64\MT", "lib\VC\x64\MTd", "lib")
$libDir = $null
foreach ($sub in $libSubdirs) {
    $d = Join-Path $found $sub
    if (Test-Path "$d\libcrypto.lib") { $libDir = $d; break }
}
if (-not $libDir) { throw "OpenSSL libcrypto.lib not found under $found" }
"OPENSSL_LIB_DIR=$libDir" >> $env:GITHUB_ENV
