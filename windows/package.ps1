param([string]$Sdk = $env:HIP_PATH, [string]$Llama = "", [string]$CMake = "cmake", [string]$Ninja = "ninja", [string]$Bun = "bun", [string]$Out = "", [string]$Version = "")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Out) { $Out = Join-Path $root "dist" }
$pkg = Join-Path $Out "Chrysalis"
& (Join-Path $PSScriptRoot "build.ps1") -Sdk $Sdk -Llama $Llama -CMake $CMake -Ninja $Ninja
if ($LASTEXITCODE) { throw "the engines did not build" }
Remove-Item $pkg -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$pkg\build", "$pkg\data\models\qwen3.8-27b", "$pkg\data\models\qwen3.8-flash-next", "$pkg\web" | Out-Null
Copy-Item "$root\build\*.dll", "$root\build\gpu.exe", "$root\build\ramdrive.exe" "$pkg\build\"
Invoke-WebRequest -UseBasicParsing "https://github.com/winfsp/winfsp/releases/download/v2.1/winfsp-2.1.25156.msi" -OutFile "$pkg\build\winfsp.msi"
if ((Get-FileHash "$pkg\build\winfsp.msi" -Algorithm SHA256).Hash.ToLower() -ne "073a70e00f77423e34bed98b86e600def93393ba5822204fac57a29324db9f7a") { throw "WinFsp's download does not match its checksum" }
Copy-Item "$root\build\hipblaslt" "$pkg\build\" -Recurse
Copy-Item "$root\chrysalis" "$pkg\" -Recurse
Get-ChildItem "$pkg\chrysalis" -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
Copy-Item "$root\LICENSE" "$pkg\"
Copy-Item "$root\data\models\tokenizer.json", "$root\data\models\tokenizer_config.json", "$root\data\models\chat_template.jinja" "$pkg\data\models\"
Copy-Item "$root\data\models\qwen3.8-27b\config.json", "$root\data\models\qwen3.8-27b\generation_config.json", "$root\data\models\qwen3.8-27b\preprocessor_config.json" "$pkg\data\models\qwen3.8-27b\"
Copy-Item "$root\data\models\qwen3.8-flash-next\preprocessor_config.json" "$pkg\data\models\qwen3.8-flash-next\"

$t = Join-Path $Out "tmp"
Remove-Item $t -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $t | Out-Null
Invoke-WebRequest -UseBasicParsing "https://github.com/astral-sh/uv/releases/download/0.13.5/uv-x86_64-pc-windows-msvc.zip" -OutFile "$t\uv.zip"
if ((Get-FileHash "$t\uv.zip" -Algorithm SHA256).Hash.ToLower() -ne "a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a") { throw "uv's download does not match its checksum" }
Expand-Archive "$t\uv.zip" -DestinationPath $t
& "$t\uv.exe" python install -q --no-bin 3.12 --install-dir "$t\python"
Move-Item (Get-ChildItem "$t\python" -Directory -Filter "cpython-3.12.*" | Select-Object -First 1).FullName "$pkg\data\venv"
$pkgs = (Select-String -Path "$root\chrysalis.sh" -Pattern '^PKGS=\((.*)\)$').Matches[0].Groups[1].Value -split " " | ForEach-Object { $_.Trim('"') }
& "$t\uv.exe" pip install -q --python "$pkg\data\venv\python.exe" --break-system-packages @pkgs
if ($LASTEXITCODE) { throw "the Python packages did not install" }

Get-ChildItem "$root\web" -File | Where-Object { $_.Extension -in ".ts", ".tsx", ".html", ".css", ".js", ".svg", ".json", ".lock", ".md" -and $_.Name -notmatch "\.test\.ts$|\.check\.ts$|^fake-|^bench-|-harness\.ts$|^tests\.ts$" } | Copy-Item -Destination "$pkg\web\"
Push-Location "$pkg\web"
& $Bun install --frozen-lockfile --production 2>&1 | Out-Null
if ($LASTEXITCODE) { Pop-Location; throw "the web chat's packages did not install" }
Pop-Location
& $Bun build --compile "$root\web\launch.ts" --outfile "$pkg\Chrysalis.exe" 2>&1 | Out-Null
if (-not (Test-Path "$pkg\Chrysalis.exe")) { throw "Chrysalis.exe did not build" }
if (-not $Version -and (Get-Command git -ErrorAction SilentlyContinue)) { $Version = git -C $root describe --tags --always 2>$null }
if ($Version) { $Version | Out-File -Encoding ascii "$pkg\VERSION" }
Remove-Item $t -Recurse -Force
$zip = Join-Path $Out "Chrysalis-windows-x64.zip"
Remove-Item $zip -ErrorAction SilentlyContinue
Compress-Archive -Path $pkg -DestinationPath $zip -CompressionLevel Optimal
"{0}: {1:N0} MB unpacked, {2:N0} MB zipped" -f $pkg, ((Get-ChildItem $pkg -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), ((Get-Item $zip).Length / 1MB)
