param([string]$Sdk = $env:HIP_PATH, [string]$Llama = "", [string]$CMake = "cmake", [string]$Ninja = "ninja", [string]$Bun = "bun", [string]$Out = "", [string]$Version = "", [ValidateSet("hip", "cuda")][string]$Backend = "hip", [string]$Cuda = $env:CUDA_PATH, [string]$Arch = "sm_86")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Out) { $Out = Join-Path $root "dist" }
$pkg = Join-Path $Out "Chrysalis"
& (Join-Path $PSScriptRoot "build.ps1") -Sdk $Sdk -Llama $Llama -CMake $CMake -Ninja $Ninja -Backend $Backend -Cuda $Cuda -Arch $Arch
if ($LASTEXITCODE) { throw "the engines did not build" }
Remove-Item $pkg -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force "$pkg\build", "$pkg\data\models\qwen3.8-27b", "$pkg\data\models\qwen3.8-flash-next", "$pkg\web" | Out-Null
Copy-Item "$root\build\*.dll", "$root\build\ramdrive.exe" "$pkg\build\"
Invoke-WebRequest -UseBasicParsing "https://github.com/winfsp/winfsp/releases/download/v2.1/winfsp-2.1.25156.msi" -OutFile "$pkg\build\winfsp.msi"
if ((Get-FileHash "$pkg\build\winfsp.msi" -Algorithm SHA256).Hash.ToLower() -ne "073a70e00f77423e34bed98b86e600def93393ba5822204fac57a29324db9f7a") { throw "WinFsp's download does not match its checksum" }
if ($Backend -eq "hip") { Copy-Item "$root\build\gpu.exe" "$pkg\build\"; Copy-Item "$root\build\hipblaslt" "$pkg\build\" -Recurse }
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
Add-Type -AssemblyName PresentationCore
$svg = [xml](Get-Content -Raw "$root\web\logo.svg")
$box = $svg.svg.viewBox -split " " | ForEach-Object { [double]$_ }
$shape = [Windows.Media.Geometry]::Parse("F0 " + $svg.svg.path.d)
$fill = [Windows.Media.SolidColorBrush]::new([Windows.Media.ColorConverter]::ConvertFromString($svg.svg.fill))
$sizes = 16, 20, 24, 32, 40, 48, 64, 256
$pngs = foreach ($s in $sizes) {
    $k = $s * 880 / 1024 / [Math]::Max($box[2], $box[3])
    $v = [Windows.Media.DrawingVisual]::new()
    $dc = $v.RenderOpen()
    $dc.PushTransform([Windows.Media.MatrixTransform]::new($k, 0, 0, $k, ($s - $k * $box[2]) / 2 - $k * $box[0], ($s - $k * $box[3]) / 2 - $k * $box[1]))
    $dc.DrawGeometry($fill, $null, $shape)
    $dc.Close()
    $bmp = [Windows.Media.Imaging.RenderTargetBitmap]::new($s, $s, 96, 96, [Windows.Media.PixelFormats]::Pbgra32)
    $bmp.Render($v)
    $enc = [Windows.Media.Imaging.PngBitmapEncoder]::new()
    $enc.Frames.Add([Windows.Media.Imaging.BitmapFrame]::Create($bmp))
    $ms = [IO.MemoryStream]::new()
    $enc.Save($ms)
    , $ms.ToArray()
}
$ico = [IO.BinaryWriter]::new([IO.File]::Create("$t\chrysalis.ico"))
$ico.Write([uint16]0); $ico.Write([uint16]1); $ico.Write([uint16]$sizes.Count)
$at = 6 + 16 * $sizes.Count
for ($i = 0; $i -lt $sizes.Count; $i++) {
    $ico.Write([byte]($sizes[$i] % 256)); $ico.Write([byte]($sizes[$i] % 256)); $ico.Write([uint16]0); $ico.Write([uint16]1); $ico.Write([uint16]32)
    $ico.Write([uint32]$pngs[$i].Length); $ico.Write([uint32]$at)
    $at += $pngs[$i].Length
}
foreach ($p in $pngs) { $ico.Write($p) }
$ico.Close()
& $Bun build --compile --windows-icon="$t\chrysalis.ico" --windows-title=Chrysalis --windows-description=Chrysalis --windows-publisher="Chrysalis Inference" "$root\web\launch.ts" --outfile "$pkg\Chrysalis.exe" 2>&1 | Out-Null
if (-not (Test-Path "$pkg\Chrysalis.exe")) { throw "Chrysalis.exe did not build" }
if (-not $Version -and (Get-Command git -ErrorAction SilentlyContinue)) { $Version = git -C $root describe --tags --always 2>$null }
if ($Version) { $Version | Out-File -Encoding ascii "$pkg\VERSION" }
$iscc = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) {
    Invoke-WebRequest -UseBasicParsing "https://github.com/jrsoftware/issrc/releases/download/is-6_7_4/innosetup-6.7.4.exe" -OutFile "$t\innosetup.exe"
    if ((Get-FileHash "$t\innosetup.exe" -Algorithm SHA256).Hash.ToLower() -ne "a549929319f9120716677c685535ca4671e4b937b5fdc87e2585f32d8eb30e8c") { throw "Inno Setup's download does not match its checksum" }
    Start-Process "$t\innosetup.exe" -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER" -Wait
}
$setup = Join-Path $Out "Chrysalis-Setup.exe"
& $iscc /Q "/DRoot=$pkg" "/DVersion=$Version" "/DIcon=$t\chrysalis.ico" "/O$Out" "$PSScriptRoot\setup.iss"
if ($LASTEXITCODE -or -not (Test-Path $setup)) { throw "Chrysalis-Setup.exe did not build" }
Remove-Item $t -Recurse -Force
"{0}: {1:N0} MB unpacked, {2:N0} MB installer" -f $pkg, ((Get-ChildItem $pkg -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), ((Get-Item $setup).Length / 1MB)
