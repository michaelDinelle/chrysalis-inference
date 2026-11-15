param([string]$Sdk = $env:HIP_PATH, [string]$Llama = "", [string]$CMake = "cmake", [string]$Ninja = "ninja")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Sdk) { throw "AMD's HIP SDK is needed to build the engines: install it, or pass -Sdk <its folder>" }
$Sdk = $Sdk.TrimEnd("\")
if (-not $Llama) { $Llama = Join-Path $root "data\llama.cpp" }
$clang = Join-Path $Sdk "bin\clang++.exe"
$out = Join-Path $root "build"
New-Item -ItemType Directory -Force $out | Out-Null
Set-Location $root
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "Visual Studio's C++ build tools are needed to build the engines" }
$vs = & $vswhere -latest -products * -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object { if ($_ -match "^(.*?)=(.*)$") { Set-Item "env:$($Matches[1])" $Matches[2] } }

$ggml = Join-Path $out "ggml"
if (-not (Test-Path (Join-Path $out "ggml-cpu.dll"))) {
    & $CMake -S $Llama -B $ggml -G Ninja "-DCMAKE_MAKE_PROGRAM=$Ninja" "-DCMAKE_C_COMPILER=$Sdk\bin\clang-cl.exe" "-DCMAKE_CXX_COMPILER=$Sdk\bin\clang-cl.exe" `
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON `
        -DGGML_OPENMP=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF | Out-Null
    if ($LASTEXITCODE) { throw "configuring ggml failed" }
    & $CMake --build $ggml --target ggml-cpu ggml-base | Out-Null
    if ($LASTEXITCODE) { throw "building ggml failed" }
    Copy-Item (Join-Path $ggml "bin\ggml-cpu.dll"), (Join-Path $ggml "bin\ggml-base.dll") $out
}
$ggmlLib = Get-ChildItem $ggml -Recurse -Filter "ggml-cpu.lib" | Select-Object -First 1

$flags = @("-x", "hip", "--offload-arch=gfx1100", "-O3", "-std=c++20", "-Wno-unused-result", "-I$Sdk\include")
$libs = @("-L$Sdk\lib", "-lamdhip64", "-lhipblaslt")
$h = "engine\hip"
$b27 = @("$h\27b\relayout.hip", "$h\27b\quant.hip", "$h\27b\ops.hip", "$h\gdn.hip", "$h\27b\attn.hip", "$h\vision.hip", "$h\gpu.hip", "$h\27b\model.cpp", "$h\27b\load.cpp", "$h\27b\api.cpp")
$fn = @("$h\flash-next\relayout.hip", "$h\flash-next\quant.hip", "$h\flash-next\ops.hip", "$h\gdn.hip", "$h\flash-next\attn.hip", "$h\flash-next\moe.hip", "$h\flash-next\ple.hip",
        "$h\vision.hip", "$h\gpu.hip", "$h\flash-next\model.cpp", "$h\flash-next\load.cpp", "$h\flash-next\experts.cpp", "$h\flash-next\api.cpp") +
      (Get-ChildItem "$h\flash-next\cpu_*.cpp" | ForEach-Object { "$h\flash-next\" + $_.Name }) + @("engine\cpu_exact.cpp")
$jobs = @(
    @{ Name = "libchrysalis.dll"; Args = $flags + @("-I$h\27b") + $b27 + @("-shared", "-o", "$out\libchrysalis.dll") + $libs },
    @{ Name = "libchrysalis_fn.dll"; Args = $flags + @("-I$h\flash-next", "-I$Llama\ggml\include") + $fn + @("-shared", "-o", "$out\libchrysalis_fn.dll") + $libs + @("-L$($ggmlLib.DirectoryName)", "-lggml-cpu", "-lggml-base") },
    @{ Name = "gpu.exe"; Args = $flags + @("$h\gpu_pick.hip", "$h\gpu.hip", "-o", "$out\gpu.exe") + $libs }
)
$running = foreach ($j in $jobs) {
    $log = Join-Path $out ($j.Name + ".log")
    @{ Name = $j.Name; Log = $log; Proc = Start-Process $clang -ArgumentList ($j.Args | ForEach-Object { if ($_ -match "\s") { "`"$_`"" } else { $_ } }) -NoNewWindow -PassThru -RedirectStandardError $log -RedirectStandardOutput "$log.out" }
}
$failed = 0
foreach ($r in $running) {
    $r.Proc.WaitForExit()
    $file = Join-Path $out $r.Name
    if ($r.Proc.ExitCode -or -not (Test-Path $file)) {
        $failed++
        "$($r.Name) failed:"
        Get-Content $r.Log | Select-String "error" | Select-Object -First 20 | ForEach-Object { "  " + $_.Line }
    } else { "{0} {1:N0} bytes" -f $r.Name, (Get-Item $file).Length }
}
Copy-Item (Join-Path $Sdk "bin\hipblaslt.dll") $out -Force
$kernels = Join-Path $out "hipblaslt\library"
New-Item -ItemType Directory -Force $kernels | Out-Null
Get-ChildItem (Join-Path $Sdk "bin\hipblaslt\library") -File | Where-Object { $_.Name -match "gfx1100" -or $_.Name -notmatch "gfx" } | Copy-Item -Destination $kernels -Force
exit $failed
