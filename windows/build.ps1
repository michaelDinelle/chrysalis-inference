param([string]$Sdk = $env:HIP_PATH, [string]$Llama = "", [string]$CMake = "cmake", [string]$Ninja = "ninja", [ValidateSet("hip", "cuda")][string]$Backend = "hip", [string]$Cuda = $env:CUDA_PATH, [string]$Arch = "sm_86")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Sdk) { throw "AMD's HIP SDK is needed to build the engines: install it, or pass -Sdk <its folder>" }
if ($Backend -eq "cuda" -and -not $Cuda) { throw "NVIDIA's CUDA Toolkit is needed to build the CUDA engines: install it, or pass -Cuda <its folder>" }
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

$failed = 0
function Finish($r) {
    $r.Proc.WaitForExit()
    if ($r.Proc.ExitCode -or -not (Test-Path $r.File)) {
        $script:failed++
        "$($r.Name) failed:"
        Get-Content $r.Log, "$($r.Log).out" | Select-String "error" | Select-Object -First 20 | ForEach-Object { "  " + $_.Line }
    } elseif ($r.File -match "\.(dll|exe)$") { "{0} {1:N0} bytes" -f $r.Name, (Get-Item $r.File).Length }
}
function Invoke-All($jobs) {
    $running = [System.Collections.Queue]::new()
    foreach ($j in $jobs) {
        if ($running.Count -ge [Environment]::ProcessorCount) { Finish $running.Dequeue() }
        $log = Join-Path $out ($j.Name + ".log")
        $p = Start-Process $j.Exe -ArgumentList ($j.Args | ForEach-Object { if ($_ -match "\s") { "`"$_`"" } else { $_ } }) -NoNewWindow -PassThru -RedirectStandardError $log -RedirectStandardOutput "$log.out"
        $null = $p.Handle
        $running.Enqueue(@{ Name = $j.Name; File = $j.File; Log = $log; Proc = $p })
    }
    while ($running.Count) { Finish $running.Dequeue() }
}

if ($Backend -eq "cuda") {
    $Cuda = $Cuda.TrimEnd("\")
    $nvcc = Join-Path $Cuda "bin\nvcc.exe"
    $flags = @("-O3", "-std=c++20", "-arch=$Arch", "-Xcompiler", "/Gy,/Gw,/Zc:preprocessor")
    $libs = @("-L$Cuda\lib\x64", "-lcublasLt", "-Xlinker", "/OPT:REF")
    $c = "engine\cuda"
    $g = "engine\gpu"
    $b27 = @("$c\27b\relayout.cu", "$c\27b\quant.cu", "$c\27b\ops.cu", "$c\gdn.cu", "$c\27b\attn.cu", "$c\vision.cu", "$g\27b\model.cpp", "$g\27b\load.cpp", "$g\27b\api.cpp")
    $fn = @("$c\flash-next\relayout.cu", "$c\flash-next\quant.cu", "$c\flash-next\ops.cu", "$c\gdn.cu", "$c\flash-next\attn.cu", "$c\flash-next\moe.cu", "$c\flash-next\ple.cu",
            "$c\vision.cu", "$g\flash-next\model.cpp", "$g\flash-next\load.cpp", "$c\flash-next\experts.cpp", "$g\flash-next\api.cpp")
    $fnCpu = @("$g\flash-next\cpu_moe.cpp", "engine\cpu_exact.cpp")
    New-Item -ItemType Directory -Force "$out\o27", "$out\ofn" | Out-Null
    $objects = @(); $o27 = @(); $ofn = @()
    foreach ($s in $b27) { $o = "o27\" + ($s -replace "\\", "_") + ".obj"; $o27 += "$out\$o"; $objects += @{ Name = $o; File = "$out\$o"; Exe = $nvcc; Args = $flags + @("-I$c\27b", "-I$c", "-I$g\27b", "-I$g", "-x", "cu", "-c", "-o", "$out\$o", $s) } }
    foreach ($s in $fn) { $o = "ofn\" + ($s -replace "\\", "_") + ".obj"; $ofn += "$out\$o"; $objects += @{ Name = $o; File = "$out\$o"; Exe = $nvcc; Args = $flags + @("-I$c\flash-next", "-I$c", "-I$g\flash-next", "-I$g", "-I$Llama\ggml\include", "-x", "cu", "-c", "-o", "$out\$o", $s) } }
    foreach ($s in $fnCpu) { $o = "ofn\" + ($s -replace "\\", "_") + ".obj"; $ofn += "$out\$o"; $objects += @{ Name = $o; File = "$out\$o"; Exe = $clang; Args = @("-O3", "-std=c++20", "-ffunction-sections", "-fdata-sections", "-Wno-unused-result", "-I$c\flash-next", "-I$c", "-I$g\flash-next", "-I$g", "-I$Llama\ggml\include", "-c", "-o", "$out\$o", $s) } }
    Invoke-All $objects
    $rt = Get-Item (& $clang --rtlib=compiler-rt -print-libgcc-file-name)
    if (-not $failed) {
        Invoke-All @(
            @{ Name = "libchrysalis.dll"; File = "$out\libchrysalis.dll"; Exe = $nvcc; Args = $flags + @("-shared", "-o", "$out\libchrysalis.dll") + $o27 + $libs },
            @{ Name = "libchrysalis_fn.dll"; File = "$out\libchrysalis_fn.dll"; Exe = $nvcc; Args = $flags + @("-shared", "-o", "$out\libchrysalis_fn.dll") + $ofn + $libs + @("-L$($ggmlLib.DirectoryName)", "-lggml-cpu", "-lggml-base", "-L$($rt.DirectoryName)", "-l$($rt.BaseName)") }
        )
    }
} else {
    $flags = @("-x", "hip", "--offload-arch=gfx1100", "-O3", "-std=c++20", "-Wno-unused-result", "-I$Sdk\include")
    $libs = @("-L$Sdk\lib", "-lamdhip64", "-lhipblaslt")
    $h = "engine\hip"
    $g = "engine\gpu"
    $b27 = @("$h\27b\relayout.hip", "$h\27b\quant.hip", "$h\27b\ops.hip", "$h\gdn.hip", "$h\27b\attn.hip", "$h\vision.hip", "$h\gpu.hip", "$g\27b\model.cpp", "$g\27b\load.cpp", "$g\27b\api.cpp")
    $fn = @("$h\flash-next\relayout.hip", "$h\flash-next\quant.hip", "$h\flash-next\ops.hip", "$h\gdn.hip", "$h\flash-next\attn.hip", "$h\flash-next\moe.hip", "$h\flash-next\ple.hip",
            "$h\vision.hip", "$h\gpu.hip", "$g\flash-next\model.cpp", "$g\flash-next\load.cpp", "$h\flash-next\experts.cpp", "$g\flash-next\api.cpp") +
          (Get-ChildItem "$g\flash-next\cpu_*.cpp" | ForEach-Object { "$g\flash-next\" + $_.Name }) + @("engine\cpu_exact.cpp")
    Invoke-All @(
        @{ Name = "libchrysalis.dll"; File = "$out\libchrysalis.dll"; Exe = $clang; Args = $flags + @("-I$h\27b", "-I$h", "-I$g\27b", "-I$g") + $b27 + @("-shared", "-o", "$out\libchrysalis.dll") + $libs },
        @{ Name = "libchrysalis_fn.dll"; File = "$out\libchrysalis_fn.dll"; Exe = $clang; Args = $flags + @("-I$h\flash-next", "-I$h", "-I$g\flash-next", "-I$g", "-I$Llama\ggml\include") + $fn + @("-shared", "-o", "$out\libchrysalis_fn.dll") + $libs + @("-L$($ggmlLib.DirectoryName)", "-lggml-cpu", "-lggml-base") },
        @{ Name = "gpu.exe"; File = "$out\gpu.exe"; Exe = $clang; Args = $flags + @("$h\gpu_pick.hip", "$h\gpu.hip", "-o", "$out\gpu.exe") + $libs }
    )
}
$fsp = Join-Path ${env:ProgramFiles(x86)} "WinFsp"
if (-not (Test-Path "$fsp\samples\memfs\memfs.cpp")) { throw "WinFsp with its developer files is needed to build incognito's RAM drive: msiexec /i winfsp.msi ADDLOCAL=ALL" }
Push-Location $out
cl /nologo /O2 /EHsc /DNDEBUG /DUNICODE /D_UNICODE "/I$fsp\inc" "$fsp\samples\memfs\memfs.cpp" "$fsp\samples\memfs\memfs-main.c" /Fe:ramdrive.exe /link "$fsp\lib\winfsp-x64.lib" /DELAYLOAD:winfsp-x64.dll delayimp.lib advapi32.lib | Out-Null
Pop-Location
if (-not (Test-Path "$out\ramdrive.exe")) { $failed++; "ramdrive.exe failed" }
Remove-Item "$out\memfs.obj", "$out\memfs-main.obj" -ErrorAction SilentlyContinue
if ($Backend -eq "cuda") {
    Get-ChildItem "$Cuda\bin" -Recurse -Filter "cublasLt64_*.dll" | Select-Object -First 1 | Copy-Item -Destination $out -Force
} else {
    Copy-Item (Join-Path $Sdk "bin\hipblaslt.dll") $out -Force
    $kernels = Join-Path $out "hipblaslt\library"
    New-Item -ItemType Directory -Force $kernels | Out-Null
    Get-ChildItem (Join-Path $Sdk "bin\hipblaslt\library") -File | Where-Object { $_.Name -match "gfx1100" -or $_.Name -notmatch "gfx" } | Copy-Item -Destination $kernels -Force
}
exit $failed
