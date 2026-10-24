# Chrysalis on Windows, in one line (PowerShell):
#   irm https://github.com/michaelDinelle/chrysalis-inference/releases/latest/download/install.ps1 | iex
# It runs the Linux release in WSL2, which reaches the RX 7900 through Windows's own AMD driver (Adrenalin; ROCm's
# runtime finds it through /dev/dxg). It sets up WSL2 and an Ubuntu of Chrysalis's own (once a restart, if WSL was not
# on yet: run the same line again after it), puts WSL on this computer's network with the RAM Flash-Next needs, unpacks
# Chrysalis there, and adds Chrysalis to the Start menu, and chrysalis.sh to the command line. The chat is
# http://localhost:8081, the API http://localhost:8080/v1 (WSL's mirrored networking shares both with Windows).
# Removing it: wsl --unregister Chrysalis; Remove-NetFirewallHyperVRule -Name Chrysalis; the Start menu's Chrysalis, and
# %LOCALAPPDATA%\Chrysalis (on the user's Path). .wslconfig keeps its settings.
#   $env:CHRYSALIS_BASE: where the release is, in place of the latest GitHub release (a URL)
# (Windows PowerShell turns a program's stderr into a failure when errors stop the script: programs' exit codes are
# checked instead, and cmdlets stop on their own errors)
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [Text.Encoding]::UTF8   # the check marks, and the Linux installer's lines as they come
$Distro = "Chrysalis"
$User = "chrysalis"
$Base = if ($env:CHRYSALIS_BASE) { $env:CHRYSALIS_BASE } else { "https://github.com/michaelDinelle/chrysalis-inference/releases/latest/download" }
$Home_ = Join-Path $env:LOCALAPPDATA "Chrysalis"
$Log = Join-Path $env:TEMP "chrysalis-install.log"
function Ok($t) { Write-Host "  $([char]0x2713) " -ForegroundColor Magenta -NoNewline; Write-Host $t }
function Fail($t) { Get-Content $Log -Tail 30 -ErrorAction SilentlyContinue; Write-Host "`n  Chrysalis: $t (above: what it said)" -ForegroundColor Red; exit 1 }
# a command in Chrysalis's Ubuntu, quietly (into the log)
function InWsl($cmd, $as = $User) { wsl.exe -d $Distro -u $as -- bash -lc $cmd *>> $Log; if ($LASTEXITCODE) { Fail "a step in WSL failed" } }
Write-Host "`n  Chrysalis`n" -ForegroundColor White

# Enabling WSL and the virtual machine platform takes an administrator and, the first time, a restart
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host "  Setting up WSL takes an administrator: asking Windows for one"
    $line = "irm $Base/install.ps1 | iex"
    if ($env:CHRYSALIS_BASE) { $line = "`$env:CHRYSALIS_BASE='$env:CHRYSALIS_BASE'; $line" }
    Start-Process powershell -Verb RunAs -ArgumentList "-NoExit", "-Command", $line
    return
}
# WSL itself: the two Windows features it runs on (a restart, the first time), and its package from Microsoft's own
# releases, which `wsl --install` fetches from the Store, which some editions (LTSC, managed PCs) do not have
$off = "Microsoft-Windows-Subsystem-Linux", "VirtualMachinePlatform" | Where-Object { (Get-WindowsOptionalFeature -Online -FeatureName $_).State -ne "Enabled" }
foreach ($f in $off) { Enable-WindowsOptionalFeature -Online -FeatureName $f -All -NoRestart -WarningAction SilentlyContinue -ErrorAction Stop | Out-Null }
wsl.exe --status *> $null
if ($LASTEXITCODE) {
    $msi = (Invoke-RestMethod https://api.github.com/repos/microsoft/WSL/releases/latest).assets | Where-Object name -like "*x64.msi" | Select-Object -First 1
    Invoke-WebRequest $msi.browser_download_url -OutFile "$env:TEMP\wsl.msi" -UseBasicParsing
    Start-Process msiexec.exe -Wait -ArgumentList "/i", "$env:TEMP\wsl.msi", "/quiet", "/norestart"
}
if ($off) { Write-Host "`n  Restart Windows, then run the same line again: it carries on from there.`n" -ForegroundColor Magenta; return }
Ok "WSL"

# WSL's settings in .wslconfig, each where the user has not set it. Mirrored networking puts WSL on this computer's own
# network: behind WSL's NAT no phone reaches the chat, and the link never hears another Chrysalis's mDNS. Flash-Next's
# cold experts live in RAM: WSL gets all but 6 GB of it (by default it gets half).
$cfg = Join-Path $env:USERPROFILE ".wslconfig"
$lines = @(if (Test-Path $cfg) { Get-Content $cfg })
$gb = [math]::Floor((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$want = [ordered]@{ networkingMode = "mirrored" }
if ($gb -ge 60) { $want.memory = "$($gb - 6)GB" }
$add = @($want.Keys | Where-Object { -not ($lines -match "^\s*$_\s*=") } | ForEach-Object { "$_=$($want[$_])" })
if ($add) {
    $lines = foreach ($l in $lines) { $l; if ($l.Trim() -eq "[wsl2]") { $add; $add = @() } }
    if ($add) { $lines = @($lines) + "[wsl2]" + $add }
    Set-Content $cfg $lines
    wsl.exe --shutdown
}
# What comes to WSL from the network then meets the Hyper-V firewall, which lets in only what a rule names (WSL's own
# rules: mDNS): the chat (TCP 8081), the link (8082) and the trust check (8083), from private addresses alone. They
# listen there only after `chrysalis.sh start --host 0.0.0.0`.
if (-not (Get-NetFirewallHyperVRule -Name Chrysalis -ErrorAction SilentlyContinue)) {
    New-NetFirewallHyperVRule -Name Chrysalis -DisplayName "Chrysalis Inference" -Direction Inbound -Action Allow `
        -VMCreatorId "{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}" -Protocol TCP -LocalPorts 8081-8083 `
        -RemoteAddresses 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 100.64.0.0/10, 169.254.0.0/16 | Out-Null
}

if (-not ((wsl.exe -l -q) -replace "`0", "" | Where-Object { $_.Trim() -eq $Distro })) {
    wsl.exe --install Ubuntu-24.04 --name $Distro --no-launch --web-download *>> $Log
    if ($LASTEXITCODE) { Fail "Ubuntu did not install" }
    InWsl "id $User >/dev/null 2>&1 || useradd -m -s /bin/bash $User; loginctl enable-linger $User; printf '[user]\ndefault=$User\n[boot]\nsystemd=true\n' > /etc/wsl.conf" root
    wsl.exe --terminate $Distro *> $null
}
# WSL gives the distro a Start menu entry and a Terminal profile of its own: a Linux shell named Chrysalis, which
# Start's search puts first
$wslReg = Get-ChildItem HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss | Get-ItemProperty | Where-Object DistributionName -eq $Distro
@($wslReg.ShortcutPath, $wslReg.TerminalProfilePath) | Where-Object { $_ } | Remove-Item -ErrorAction SilentlyContinue
Ok "Ubuntu for Chrysalis"

# AMD's runtime for ROCm in WSL: the GPU through Windows's driver
InWsl "dpkg -s amdgpu-install >/dev/null 2>&1 || { apt-get update -q && apt-get install -yq curl ca-certificates && curl -fsSL -o /tmp/amdgpu-install.deb https://repo.radeon.com/amdgpu-install/7.2/ubuntu/noble/amdgpu-install_7.2.70200-1_all.deb && apt-get install -yq /tmp/amdgpu-install.deb && amdgpu-install -y --usecase=wsl --no-dkms; }" root
Ok "AMD's ROCm for WSL"

# Chrysalis itself, unpacked in the home folder there (nothing to install: the folder is Chrysalis), and its first
# start, which downloads the model: its lines shown as they come
$run = "~/chrysalis/chrysalis.sh start"
wsl.exe -d $Distro -u $User --cd "~" -- bash -lc "curl -fsSL $Base/chrysalis-linux-x86_64.tar.gz | tar -xz && $run"
if ($LASTEXITCODE) { Fail "Chrysalis did not start" }

# The Start menu's Chrysalis, with the butterfly: a window that is Chrysalis. WSL stops a distro soon after its last
# wsl.exe exits, services and all, so closing the window stops it; meanwhile it shows the request log.
New-Item -ItemType Directory -Force $Home_ | Out-Null
Invoke-WebRequest "$Base/chrysalis.ico" -OutFile "$Home_\chrysalis.ico" -UseBasicParsing -ErrorAction SilentlyContinue
$lnk = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path ([Environment]::GetFolderPath("Programs")) "Chrysalis.lnk"))
$lnk.TargetPath = "wsl.exe"
$lnk.Arguments = "-d $Distro -u $User -- bash -lc `"$run && explorer.exe http://localhost:8081; exec ~/chrysalis/chrysalis.sh logs`""
if (Test-Path "$Home_\chrysalis.ico") { $lnk.IconLocation = "$Home_\chrysalis.ico" }
$lnk.Save()
# chrysalis.sh at Windows's command line too, as in Linux's (`chrysalis.sh start --host 0.0.0.0`, `chrysalis.sh allow`):
# a command typed without its extension finds chrysalis.sh.cmd
Set-Content -Encoding ascii "$Home_\chrysalis.sh.cmd" "@wsl.exe -d $Distro -u $User --cd ~ -- ~/chrysalis/chrysalis.sh %*"
$path = [Environment]::GetEnvironmentVariable("Path", "User")
if (($path -split ";") -notcontains $Home_) { [Environment]::SetEnvironmentVariable("Path", (@($path, $Home_) | Where-Object { $_ }) -join ";", "User") }
Ok "Chrysalis in the Start menu"
Start-Process $lnk.FullName
