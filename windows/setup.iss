[Setup]
AppId={{B788DC7D-373F-4C2B-820D-9525017A82D2}
AppName=Chrysalis
AppVersion={#Version}
AppVerName=Chrysalis {#Version}
AppPublisherURL=https://github.com/michaelDinelle/chrysalis-inference
DefaultDirName={autopf}\Chrysalis
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile={#Root}\LICENSE
SetupIconFile={#Icon}
UninstallDisplayIcon={app}\Chrysalis.exe
WizardStyle=modern
Compression=lzma2/ultra64
SolidCompression=yes
LZMANumBlockThreads=8
LZMABlockSize=131072
LZMAUseSeparateProcess=yes
OutputBaseFilename=Chrysalis-Setup

[Tasks]
Name: desktopicon; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
Type: filesandordirs; Name: "{app}\build"
Type: filesandordirs; Name: "{app}\chrysalis"
Type: filesandordirs; Name: "{app}\web"
Type: filesandordirs; Name: "{app}\data\venv"

[Files]
Source: "{#Root}\*"; Excludes: "\data\models"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Root}\data\models\*"; DestDir: "{app}\data\models"; Flags: ignoreversion recursesubdirs createallsubdirs uninsneveruninstall

[Icons]
Name: "{autoprograms}\Chrysalis"; Filename: "{app}\Chrysalis.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Chrysalis"; Filename: "{app}\Chrysalis.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Chrysalis.exe"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,Chrysalis}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\build"
Type: filesandordirs; Name: "{app}\chrysalis"
Type: filesandordirs; Name: "{app}\web"
Type: filesandordirs; Name: "{app}\data\venv"
Type: filesandordirs; Name: "{app}\data\cache"
