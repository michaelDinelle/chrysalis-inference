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
Name: "{autoprograms}\Uninstall Chrysalis"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\Chrysalis.exe"; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,Chrysalis}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
Type: filesandordirs; Name: "{%USERPROFILE}\.config\chrysalis"

[Code]
const
  Entry = 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{B788DC7D-373F-4C2B-820D-9525017A82D2}_is1';

function InitializeSetup(): Boolean;
var
  Uninstaller: String;
  Code: Integer;
begin
  Result := True;
  if WizardSilent or not (RegQueryStringValue(HKCU, Entry, 'UninstallString', Uninstaller) or RegQueryStringValue(HKLM, Entry, 'UninstallString', Uninstaller)) then
    Exit;
  case TaskDialogMsgBox('Chrysalis is already on this computer', 'Update it to this version and keep its models, or uninstall it: that removes Chrysalis with its models, its caches and its settings.', mbConfirmation, MB_YESNOCANCEL, ['Update', 'Uninstall', 'Cancel'], 0) of
    IDNO:
      begin
        Exec(RemoveQuotes(Uninstaller), '/SILENT', '', SW_SHOW, ewNoWait, Code);
        Result := False;
      end;
    IDCANCEL:
      Result := False;
  end;
end;
