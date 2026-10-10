; Inno Setup script (free) for Project Away. Built by .github/workflows/release.yml:
;   iscc /DAppVersion=0.1.0 packaging\windows\away.iss
; Expects the PyInstaller onedir output in dist\ProjectAway\.
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6F0B1D7E-3C2A-4E4B-9C55-7A9E2B1D0A11}
AppName=Project Away
AppVersion={#AppVersion}
AppPublisher=Fernando Garza
DefaultDirName={localappdata}\Programs\ProjectAway
DefaultGroupName=Project Away
PrivilegesRequired=lowest
OutputDir=..\..\dist
OutputBaseFilename=ProjectAway-{#AppVersion}-windows-x64-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayName=Project Away
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "autostart"; Description: "Start Project Away when I sign in"; GroupDescription: "Options:"; Flags: unchecked
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Options:"; Flags: unchecked

[Files]
Source: "..\..\dist\ProjectAway\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Project Away"; Filename: "{app}\ProjectAway.exe"
Name: "{group}\Project Away setup (download models)"; Filename: "{app}\ProjectAway.exe"; Parameters: "setup"
Name: "{group}\Uninstall Project Away"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Project Away"; Filename: "{app}\ProjectAway.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "ProjectAway"; ValueData: """{app}\ProjectAway.exe"" tray"; Flags: uninsdeletevalue; Tasks: autostart

[Run]
Filename: "{app}\ProjectAway.exe"; Description: "Launch Project Away"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; remove the start-at-login entry the app itself may have created (`away autostart on`)
Filename: "{sys}\reg.exe"; Parameters: "delete HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v ProjectAway /f"; Flags: runhidden; RunOnceId: "DelRun"

[UninstallDelete]
; models/config live in %LOCALAPPDATA%\ProjectAway and %APPDATA%\ProjectAway; ask-free cleanup is opt-in:
; run "ProjectAway.exe uninstall" before uninstalling to delete them.
Type: filesandordirs; Name: "{app}"
