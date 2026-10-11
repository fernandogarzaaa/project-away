; Inno Setup script (free) for Project Away. Built by .github/workflows/release.yml:
;   iscc /DAppVersion=0.1.0 packaging\windows\away.iss
; Expects the PyInstaller onedir output in dist\ProjectAway\.
;
; One click: installs the app (every feature bundled: tracking, dictation, tray, settings, training),
; then downloads the MediaPipe + Whisper models in a visible console window, enables start-at-sign-in
; and a desktop shortcut (both pre-ticked), and launches the tray app.
#ifndef AppVersion
  #define AppVersion "0.1.2"
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
Name: "models"; Description: "Download hand-tracking and speech models now (recommended, ~1.6 GB)"; GroupDescription: "Models:"
Name: "models\small"; Description: "Use the smaller speech model instead (Whisper small, ~0.5 GB) for slower PCs"; GroupDescription: "Models:"; Flags: unchecked
Name: "autostart"; Description: "Start Project Away when I sign in"; GroupDescription: "Options:"
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Options:"

[Files]
Source: "..\..\dist\ProjectAway\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\Project Away"; Filename: "{app}\ProjectAway.exe"
Name: "{group}\Project Away: re-download models"; Filename: "{app}\ProjectAway.exe"; Parameters: "setup --yes --skip-probe"
Name: "{group}\Project Away settings"; Filename: "{app}\ProjectAway.exe"; Parameters: "settings"
Name: "{group}\Uninstall Project Away"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Project Away"; Filename: "{app}\ProjectAway.exe"; Tasks: desktopicon

[Registry]
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "ProjectAway"; ValueData: """{app}\ProjectAway.exe"" tray"; Flags: uninsdeletevalue; Tasks: autostart

[Run]
; Model download during installation. ProjectAway.exe is a windowed app; for `setup` it opens its own
; console window so the download progress is visible. A failed download is retried on first launch.
Filename: "{app}\ProjectAway.exe"; Parameters: "setup --yes --skip-probe"; StatusMsg: "Downloading hand-tracking and speech models (~1.6 GB). A console window shows progress..."; Flags: waituntilterminated; Tasks: models and not models\small
Filename: "{app}\ProjectAway.exe"; Parameters: "setup --yes --skip-probe --asr-model small"; StatusMsg: "Downloading hand-tracking and speech models (~0.5 GB). A console window shows progress..."; Flags: waituntilterminated; Tasks: models\small
; Launch the tray app when the wizard closes (ticked by default).
Filename: "{app}\ProjectAway.exe"; Parameters: "tray"; Description: "Launch Project Away now"; Flags: nowait postinstall skipifsilent

[UninstallRun]
; remove the start-at-login entry the app itself may have created (`away autostart on`)
Filename: "{sys}\reg.exe"; Parameters: "delete HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v ProjectAway /f"; Flags: runhidden; RunOnceId: "DelRun"

[UninstallDelete]
; models/config live in %LOCALAPPDATA%\ProjectAway and %APPDATA%\ProjectAway; ask-free cleanup is opt-in:
; run "ProjectAway.exe uninstall" before uninstalling to delete them.
Type: filesandordirs; Name: "{app}"
