; Inno Setup script (free) for Project Away. Built by .github/workflows/release.yml:
;   iscc /DAppVersion=0.1.0 packaging\windows\away.iss
; Expects the PyInstaller onedir output in dist\ProjectAway\.
;
; One click: installs the app (every feature bundled: tracking, dictation, tray, settings, training),
; then downloads the MediaPipe + Whisper models in a visible console window, enables start-at-sign-in
; and a desktop shortcut (both pre-ticked), and launches the tray app.
;
; Auto-update (away/app/updater.py) runs this installer silently, from a detached helper that first waits
; for every ProjectAway.exe of this install to exit (so no file is locked):
;   ProjectAway-X.Y.Z-windows-x64-setup.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS
;       /NORESTARTAPPLICATIONS /AUTOUPDATE=1 /DIR="<install folder>" /LOG="<data dir>\logs\update-install.log"
; CloseApplications=force: if a copy is somehow still running, Setup closes it instead of stopping at
; "Preparing to install" (which /SUPPRESSMSGBOXES would answer with abort). /AUTOUPDATE=1 relaunches the
; tray app when the install finishes. A silent install skips the model
; download when the models are already on disk (an upgrade keeps them); missing ones download on first launch.
#ifndef AppVersion
  #define AppVersion "0.1.15"
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
CloseApplications=force
CloseApplicationsFilter=*.exe,*.dll,*.pyd
RestartApplications=no
SetupLogging=yes

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
Filename: "{app}\ProjectAway.exe"; Parameters: "setup --yes --skip-probe"; StatusMsg: "Detecting your hardware and downloading the models it needs (0.5-2.2 GB; with an NVIDIA GPU also the GPU pack). A console window shows progress..."; Flags: waituntilterminated; Tasks: models and not models\small; Check: ShouldDownloadModels
Filename: "{app}\ProjectAway.exe"; Parameters: "setup --yes --skip-probe --asr-model small"; StatusMsg: "Downloading hand-tracking and speech models (~0.5 GB). A console window shows progress..."; Flags: waituntilterminated; Tasks: models\small; Check: ShouldDownloadModels
; Launch the tray app when the wizard closes (ticked by default).
Filename: "{app}\ProjectAway.exe"; Parameters: "tray"; Description: "Launch Project Away now"; Flags: nowait postinstall skipifsilent
; Auto-update: relaunch the tray app after a silent update (only when the updater passed /AUTOUPDATE=1).
Filename: "{app}\ProjectAway.exe"; Parameters: "tray"; Flags: nowait; Check: IsAutoUpdate

[UninstallRun]
; remove the start-at-login entry the app itself may have created (`away autostart on`)
Filename: "{sys}\reg.exe"; Parameters: "delete HKCU\Software\Microsoft\Windows\CurrentVersion\Run /v ProjectAway /f"; Flags: runhidden; RunOnceId: "DelRun"

[UninstallDelete]
; models/config live in %LOCALAPPDATA%\ProjectAway and %APPDATA%\ProjectAway; ask-free cleanup is opt-in:
; run "ProjectAway.exe uninstall" before uninstalling to delete them.
Type: filesandordirs; Name: "{app}"

[Code]
function IsAutoUpdate: Boolean;
begin
  Result := ExpandConstant('{param:AUTOUPDATE|0}') = '1';
end;

function ModelsPresent: Boolean;
begin
  Result := FileExists(ExpandConstant('{localappdata}\ProjectAway\models\hand_landmarker.task'))
    and FileExists(ExpandConstant('{localappdata}\ProjectAway\models\face_landmarker.task'));
end;

// interactive installs keep the user's choice; silent installs (auto-update) skip models already on disk
function ShouldDownloadModels: Boolean;
begin
  Result := (not WizardSilent) or (not ModelsPresent);
end;
