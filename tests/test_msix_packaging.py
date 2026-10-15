"""MSIX packaging, SignPath signing workflow and public policy (scripts/build_msix.py, packaging/msix, release.yml)."""

import importlib.util
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
NS = {
    "f": "http://schemas.microsoft.com/appx/manifest/foundation/windows10",
    "uap": "http://schemas.microsoft.com/appx/manifest/uap/windows10",
    "uap3": "http://schemas.microsoft.com/appx/manifest/uap/windows10/3",
    "desktop": "http://schemas.microsoft.com/appx/manifest/desktop/windows10",
    "rescap": "http://schemas.microsoft.com/appx/manifest/foundation/windows10/restrictedcapabilities",
}
SIGNPATH_FOUNDATION = "CN=SignPath Foundation, O=SignPath Foundation, L=Lewes, S=Delaware, C=US"


def _load():
    spec = importlib.util.spec_from_file_location("build_msix", ROOT / "scripts" / "build_msix.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bm = _load()


def _project_version() -> str:
    return re.search(r'(?m)^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text()).group(1)


# ------------------------------------------------------------------------------------------------ version
def test_version_comes_from_pyproject():
    assert bm.pyproject_version() == _project_version()


@pytest.mark.parametrize(
    ("given", "want"),
    [
        ("0.1.22", "0.1.22.0"),
        ("v0.1.23", "0.1.23.0"),
        ("1.2", "1.2.0.0"),
        ("3", "3.0.0.0"),
        ("1.2.0-rc.1", "1.2.0.0"),  # the Store wants the 4th number 0, so a pre-release suffix is dropped
        ("2.0.1.7", "2.0.1.0"),
    ],
)
def test_msix_version(given, want):
    assert bm.msix_version(given) == want


@pytest.mark.parametrize("bad", ["", "main", "x1.2", "70000.0.0"])
def test_msix_version_rejects(bad):
    with pytest.raises(bm.BuildError):
        bm.msix_version(bad)


# ----------------------------------------------------------------------------------------------- manifest
def _manifest(**kw) -> ET.Element:
    return ET.fromstring(bm.render_manifest(kw.pop("version", _project_version()), **kw).encode("utf-8"))


def test_manifest_renders_with_the_project_version_and_identity():
    root = _manifest(identity_name="12345FernandoGarza.ProjectAway", publisher="CN=ABCD-1234")
    ident = root.find("f:Identity", NS)
    assert ident is not None
    assert ident.get("Name") == "12345FernandoGarza.ProjectAway"
    assert ident.get("Publisher") == "CN=ABCD-1234"
    assert ident.get("Version") == bm.msix_version(_project_version())
    assert ident.get("ProcessorArchitecture") == "x64"
    assert root.findtext("f:Properties/f:PublisherDisplayName", namespaces=NS) == "Fernando Garza"
    assert root.findtext("f:Properties/f:DisplayName", namespaces=NS) == "Project Away"


def test_manifest_escapes_a_full_certificate_subject():
    text = bm.render_manifest("0.1.22", publisher='CN=A & "B", O=C', publisher_display_name="Me & Co")
    root = ET.fromstring(text.encode("utf-8"))
    assert root.find("f:Identity", NS).get("Publisher") == 'CN=A & "B", O=C'
    assert root.findtext("f:Properties/f:PublisherDisplayName", namespaces=NS) == "Me & Co"
    root = _manifest(publisher=SIGNPATH_FOUNDATION)
    assert root.find("f:Identity", NS).get("Publisher") == SIGNPATH_FOUNDATION


def test_manifest_capabilities():
    caps = _manifest().find("f:Capabilities", NS)
    assert caps is not None
    general = {c.get("Name") for c in caps.findall("f:Capability", NS)}
    assert general == {"internetClient", "privateNetworkClientServer"}
    assert [c.get("Name") for c in caps.findall("rescap:Capability", NS)] == ["runFullTrust"]
    assert {c.get("Name") for c in caps.findall("f:DeviceCapability", NS)} == {"webcam", "microphone"}
    # schema order: Capability (any namespace) before DeviceCapability
    tags = [c.tag.split("}")[1] for c in caps]
    assert tags.index("DeviceCapability") > max(i for i, t in enumerate(tags) if t == "Capability")


def test_manifest_full_trust_app_alias_and_startup_task_off():
    app = _manifest().find("f:Applications/f:Application", NS)
    assert app is not None
    assert app.get("Executable") == "ProjectAway.exe" and app.get("EntryPoint") == "Windows.FullTrustApplication"
    alias = app.find("f:Extensions/uap3:Extension/uap3:AppExecutionAlias/desktop:ExecutionAlias", NS)
    assert alias is not None and alias.get("Alias") == "away.exe"
    task = app.find("f:Extensions/desktop:Extension/desktop:StartupTask", NS)
    assert task is not None and task.get("Enabled") == "false"
    ext = app.find("f:Extensions/desktop:Extension", NS)
    assert ext is not None and ext.get("Category") == "windows.startupTask"


def test_manifest_references_only_generated_assets():
    text = bm.render_manifest("0.1.22")
    body = text[text.index("-->") :]
    refs = set(re.findall(r"Assets\\([A-Za-z0-9]+\.png)", body))
    assert refs == {f"{n}.png" for n in bm.LOGOS}
    assert refs <= set(bm.asset_files())


@pytest.mark.parametrize(
    ("name", "publisher"),
    [("ab", "CN=X"), ("has space", "CN=X"), ("x" * 51, "CN=X"), ("Good.Name", "O=missing CN"), ("Good.Name", "")],
)
def test_identity_validation(name, publisher):
    with pytest.raises(bm.BuildError):
        bm.render_manifest("0.1.22", identity_name=name, publisher=publisher)


def test_template_placeholders_all_filled(tmp_path):
    t = tmp_path / "t.xml"
    t.write_text("<!-- {{X}} -->\n<a v='{{VERSION}}' w='{{UNKNOWN}}'/>")
    with pytest.raises(bm.BuildError, match="UNKNOWN"):
        bm.render_manifest("1.0", template=t)


# ------------------------------------------------------------------------------------------------- assets
def test_asset_sizes():
    files = bm.asset_files()
    assert files["Square44x44Logo.scale-100.png"] == (44, 44)
    assert files["Square150x150Logo.scale-125.png"] == (188, 188)
    assert files["Wide310x150Logo.scale-200.png"] == (620, 300)
    assert files["StoreLogo.scale-400.png"] == (200, 200)
    assert files["SplashScreen.scale-100.png"] == (620, 300)
    assert files["LargeTile.scale-150.png"] == (465, 465)
    for t in (16, 24, 32, 48, 256):
        assert files[f"Square44x44Logo.targetsize-{t}.png"] == (t, t)
        assert files[f"Square44x44Logo.altform-unplated_targetsize-{t}.png"] == (t, t)
    for name in bm.LOGOS:
        assert f"{name}.png" in files  # the unqualified name the manifest uses


def test_make_assets_writes_every_png_at_its_size(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    written = bm.make_assets(tmp_path / "Assets")
    assert len(written) == len(bm.asset_files())
    for p in written:
        with Image.open(p) as im:
            assert im.size == bm.asset_files()[p.name], p.name
            assert im.mode == "RGBA"
            assert im.getpixel((im.width // 2, im.height // 2))[3] == 255  # the icon is centered, not blank
            assert im.getpixel((0, 0))[3] == 0  # transparent corners


# -------------------------------------------------------------------------------------------- build (fake)
def _fake_app(tmp_path) -> Path:
    app = tmp_path / "dist" / "ProjectAway"
    (app / "_internal" / "en").mkdir(parents=True)
    (app / "ProjectAway.exe").write_bytes(b"MZ")
    (app / "_internal" / "en" / "x.mo").write_bytes(b"x")
    return app


def test_build_runs_makepri_then_makeappx_for_both_packages(tmp_path):
    pytest.importorskip("PIL")
    app = _fake_app(tmp_path)
    calls: list[list[str]] = []

    def run(cmd):
        calls.append(cmd)
        if cmd[1] == "pack":  # pretend makeappx wrote the package
            Path(cmd[-1]).write_bytes(b"PK-msix")

    out = tmp_path / "out"
    produced = bm.build(
        "0.1.23",
        app,
        out,
        "FernandoGarza.ProjectAway",
        "CN=STORE-GUID",
        "Fernando Garza",
        "CN=Signer",
        run=run,
        tools={"makeappx": Path("makeappx.exe"), "makepri": Path("makepri.exe")},
    )
    assert [p.name for p in produced] == [
        "ProjectAway-0.1.23-windows-x64-store.msixupload",
        "ProjectAway-0.1.23-windows-x64.msix",
    ]
    assert [c[0:2] for c in calls] == [
        ["makepri.exe", "createconfig"],
        ["makepri.exe", "new"],
        ["makeappx.exe", "pack"],
        ["makeappx.exe", "pack"],
    ]
    for c in calls[2:]:
        assert c[c.index("/h") + 1] == "SHA256"  # SignPath needs the block map hash to match its signature
    with zipfile.ZipFile(produced[0]) as z:
        assert z.namelist() == ["ProjectAway-0.1.23-windows-x64-store.msix"]
    # the last manifest written is the sideload one: same name, the signer as publisher
    manifest = (out / "msix-work" / "stage" / "AppxManifest.xml").read_text(encoding="utf-8")
    assert 'Publisher="CN=Signer"' in manifest and 'Version="0.1.23.0"' in manifest
    mapping = (out / "msix-work" / "mapping.txt").read_text(encoding="utf-8").splitlines()
    assert mapping[0] == "[Files]"
    dests = [ln.rsplit('" "', 1)[1].rstrip('"') for ln in mapping[1:]]
    assert "ProjectAway.exe" in dests and "AppxManifest.xml" in dests and "resources.pri" not in dests  # (fake)
    assert "_internal\\en\\x.mo" in dests and "Assets\\Square44x44Logo.png" in dests
    assert not any(d.endswith("priconfig.xml") for d in dests)


def test_build_without_sideload_publisher_makes_only_the_store_upload(tmp_path, capsys):
    pytest.importorskip("PIL")
    produced = bm.build(
        "0.1.23",
        _fake_app(tmp_path),
        tmp_path / "out",
        bm.DEFAULT_IDENTITY_NAME,
        bm.UNCONFIGURED_PUBLISHER,
        "Fernando Garza",
        None,
        run=lambda cmd: None,
        tools={"makeappx": Path("makeappx.exe"), "makepri": Path("makepri.exe")},
    )
    assert [p.suffix for p in produced] == [".msixupload"]
    out = capsys.readouterr().out
    assert "MSIX_PUBLISHER is not set" in out and "skipping the sideload" in out


def test_mapping_refuses_missing_exe_and_clashing_files(tmp_path):
    stage = tmp_path / "stage"
    (stage / "Assets").mkdir(parents=True)
    (stage / "AppxManifest.xml").write_text("<x/>")
    with pytest.raises(bm.BuildError, match="PyInstaller"):
        bm.write_mapping(tmp_path / "nothing", stage, tmp_path / "m.txt")
    app = _fake_app(tmp_path)
    (app / "AppxManifest.xml").write_text("<old/>")
    with pytest.raises(bm.BuildError, match="AppxManifest"):
        bm.write_mapping(app, stage, tmp_path / "m.txt")


def test_find_sdk_tool_picks_the_newest_x64(tmp_path, monkeypatch):
    monkeypatch.setattr(bm.shutil, "which", lambda name: None)
    for ver in ("10.0.19041.0", "10.0.22621.0", "10.0.26100.0"):
        d = tmp_path / ver / "x64"
        d.mkdir(parents=True)
        if ver != "10.0.26100.0":  # the newest folder has no makeappx: skip it
            (d / "makeappx.exe").write_bytes(b"")
    (tmp_path / "10.0.99999.0" / "x86").mkdir(parents=True)
    (tmp_path / "10.0.99999.0" / "x86" / "makeappx.exe").write_bytes(b"")
    assert bm.find_sdk_tool("makeappx.exe", [tmp_path]) == tmp_path / "10.0.22621.0" / "x64" / "makeappx.exe"
    with pytest.raises(bm.BuildError, match="Windows 10/11 SDK"):
        bm.find_sdk_tool("makepri.exe", [tmp_path])


# ---------------------------------------------------------------------------------------------------- CLI
def test_cli_render_manifest(tmp_path, monkeypatch):
    for k in ("MSIX_IDENTITY_NAME", "MSIX_PUBLISHER", "MSIX_PUBLISHER_DISPLAY_NAME", "MSIX_SIDELOAD_PUBLISHER"):
        monkeypatch.delenv(k, raising=False)
    out = tmp_path / "m.xml"
    assert bm.main(["--render-manifest", str(out), "--version", "v2.3.4"]) == 0
    root = ET.fromstring(out.read_bytes())
    assert root.find("f:Identity", NS).get("Version") == "2.3.4.0"
    assert root.find("f:Identity", NS).get("Publisher") == bm.UNCONFIGURED_PUBLISHER


def test_cli_reads_identity_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MSIX_IDENTITY_NAME", "999Me.ProjectAway")
    monkeypatch.setenv("MSIX_PUBLISHER", "CN=STORE-GUID")
    monkeypatch.setenv("MSIX_PUBLISHER_DISPLAY_NAME", "Fernando G")
    monkeypatch.setenv("MSIX_SIDELOAD_PUBLISHER", SIGNPATH_FOUNDATION)
    args = bm.build_parser().parse_args([])
    assert (args.identity_name, args.publisher, args.publisher_display_name) == (
        "999Me.ProjectAway",
        "CN=STORE-GUID",
        "Fernando G",
    )
    assert args.sideload_publisher == SIGNPATH_FOUNDATION
    monkeypatch.setenv("MSIX_SIDELOAD_PUBLISHER", "")  # an unset GitHub variable arrives as ""
    assert bm.build_parser().parse_args([]).sideload_publisher is None


def test_cli_errors_are_one_line_and_exit_1(tmp_path, capsys):
    assert bm.main(["--render-manifest", str(tmp_path / "m.xml"), "--version", "main"]) == 1
    assert "not a version" in capsys.readouterr().err
    assert bm.main(["--input", str(tmp_path / "missing"), "--out", str(tmp_path), "--publisher", "nope"]) == 1
    assert "CN=" in capsys.readouterr().err


def test_cli_dry_run(tmp_path, capsys):
    pytest.importorskip("PIL")
    app = _fake_app(tmp_path)
    rc = bm.main(["--dry-run", "--input", str(app), "--out", str(tmp_path / "o"), "--sideload-publisher", "CN=S"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("would run: makeappx.exe pack") == 2 and "would run: makepri.exe new" in out


# --------------------------------------------------------------------------------------- release workflow
def _release() -> str:
    return (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")


def _windows_job(yml: str) -> str:
    return yml[yml.index("\n  windows:") : yml.index("\n  macos:")]


def test_release_builds_msix_and_attaches_it():
    win = _windows_job(_release())
    assert "python @a" in win and '"scripts/build_msix.py"' in win
    upload = win[win.rindex("name: windows-installer") :]
    for pattern in ("dist/*-setup.exe", "dist/*.msixupload", "dist/*-windows-x64.msix"):
        assert pattern in upload
    assert "MSIX_PUBLISHER: ${{ vars.MSIX_PUBLISHER }}" in win
    assert "--max-size-mb 1900 $pk" in win


def test_signpath_steps_are_gated_and_skip_without_secrets():
    yml = _release()
    data = yaml.safe_load(yml)
    job = data["jobs"]["windows"]
    gate = job["env"]["SIGNPATH_ENABLED"]
    for need in ("secrets.SIGNPATH_API_TOKEN", "secrets.SIGNPATH_ORGANIZATION_ID", "vars.SIGNPATH_PROJECT_SLUG",
                 "vars.SIGNPATH_POLICY_SLUG", "refs/tags/v"):  # fmt: skip
        assert need in gate
    signpath = [s for s in job["steps"] if "SignPath" in s.get("name", "")]
    assert len(signpath) == 6
    for s in signpath:
        assert s["if"] == "env.SIGNPATH_ENABLED == 'true'", s["name"]
    actions = [s for s in signpath if str(s.get("uses", "")).startswith("signpath/")]
    assert [a["uses"] for a in actions] == ["signpath/github-action-submit-signing-request@v3"] * 2
    for a in actions:
        w = a["with"]
        assert w["api-token"] == "${{ secrets.SIGNPATH_API_TOKEN }}"
        assert w["organization-id"] == "${{ secrets.SIGNPATH_ORGANIZATION_ID }}"
        assert w["wait-for-completion"] is True and w["output-artifact-directory"]
        assert "version:" in w["parameters"]
    assert job["permissions"]["actions"] == "read"


def test_signing_happens_before_packaging_and_before_the_release_upload():
    win = _windows_job(_release())
    exe_signed = win.index("SignPath: use the signed ProjectAway.exe")
    assert exe_signed < win.index("Inno Setup 6\\ISCC.exe") < win.index("scripts/build_msix.py")
    assert win.index("scripts/build_msix.py") < win.index("SignPath: sign the installer and MSIX")
    assert win.index("SignPath: use the signed installer and MSIX") < win.rindex("name: windows-installer")


def test_sha256sums_cover_only_the_published_signed_files():
    yml = _release()
    publish = yml.split("\n  publish:")[1]
    assert "needs: [python-dist, linux, windows, macos]" in publish
    assert 'pattern: "{python-dist,linux-installers,windows-installer,macos-dmg}"' in publish  # no signpath-unsigned-*
    assert publish.index("pattern:") < publish.index("sha256sum") < publish.index("softprops/action-gh-release")
    names = re.findall(r"name: (signpath-unsigned-[a-z]+)", yml)
    assert names == ["signpath-unsigned-app", "signpath-unsigned-release"]


def test_signpath_artifact_configurations_match_the_file_names():
    app = (ROOT / "packaging" / "signpath" / "app-exe.xml").read_text()
    rel = (ROOT / "packaging" / "signpath" / "release.xml").read_text()
    for x in (app, rel):
        ET.fromstring(x.encode())
        assert 'product-name="Project Away"' in x and 'product-version="${version}"' in x
    assert '<pe-file path="ProjectAway.exe"' in app
    names = bm.package_names("${version}")
    assert f'path="{names["sideload_msix"]}"' in rel
    assert 'path="ProjectAway-${version}-windows-x64-setup.exe"' in rel
    iss = (ROOT / "packaging" / "windows" / "away.iss").read_text()
    assert "OutputBaseFilename=ProjectAway-{#AppVersion}-windows-x64-setup" in iss


def test_spec_sets_the_windows_version_resource_and_icon():
    spec = (ROOT / "packaging" / "pyinstaller" / "away.spec").read_text()
    assert "icon=_win_icon" in spec and "version=_win_version" in spec
    assert 'vi.StringStruct("ProductName", "Project Away")' in spec
    assert 'vi.StringStruct("ProductVersion", raw)' in spec
    assert 'if sys.platform != "win32":\n        return None, None' in spec


def test_ci_builds_and_installs_a_test_msix():
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    job = ci["jobs"]["msix"]
    assert job["runs-on"] == "windows-latest"
    script = "\n".join(s.get("run", "") for s in job["steps"])
    assert "ci_standin.py" in script and "build_msix.py" in script
    assert "Add-AppxPackage" in script and "Remove-AppxPackage" in script and "away.exe" in script


# ---------------------------------------------------------------------------------------------- docs
def test_privacy_and_signing_policy_are_public_without_publishing_the_playbook():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[Privacy policy](PRIVACY.md)" in readme
    assert "Code signing policy" in readme
    assert "Free code signing provided by [SignPath.io](https://about.signpath.io)" in readme
    assert "certificate by [SignPath Foundation](https://signpath.org)" in readme
    assert "docs/SIGNING-AND-STORE.md" not in readme
    assert not (ROOT / "docs" / "SIGNING-AND-STORE.md").exists()
    privacy = (ROOT / "PRIVACY.md").read_text(encoding="utf-8")
    for must in ("camera", "microphone", "Agent Link", "telemetry", "updates.check"):
        assert must in privacy


def test_winget_manifests_are_valid_yaml_with_the_installer_name():
    d = ROOT / "packaging" / "winget"
    files = sorted(d.glob("*.yaml"))
    assert len(files) == 3
    for f in files:
        data = yaml.safe_load(f.read_text().replace("{{VERSION}}", "0.1.23").replace("{{SHA256}}", "A" * 64))
        assert data["PackageIdentifier"] == "FernandoGarza.ProjectAway"
        assert data["PackageVersion"] == "0.1.23"
    inst = yaml.safe_load((d / "FernandoGarza.ProjectAway.installer.yaml").read_text())
    url = inst["Installers"][0]["InstallerUrl"]
    assert url.endswith("ProjectAway-{{VERSION}}-windows-x64-setup.exe")


def test_release_msix_step_never_blocks_the_normal_release():
    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    step = text.split("- name: MSIX packages", 1)[1].split("- name:", 1)[0]
    assert "continue-on-error: true" in step
    # signing switches itself on only with every SignPath secret/variable present, and only on version tags
    assert "SIGNPATH_ENABLED: ${{ startsWith(github.ref, 'refs/tags/v') && secrets.SIGNPATH_API_TOKEN != ''" in text
