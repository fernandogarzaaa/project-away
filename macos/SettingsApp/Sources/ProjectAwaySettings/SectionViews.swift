// v0.1.35: the sidebar and the pages. Every page is a grouped Form; the bridge's groups become its sections,
// the "Fine-tuning" groups (Settings > Tuning sliders) fold away under a disclosure.
import SwiftUI

enum Palette {
    static func color(_ id: String) -> Color {
        switch id {
        case "general": return .gray
        case "pointer": return .blue
        case "voice": return .pink
        case "face": return .orange
        case "cameras": return .green
        case "models": return .purple
        case "advanced": return .indigo
        case "permissions": return .teal
        default: return .gray
        }
    }

    static let subtitles: [String: String] = [
        "general": "Start-up, comfort, pausing and updates.",
        "pointer": "How your hand moves, clicks and scrolls.",
        "voice": "Dictation, voice commands, hotkeys and claps.",
        "face": "Face triggers and looking at the screen.",
        "cameras": "The cameras Project Away watches.",
        "models": "Which model does each job, on this Mac.",
        "advanced": "Expert numbers and the classic Settings pages.",
        "permissions": "What macOS lets Project Away do.",
        "about": "Version, files and help.",
    ]
}

/// The coloured rounded-square icon of a sidebar row (like System Settings).
struct SectionIcon: View {
    let id: String
    let symbol: String
    var size: CGFloat = 22

    var body: some View {
        let c = Palette.color(id)
        RoundedRectangle(cornerRadius: size * 0.27, style: .continuous)
            .fill(LinearGradient(colors: [c.opacity(0.85), c], startPoint: .top, endPoint: .bottom))
            .frame(width: size, height: size)
            .overlay(
                Image(systemName: symbol)
                    .font(.system(size: size * 0.55, weight: .semibold))
                    .foregroundStyle(.white)
            )
    }
}

struct Sidebar: View {
    @EnvironmentObject var store: SettingsStore

    var body: some View {
        List(selection: $store.selection) {
            ForEach(store.sections) { s in
                NavigationLink(value: s.id) {
                    Label {
                        Text(s.title)
                    } icon: {
                        SectionIcon(id: s.id, symbol: s.symbol)
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .navigationSplitViewColumnWidth(min: 200, ideal: 220, max: 280)
    }
}

/// The page header: a large symbol in a glass circle, the title and one line about the page.
struct Hero: View {
    let section: SettingsSection

    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: section.symbol)
                .font(.system(size: 28, weight: .medium))
                .foregroundStyle(Palette.color(section.id))
                .frame(width: 60, height: 60)
                .glassCircle(tint: Palette.color(section.id).opacity(0.25))
            VStack(alignment: .leading, spacing: 3) {
                Text(section.title).font(.title2.weight(.semibold))
                Text(Palette.subtitles[section.id] ?? "").foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.vertical, 6)
    }
}

struct SectionPage: View {
    @EnvironmentObject var store: SettingsStore
    let section: SettingsSection

    var body: some View {
        Form {
            Section { Hero(section: section) }
                .listRowBackground(Color.clear)
            switch section.id {
            case "general": GeneralActions()
            case "cameras": CamerasList()
            case "models": ModelsList()
            case "permissions": PermissionsList()
            case "about": AboutPage()
            default: EmptyView()
            }
            ForEach(section.groups.filter { !$0.tuning }) { g in
                GroupSection(group: g)
            }
            let tuning = section.groups.filter(\.tuning)
            if !tuning.isEmpty {
                Section {
                    ForEach(tuning) { g in TuningDisclosure(group: g) }
                } header: {
                    Text("Fine-tuning")
                } footer: {
                    FooterText("The same sliders as the Tuning tab. Changes save and apply at once.")
                }
            }
            if section.id == "advanced" { ClassicPages() }
        }
        .formStyle(.grouped)
        .navigationTitle(section.title)
    }
}

struct GroupSection: View {
    let group: FieldGroup

    var body: some View {
        Section {
            ForEach(group.fields) { FieldRow(field: $0) }
        } header: {
            Text(group.title)
        } footer: {
            if !group.footer.isEmpty { FooterText(group.footer) }
        }
    }
}

struct TuningDisclosure: View {
    let group: FieldGroup
    @State private var open = false

    var body: some View {
        DisclosureGroup(isExpanded: $open) {
            ForEach(group.fields) { FieldRow(field: $0).padding(.vertical, 2) }
        } label: {
            Button {
                withAnimation(.easeInOut(duration: 0.2)) { open.toggle() }
            } label: {
                HStack {
                    Text(group.title)
                    Spacer()
                    Text("\(group.fields.count)").foregroundStyle(.secondary).monospacedDigit()
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
    }
}

struct SearchResults: View {
    @EnvironmentObject var store: SettingsStore

    var body: some View {
        let found = store.matches()
        Form {
            if found.isEmpty {
                Section {
                    Text("No settings match “\(store.search)”.").foregroundStyle(.secondary)
                }
            }
            ForEach(store.sections) { s in
                let rows = found.filter { $0.0.id == s.id }
                if !rows.isEmpty {
                    Section {
                        ForEach(rows, id: \.2.key) { row in FieldRow(field: row.2) }
                    } header: {
                        Label { Text(s.title) } icon: { SectionIcon(id: s.id, symbol: s.symbol, size: 16) }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Search")
    }
}

// MARK: - Hand-made parts of some pages

struct GeneralActions: View {
    @EnvironmentObject var store: SettingsStore

    var body: some View {
        Section {
            GlassGroup(spacing: 10) {
                HStack(spacing: 10) {
                    Button { store.run("first_run") } label: {
                        Label("Guided First Run", systemImage: "sparkles").frame(maxWidth: .infinity)
                    }
                    .glassButton(prominent: true)
                    Button { store.run("calibrate") } label: {
                        Label("Calibrate Tracking", systemImage: "scope").frame(maxWidth: .infinity)
                    }
                    .glassButton()
                    Button { store.run("check_pc") } label: {
                        Label("Check My Mac", systemImage: "stethoscope").frame(maxWidth: .infinity)
                    }
                    .glassButton()
                }
                .controlSize(.large)
            }
        } header: {
            Text("Get started")
        } footer: {
            FooterText("The guided first run sets everything up in about five minutes; Check My Mac tests cameras, voice and pointer.")
        }
    }
}

struct CamerasList: View {
    @EnvironmentObject var store: SettingsStore

    var body: some View {
        Section {
            ForEach(Array((store.document?.cameras ?? []).enumerated()), id: \.offset) { _, cam in
                HStack(spacing: 12) {
                    Image(systemName: cam.role == "primary" ? "web.camera.fill" : "web.camera")
                        .font(.title3)
                        .foregroundStyle(Palette.color("cameras"))
                        .frame(width: 28)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(cameraName(cam.device))
                        Text("\(cam.width)×\(cam.height) at \(cam.fps) fps\(cam.mirror ? " · mirrored" : "")")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Text(cam.role.capitalized)
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .glassCapsule(tint: cam.role == "primary" ? Palette.color("cameras").opacity(0.3) : nil)
                }
            }
            HStack {
                Spacer()
                Button("Calibrate Tracking…") { store.run("calibrate") }.glassButton()
                Button("Manage Cameras…") { store.openClassic(tab: "Cameras") }.glassButton()
            }
        } header: {
            Text((store.document?.cameras.count ?? 0) > 1 ? "Cameras (fused)" : "Camera")
        } footer: {
            FooterText("Add, test and calibrate cameras in Manage Cameras (the classic Cameras page).")
        }
    }

    private func cameraName(_ v: JSONValue) -> String {
        if case .number(let n) = v { return "Camera \(Int(n))" }
        return v.string
    }
}

struct ModelsList: View {
    @EnvironmentObject var store: SettingsStore

    var body: some View {
        Section {
            ForEach(store.document?.models ?? []) { m in
                LabeledContent {
                    HStack(spacing: 8) {
                        Text(m.optional && !m.on ? "Off" : m.current).foregroundStyle(.secondary)
                        if !m.runs_on.isEmpty && (m.on || !m.optional) {
                            Text(m.runs_on)
                                .font(.caption.weight(.medium))
                                .padding(.horizontal, 7).padding(.vertical, 2)
                                .glassCapsule()
                        }
                    }
                } label: {
                    Text(m.title)
                }
            }
            HStack {
                Spacer()
                Button("Change Models…") { store.openClassic(tab: "Models") }.glassButton()
            }
        } header: {
            Text("Models in use")
        }
    }
}

struct PermissionsList: View {
    @EnvironmentObject var store: SettingsStore

    var body: some View {
        Section {
            ForEach(store.permissions) { p in
                HStack(spacing: 12) {
                    Image(systemName: icon(p.id))
                        .font(.title3)
                        .foregroundStyle(.tint)
                        .frame(width: 28)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(p.title)
                        Text(p.help).font(.callout).foregroundStyle(.secondary)
                    }
                    Spacer()
                    StateBadge(granted: p.granted)
                    Button("Open…") { store.open(url: p.url) }
                        .glassButton(prominent: p.granted == false)
                        .help("Open System Settings > Privacy & Security > \(p.title)")
                }
                .padding(.vertical, 3)
            }
        } header: {
            Text("Privacy & Security")
        } footer: {
            FooterText("After you allow Accessibility or Input Monitoring, quit Project Away from the menu bar and open it again.")
        }
        Section {
            HStack {
                Spacer()
                Button {
                    store.refreshPermissions()
                    store.showToast("Checked again")
                } label: {
                    Label("Check Again", systemImage: "arrow.clockwise")
                }
                .glassButton()
            }
        }
    }

    private func icon(_ id: String) -> String {
        switch id {
        case "accessibility": return "accessibility"
        case "input_monitoring": return "keyboard"
        case "camera": return "camera"
        case "microphone": return "mic"
        default: return "lock"
        }
    }
}

struct StateBadge: View {
    let granted: Bool?

    var body: some View {
        let (text, symbol, color): (String, String, Color) = {
            switch granted {
            case .some(true): return ("Allowed", "checkmark.circle.fill", .green)
            case .some(false): return ("Not allowed", "xmark.circle.fill", .orange)
            case .none: return ("Not asked yet", "questionmark.circle", .secondary)
            }
        }()
        Label(text, systemImage: symbol)
            .font(.callout.weight(.medium))
            .foregroundStyle(color)
            .labelStyle(.titleAndIcon)
    }
}

struct ClassicPages: View {
    @EnvironmentObject var store: SettingsStore

    var body: some View {
        Section {
            ForEach(store.document?.tk_tabs ?? [], id: \.self) { tab in
                Button {
                    store.openClassic(tab: tab)
                } label: {
                    HStack {
                        Image(systemName: symbol(tab)).frame(width: 22).foregroundStyle(.tint)
                        Text(tab).foregroundStyle(.primary)
                        Spacer()
                        Image(systemName: "arrow.up.forward.app").foregroundStyle(.secondary)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        } header: {
            Text("More settings")
        } footer: {
            FooterText("These pages open in the classic Settings window for now.")
        }
    }

    private func symbol(_ tab: String) -> String {
        switch tab {
        case "Bindings": return "hand.raised"
        case "Gesture Studio": return "hand.wave"
        case "Shortcuts": return "command"
        case "Agents": return "person.wave.2"
        case "Tuning": return "dial.medium"
        case "Performance": return "gauge.with.dots.needle.67percent"
        default: return "square.grid.2x2"
        }
    }
}

struct AboutPage: View {
    @EnvironmentObject var store: SettingsStore

    var body: some View {
        Section {
            LabeledContent("Version", value: store.document?.version ?? "–")
            LabeledContent("Settings file") {
                Text(store.document?.config_path ?? "–")
                    .font(.callout.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }
            HStack {
                Spacer()
                Button("Show in Finder") { store.revealConfig() }.glassButton()
                Button("Classic Settings Window…") { store.openClassic() }.glassButton()
            }
        } header: {
            Text("Project Away")
        } footer: {
            FooterText("Everything runs on this Mac. Changes are saved as you make them and Project Away picks them up at once.")
        }
        Section {
            Link(destination: URL(string: "https://github.com/fernandogarzaaa/project-away")!) {
                Label("Project Away on GitHub", systemImage: "link")
            }
            Link(destination: URL(string: "https://github.com/fernandogarzaaa/project-away/blob/main/PRIVACY.md")!) {
                Label("Privacy", systemImage: "hand.raised.square")
            }
        }
    }
}

/// Section footers read left to right, like System Settings.
struct FooterText: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        Text(text)
            .font(.callout)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .fixedSize(horizontal: false, vertical: true)
    }
}
