// v0.1.35: the window's state. Values change optimistically, are sent to the bridge (which validates, saves and
// returns the fresh document) and snap back with the bridge's message when Python rejects them.
import AppKit
import AVFoundation
import ApplicationServices
import Foundation
import IOKit.hid
import SwiftUI

@MainActor
final class SettingsStore: ObservableObject {
    @Published var document: SettingsDocument?
    @Published var values: [String: JSONValue] = [:]
    @Published var errors: [String: String] = [:]
    @Published var loadError: String?
    @Published var toast: Toast?
    @Published var permissions: [PermissionRow] = []
    @Published var selection: String? = "general"
    @Published var search = ""

    struct Toast: Equatable {
        let id = UUID()
        let text: String
        let isError: Bool
    }

    private let bridge: Bridge?
    private var toastTask: Task<Void, Never>?

    init(section: String?) {
        if let argv = Bridge.resolveArgv() { bridge = Bridge(argv: argv) } else { bridge = nil }
        if let section { selection = SettingsStore.sectionID(for: section) }
    }

    /// `--section voice` / `--tab "Voice & input"` -> a sidebar id (unknown -> General).
    static func sectionID(for name: String, sections: [SettingsSection]? = nil) -> String {
        let want = name.lowercased().trimmingCharacters(in: .whitespaces)
        let known: [(String, String, String)] = (sections ?? []).map { ($0.id, $0.title.lowercased(), $0.tk.lowercased()) }
        let fallback: [(String, String, String)] = [
            ("general", "general", "getting started"), ("pointer", "hands & pointer", ""),
            ("voice", "voice & dictation", "voice & input"), ("face", "face & eyes", ""),
            ("cameras", "cameras", "cameras"), ("models", "models", "models"), ("advanced", "advanced", ""),
            ("permissions", "permissions", ""), ("about", "about", ""),
        ]
        for (id, title, tk) in known.isEmpty ? fallback : known
        where want == id || want == title || (!tk.isEmpty && want == tk) || (!want.isEmpty && title.contains(want)) {
            return id
        }
        return "general"
    }

    var sections: [SettingsSection] { document?.sections ?? [] }

    func section(_ id: String?) -> SettingsSection? { sections.first { $0.id == id } }

    // MARK: loading and saving

    func load() async {
        guard let bridge else {
            loadError = BridgeError.notFound.errorDescription
            return
        }
        do {
            let r = try await bridge.request(["op": "get"])
            if let doc = r.document { apply(doc) } else {
                loadError = r.errors?.values.joined(separator: "\n") ?? "Project Away could not read its settings."
            }
        } catch {
            loadError = error.localizedDescription
        }
        refreshPermissions()
    }

    private func apply(_ doc: SettingsDocument) {
        document = doc
        loadError = nil
        var v: [String: JSONValue] = [:]
        for s in doc.sections { for g in s.groups { for f in g.fields { v[f.key] = f.value } } }
        values = v
        mergePermissions(doc.permissions)
    }

    func value(_ key: String) -> JSONValue { values[key] ?? .null }

    func set(_ key: String, _ value: JSONValue) {
        guard values[key] != value else { return }
        values[key] = value  // optimistic; the bridge's document wins below
        errors[key] = nil
        guard let bridge else { return }
        Task {
            do {
                let r = try await bridge.request(["op": "set", "values": [key: value.any]])
                if let doc = r.document { apply(doc) }
                if r.ok {
                    showToast("Saved")
                } else {
                    let msg = r.errors?[key] ?? r.errors?.values.first ?? "Not saved"
                    errors[key] = msg
                    showToast(msg, error: true)
                }
            } catch {
                showToast(error.localizedDescription, error: true)
            }
        }
    }

    func showToast(_ text: String, error: Bool = false) {
        toastTask?.cancel()
        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) { toast = Toast(text: text, isError: error) }
        toastTask = Task {
            try? await Task.sleep(nanoseconds: error ? 4_000_000_000 : 1_600_000_000)
            if !Task.isCancelled { withAnimation(.easeOut(duration: 0.3)) { toast = nil } }
        }
    }

    // MARK: search

    /// Fields whose label, help or key match the search text, with the section they live in.
    func matches() -> [(SettingsSection, FieldGroup, Field)] {
        let q = search.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return [] }
        var out: [(SettingsSection, FieldGroup, Field)] = []
        for s in sections {
            for g in s.groups {
                for f in g.fields where f.label.lowercased().contains(q) || f.help.lowercased().contains(q)
                    || f.key.lowercased().contains(q) || g.title.lowercased().contains(q) {
                    out.append((s, g, f))
                }
            }
        }
        return out
    }

    // MARK: permissions (read natively; this process counts as Project Away when Project Away started it)

    func refreshPermissions() {
        mergePermissions(document?.permissions ?? permissions)
    }

    private func mergePermissions(_ rows: [PermissionRow]) {
        let base = rows.isEmpty ? SettingsStore.defaultPermissions : rows
        permissions = base.map { p in
            PermissionRow(id: p.id, title: p.title, help: p.help, granted: SettingsStore.nativeState(p.id) ?? p.granted,
                          url: p.url)
        }
    }

    static func nativeState(_ id: String) -> Bool? {
        switch id {
        case "accessibility": return AXIsProcessTrusted()
        case "input_monitoring":
            let s = IOHIDCheckAccess(kIOHIDRequestTypeListenEvent)
            return s == kIOHIDAccessTypeGranted
        case "camera": return avState(.video)
        case "microphone": return avState(.audio)
        default: return nil
        }
    }

    private static func avState(_ media: AVMediaType) -> Bool? {
        switch AVCaptureDevice.authorizationStatus(for: media) {
        case .authorized: return true
        case .denied, .restricted: return false
        default: return nil  // not asked yet: Project Away asks the first time it needs it
        }
    }

    static let defaultPermissions: [PermissionRow] = [
        PermissionRow(id: "accessibility", title: "Accessibility", help: "Moves the pointer, clicks and types for you.",
                      granted: nil, url: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"),
        PermissionRow(id: "input_monitoring", title: "Input Monitoring", help: "Lets the hotkeys work.", granted: nil,
                      url: "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"),
        PermissionRow(id: "camera", title: "Camera", help: "Tracks your hands and face, on this Mac only.", granted: nil,
                      url: "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"),
        PermissionRow(id: "microphone", title: "Microphone", help: "Hears dictation and voice commands, on this Mac only.",
                      granted: nil, url: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"),
    ]

    // MARK: other windows (their own processes, like the Tk tabs start them)

    func run(_ command: String, extra: [String] = []) {
        guard let argv = document?.commands[command], let exe = argv.first else {
            showToast("Not available in this build", error: true)
            return
        }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = Array(argv.dropFirst()) + extra
        p.standardInput = FileHandle.nullDevice
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        do { try p.run() } catch { showToast("Could not open it: \(error.localizedDescription)", error: true) }
    }

    func openClassic(tab: String? = nil) {
        run("tk_settings", extra: tab.map { ["--tab", $0] } ?? [])
        showToast("Opening \(tab ?? "more settings")…")
    }

    func open(url: String) {
        if let u = URL(string: url) { NSWorkspace.shared.open(u) }
    }

    func revealConfig() {
        guard let path = document?.config_path else { return }
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }

    func shutdown() { bridge?.stop() }
}
