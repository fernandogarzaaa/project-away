// v0.1.35: Project Away Settings, the native macOS Settings window.
//
//   ProjectAwaySettings --section voice --bridge '["/…/ProjectAway", "settings-bridge", "serve"]'
//
// One window at a time: a second launch posts its section to the running copy (a distributed notification),
// brings it forward and quits. Project Away starts this app from `away settings` (away.app.settings_native).
import AppKit
import SwiftUI

let showNotification = Notification.Name("com.fernandogarzaaa.projectaway.settings.show")

func argument(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}

let requestedSection = argument("--section") ?? argument("--tab")

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var store: SettingsStore?

    func applicationWillFinishLaunching(_ notification: Notification) {
        signal(SIGPIPE, SIG_IGN)  // a bridge that died must not take the window down with it
        let me = NSRunningApplication.current
        if let id = Bundle.main.bundleIdentifier,
           let other = NSRunningApplication.runningApplications(withBundleIdentifier: id)
            .first(where: { $0.processIdentifier != me.processIdentifier }) {
            DistributedNotificationCenter.default().postNotificationName(
                showNotification, object: requestedSection ?? "", userInfo: nil, deliverImmediately: true)
            if #available(macOS 14.0, *) {
                NSApp.yieldActivation(to: other)
                other.activate(from: me, options: [])
            } else {
                other.activate(options: [.activateIgnoringOtherApps])
            }
            exit(0)
        }
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        if let a = argument("--appearance") {  // screenshots / testing: light or dark regardless of the system
            NSApp.appearance = NSAppearance(named: a == "dark" ? .darkAqua : .aqua)
        }
        bringForward()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) { self.bringForward() }
        DistributedNotificationCenter.default().addObserver(
            forName: showNotification, object: nil, queue: .main
        ) { [weak self] note in
            let section = note.object as? String
            Task { @MainActor in
                if let s = section, !s.isEmpty, let store = self?.store {
                    store.search = ""
                    store.selection = SettingsStore.sectionID(for: s, sections: store.sections)
                }
                self?.bringForward()
            }
        }
    }

    /// Started by the menu-bar app (in the background), so ask for the front explicitly.
    func bringForward() {
        NSApp.activate(ignoringOtherApps: true)
        for w in NSApp.windows where w.canBecomeMain {
            if w.isMiniaturized { w.deminiaturize(nil) }
            if let size = argument("--size")?.split(separator: "x").compactMap({ Double($0) }), size.count == 2 {
                w.setFrame(NSRect(x: w.frame.minX, y: 60, width: size[0], height: size[1]), display: true)  // testing
            }
            w.makeKeyAndOrderFront(nil)
            w.orderFrontRegardless()
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        store?.shutdown()
    }
}

@main
struct ProjectAwaySettingsApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var store = SettingsStore(section: requestedSection)

    var body: some Scene {
        Window("Project Away Settings", id: "settings") {
            ContentView()
                .environmentObject(store)
                .onAppear { delegate.store = store }
                .task { await store.load() }
                .frame(minWidth: 760, minHeight: 520)
        }
        .defaultSize(width: 900, height: 680)
        .windowToolbarStyle(.unified)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandGroup(after: .appInfo) {
                Button("Classic Settings Window…") { store.openClassic() }
                    .keyboardShortcut(",", modifiers: [.command, .shift])
            }
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var store: SettingsStore
    @Environment(\.scenePhase) private var phase

    var body: some View {
        NavigationSplitView {
            Sidebar()
        } detail: {
            detail
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .overlay(alignment: .bottom) { ToastView() }
        }
        .searchable(text: $store.search, placement: .sidebar, prompt: "Search")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    store.openClassic()
                } label: {
                    Label("More Settings", systemImage: "square.grid.2x2")
                }
                .help("Open the classic Settings window (Bindings, Gesture Studio, Shortcuts, Agents…)")
            }
        }
        .onChange(of: phase) { p in if p == .active { store.refreshPermissions() } }
        .onChange(of: store.selection) { _ in store.search = "" }  // picking a page ends the search
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            store.refreshPermissions()
        }
    }

    @ViewBuilder
    private var detail: some View {
        if let err = store.loadError, store.document == nil {
            LoadFailed(message: err)
        } else if store.document == nil {
            ProgressView("Loading settings…").frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if !store.search.trimmingCharacters(in: .whitespaces).isEmpty {
            SearchResults()
        } else if let s = store.section(store.selection) {
            SectionPage(section: s).id(s.id)
        } else {
            Text("Select a section").foregroundStyle(.secondary)
        }
    }
}

struct ToastView: View {
    @EnvironmentObject var store: SettingsStore

    var body: some View {
        ZStack {
            if let t = store.toast {
                Label(t.text, systemImage: t.isError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                    .font(.callout.weight(.medium))
                    .foregroundStyle(t.isError ? Color.orange : Color.primary)
                    .lineLimit(2)
                    .padding(.horizontal, 16).padding(.vertical, 9)
                    .glassCapsule(tint: t.isError ? Color.orange.opacity(0.15) : nil)
                    .frame(maxWidth: 560)
                    .transition(.opacity.combined(with: .scale(scale: 0.9)))
            }
        }
        .padding(.bottom, 18)
        .allowsHitTesting(false)
    }
}

struct LoadFailed: View {
    @EnvironmentObject var store: SettingsStore
    let message: String

    var body: some View {
        VStack(spacing: 14) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 36))
                .foregroundStyle(.orange)
                .frame(width: 72, height: 72)
                .glassCircle(tint: .orange.opacity(0.2))
            Text("Settings could not be loaded").font(.title3.weight(.semibold))
            Text(message).foregroundStyle(.secondary).multilineTextAlignment(.center).frame(maxWidth: 420)
            Button("Try Again") { Task { await store.load() } }.glassButton(prominent: true)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
