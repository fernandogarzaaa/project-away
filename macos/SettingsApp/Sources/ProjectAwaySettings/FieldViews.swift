// v0.1.35: one control per bridge field type (bool / choice / text / hotkey / number). A field the bridge adds
// shows up here without a Swift change.
import SwiftUI

struct FieldRow: View {
    @EnvironmentObject var store: SettingsStore
    let field: Field

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            control
            if let err = store.errors[field.key] {
                Label(err, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.orange)
                    .transition(.opacity)
            }
        }
        .help(field.key)
    }

    @ViewBuilder
    private var control: some View {
        switch field.type {
        case "bool":
            Toggle(isOn: Binding(get: { store.value(field.key).bool }, set: { store.set(field.key, .bool($0)) })) {
                TitleAndHelp(title: field.label, help: field.help)
            }
            .toggleStyle(.switch)
        case "choice":
            Picker(selection: Binding(get: { store.value(field.key) }, set: { store.set(field.key, $0) })) {
                ForEach(field.choices ?? [], id: \.self) { c in Text(c.label).tag(c.value) }
            } label: {
                TitleAndHelp(title: field.label, help: field.help)
            }
        case "number":
            NumberRow(field: field)
        default:
            TextRow(field: field)
        }
    }
}

struct TitleAndHelp: View {
    let title: String
    let help: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
            if !help.isEmpty {
                Text(help)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// Text and hotkey fields: saved on Return or when the field loses focus.
struct TextRow: View {
    @EnvironmentObject var store: SettingsStore
    let field: Field
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        LabeledContent {
            TextField(text: $text, prompt: Text(field.type == "hotkey" ? "off" : "")) { EmptyView() }
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
                .font(field.type == "hotkey" ? .body.monospaced() : .body)
                .multilineTextAlignment(.trailing)
                .frame(maxWidth: 220)
                .focused($focused)
                .onSubmit(commit)
                .onChange(of: focused) { f in if !f { commit() } }
        } label: {
            TitleAndHelp(title: field.label, help: field.help)
        }
        .onAppear { text = store.value(field.key).string }
        .onChange(of: store.value(field.key)) { v in if !focused { text = v.string } }
    }

    private func commit() {
        let t = text.trimmingCharacters(in: .whitespaces)
        if t != store.value(field.key).string { store.set(field.key, .string(t)) }
    }
}

/// Sliders: the value is sent when the drag ends (one save per gesture, not per pixel).
struct NumberRow: View {
    @EnvironmentObject var store: SettingsStore
    let field: Field
    @State private var draft: Double?

    private var lo: Double { field.min ?? 0 }
    private var hi: Double { Swift.max(field.max ?? 1, lo + 1e-9) }
    private var current: Double { draft ?? store.value(field.key).double }

    var body: some View {
        LabeledContent {
            HStack(spacing: 8) {
                Slider(
                    value: Binding(get: { current }, set: { draft = snap($0) }),
                    in: lo...hi,
                    onEditingChanged: { editing in
                        if !editing, let d = draft {
                            store.set(field.key, .number(d))
                            draft = nil
                        }
                    }
                )
                .labelsHidden()
                .controlSize(.small)
                .frame(width: 180)
                Text(format(current))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .frame(minWidth: 64, alignment: .trailing)
                Button {
                    if let d = field.default { store.set(field.key, .number(d)) }
                } label: {
                    Image(systemName: "arrow.counterclockwise")
                }
                .buttonStyle(.borderless)
                .help(field.default.map { "Back to the default (\(format($0)))" } ?? "")
                .opacity(isDefault ? 0 : 1)
                .disabled(isDefault)
            }
        } label: {
            TitleAndHelp(title: field.label, help: field.help)
        }
    }

    private var isDefault: Bool {
        guard let d = field.default else { return true }
        return abs(d - current) <= (field.step ?? 0) / 2 + 1e-12
    }

    private func snap(_ v: Double) -> Double {
        let step = field.step ?? 0
        guard step > 0 else { return v }
        let s = (v / step).rounded() * step
        return field.is_int == true ? s.rounded() : (s * 1e9).rounded() / 1e9
    }

    private func format(_ v: Double) -> String {
        let unit = field.unit.map { $0.isEmpty ? "" : ($0 == "×" || $0 == "%" ? $0 : " \($0)") } ?? ""
        if field.is_int == true { return "\(Int(v.rounded()))\(unit)" }
        let step = field.step ?? 0.01
        let digits = step >= 1 ? 0 : Swift.min(4, Int((-log10(step)).rounded(.up)))
        return String(format: "%.\(digits)f", v) + unit
    }
}
