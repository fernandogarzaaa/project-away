// v0.1.35: Liquid Glass where the SDK and the OS have it (Xcode 26+ / macOS 26+), standard materials before.
// `#if compiler(>=6.2)` keeps the file compiling with an older Xcode (whose SDK has no Glass type at all);
// `#available` picks the look at run time on an older macOS.
import SwiftUI

extension View {
    /// A glass panel (rounded rectangle). Before macOS 26: the regular material with a hairline.
    @ViewBuilder
    func glassPanel(cornerRadius: CGFloat = 18, tint: Color? = nil) -> some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            self.glassEffect(.regular.tint(tint), in: shape)
        } else {
            self.background(.regularMaterial, in: shape)
                .overlay(shape.strokeBorder(.primary.opacity(0.08)))
        }
        #else
        self.background(.regularMaterial, in: shape)
            .overlay(shape.strokeBorder(.primary.opacity(0.08)))
        #endif
    }

    /// A glass capsule (toasts, badges).
    @ViewBuilder
    func glassCapsule(tint: Color? = nil) -> some View {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            self.glassEffect(.regular.tint(tint), in: Capsule())
        } else {
            self.background(.regularMaterial, in: Capsule())
        }
        #else
        self.background(.regularMaterial, in: Capsule())
        #endif
    }

    /// A glass circle behind an SF Symbol.
    @ViewBuilder
    func glassCircle(tint: Color? = nil) -> some View {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            self.glassEffect(.regular.tint(tint), in: Circle())
        } else {
            self.background((tint ?? .accentColor).opacity(0.18), in: Circle())
        }
        #else
        self.background((tint ?? .accentColor).opacity(0.18), in: Circle())
        #endif
    }

    /// `.glass` / `.glassProminent` buttons on macOS 26+, bordered / borderedProminent before.
    @ViewBuilder
    func glassButton(prominent: Bool = false) -> some View {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            if prominent { self.buttonStyle(.glassProminent) } else { self.buttonStyle(.glass) }
        } else {
            if prominent { self.buttonStyle(.borderedProminent) } else { self.buttonStyle(.bordered) }
        }
        #else
        if prominent { self.buttonStyle(.borderedProminent) } else { self.buttonStyle(.bordered) }
        #endif
    }
}

/// Groups glass shapes so neighbours blend into one another (macOS 26+); a plain container before.
struct GlassGroup<Content: View>: View {
    var spacing: CGFloat = 12
    @ViewBuilder var content: () -> Content

    var body: some View {
        #if compiler(>=6.2)
        if #available(macOS 26.0, *) {
            GlassEffectContainer(spacing: spacing) { content() }
        } else {
            content()
        }
        #else
        content()
        #endif
    }
}
