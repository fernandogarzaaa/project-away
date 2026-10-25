// swift-tools-version:5.9
// v0.1.35: the native macOS Settings window of Project Away (SwiftUI; Liquid Glass on macOS 26+).
// Build the .app with packaging/macos/build_settings_app.sh (it wraps `swift build -c release`).
import PackageDescription

let package = Package(
    name: "ProjectAwaySettings",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "ProjectAwaySettings",
            path: "Sources/ProjectAwaySettings",
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("IOKit"),
                .linkedFramework("ApplicationServices"),
            ]
        )
    ]
)
