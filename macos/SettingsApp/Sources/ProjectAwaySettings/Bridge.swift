// v0.1.35: talks to `away settings-bridge serve` (src/away/app/settings_bridge.py): one JSON request per line on
// its stdin, one JSON response per line on its stdout. Every config rule (ranges, choices, hotkey syntax) stays in
// Python; this side only renders the document it gets and sends {key: value} changes back.
import Foundation

// MARK: - The settings document

enum JSONValue: Codable, Hashable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let n = try? c.decode(Double.self) { self = .number(n) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let n): try c.encode(n)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }

    var bool: Bool { if case .bool(let b) = self { return b }; return false }
    var double: Double {
        switch self {
        case .number(let n): return n
        case .bool(let b): return b ? 1 : 0
        case .string(let s): return Double(s) ?? 0
        default: return 0
        }
    }
    var string: String {
        switch self {
        case .string(let s): return s
        case .number(let n): return n == n.rounded() && abs(n) < 1e15 ? String(Int(n)) : String(n)
        case .bool(let b): return b ? "true" : "false"
        case .null: return ""
        default: return "\(self)"
        }
    }
    /// What JSONSerialization sends back to Python.
    var any: Any {
        switch self {
        case .null: return NSNull()
        case .bool(let b): return b
        case .number(let n): return n
        case .string(let s): return s
        case .array(let a): return a.map(\.any)
        case .object(let o): return o.mapValues(\.any)
        }
    }
}

struct Choice: Codable, Hashable {
    let value: JSONValue
    let label: String
}

struct Field: Codable, Identifiable, Hashable {
    let key: String
    let type: String  // bool | choice | text | hotkey | number
    let label: String
    let help: String
    let value: JSONValue
    let choices: [Choice]?
    let min: Double?
    let max: Double?
    let step: Double?
    let unit: String?
    let is_int: Bool?
    let `default`: Double?
    var id: String { key }
}

struct FieldGroup: Codable, Identifiable, Hashable {
    let title: String
    let footer: String
    let tuning: Bool
    let fields: [Field]
    var id: String { title }
}

struct SettingsSection: Codable, Identifiable, Hashable {
    let id: String
    let title: String
    let symbol: String
    let tk: String
    let groups: [FieldGroup]
}

struct ModelRow: Codable, Identifiable, Hashable {
    let slot: String
    let title: String
    let current: String
    let runs_on: String
    let optional: Bool
    let on: Bool
    var id: String { slot }
}

struct CameraRow: Codable, Hashable {
    let device: JSONValue
    let role: String
    let width: Int
    let height: Int
    let fps: Int
    let mirror: Bool
    let implicit: Bool
}

struct PermissionRow: Codable, Identifiable, Hashable {
    let id: String
    let title: String
    let help: String
    let granted: Bool?
    let url: String
}

struct SettingsDocument: Codable {
    let version: String
    let config_path: String
    let platform: String
    let sections: [SettingsSection]
    let tk_tabs: [String]
    let models: [ModelRow]
    let cameras: [CameraRow]
    let permissions: [PermissionRow]
    let commands: [String: [String]]
}

struct BridgeResponse: Decodable {
    let ok: Bool
    let errors: [String: String]?
    let document: SettingsDocument?
    let permissions: [PermissionRow]?
}

// MARK: - The bridge process

enum BridgeError: LocalizedError {
    case notFound
    case died(String)
    case bad(String)

    var errorDescription: String? {
        switch self {
        case .notFound: return "Project Away was not found, so its settings cannot be read."
        case .died(let why): return "The settings helper stopped (\(why))."
        case .bad(let why): return why
        }
    }
}

/// One long-lived `away settings-bridge serve`; requests run one at a time on a private queue.
final class Bridge: @unchecked Sendable {
    private let argv: [String]
    private let queue = DispatchQueue(label: "projectaway.settings.bridge")
    private var process: Process?
    private var input: FileHandle?
    private var output: FileHandle?
    private var buffer = Data()
    private var nextID = 0

    init(argv: [String]) { self.argv = argv }

    /// The argv for `settings-bridge serve`: --bridge '<json argv>' (what Project Away passes), else
    /// AWAY_SETTINGS_BRIDGE, else the ProjectAway executable of the bundle this helper sits in.
    static func resolveArgv(arguments: [String] = CommandLine.arguments,
                            env: [String: String] = ProcessInfo.processInfo.environment) -> [String]? {
        func decode(_ s: String) -> [String]? {
            guard let d = s.data(using: .utf8), let a = try? JSONSerialization.jsonObject(with: d) as? [String],
                  !a.isEmpty else { return nil }
            return a
        }
        if let i = arguments.firstIndex(of: "--bridge"), i + 1 < arguments.count, let a = decode(arguments[i + 1]) {
            return a
        }
        if let s = env["AWAY_SETTINGS_BRIDGE"], let a = decode(s) { return a }
        // .../Project Away.app/Contents/Helpers/Project Away Settings.app/Contents/MacOS/ProjectAwaySettings
        let me = URL(fileURLWithPath: Bundle.main.executablePath ?? CommandLine.arguments[0]).resolvingSymlinksInPath()
        let contents = me.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let host = contents.appendingPathComponent("MacOS/ProjectAway")
        if FileManager.default.isExecutableFile(atPath: host.path) {
            return [host.path, "settings-bridge", "serve"]
        }
        return nil
    }

    private func start() throws {
        guard let exe = argv.first else { throw BridgeError.notFound }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = Array(argv.dropFirst())
        let inPipe = Pipe(), outPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = FileHandle.nullDevice
        try p.run()
        process = p
        input = inPipe.fileHandleForWriting
        output = outPipe.fileHandleForReading
        buffer = Data()
    }

    private func readLine() throws -> Data {
        while true {
            if let nl = buffer.firstIndex(of: 0x0A) {
                let line = buffer[buffer.startIndex..<nl]
                buffer = Data(buffer[buffer.index(after: nl)...])
                return Data(line)
            }
            guard let out = output else { throw BridgeError.died("no pipe") }
            let chunk = out.availableData  // blocks until data or EOF
            if chunk.isEmpty { throw BridgeError.died("it exited") }
            buffer.append(chunk)
        }
    }

    private func roundTrip(_ request: [String: Any]) throws -> BridgeResponse {
        if process == nil || process?.isRunning != true { try start() }
        nextID += 1
        var req = request
        req["id"] = nextID
        var data = try JSONSerialization.data(withJSONObject: req)
        data.append(0x0A)
        try input?.write(contentsOf: data)
        let line = try readLine()
        do {
            return try JSONDecoder().decode(BridgeResponse.self, from: line)
        } catch {
            throw BridgeError.bad("Unexpected answer from Project Away: \(error.localizedDescription)")
        }
    }

    func request(_ request: [String: Any]) async throws -> BridgeResponse {
        try await withCheckedThrowingContinuation { cont in
            queue.async {
                do {
                    cont.resume(returning: try self.roundTrip(request))
                } catch {
                    // one retry with a fresh process (it may have been killed while the window sat open)
                    self.stopLocked()
                    do { cont.resume(returning: try self.roundTrip(request)) } catch { cont.resume(throwing: error) }
                }
            }
        }
    }

    private func stopLocked() {
        try? input?.close()
        if let p = process, p.isRunning { p.terminate() }
        process = nil
        input = nil
        output = nil
    }

    func stop() { queue.sync { stopLocked() } }
}
