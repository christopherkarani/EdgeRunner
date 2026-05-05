import EdgeRunnerChatAppCore
import SwiftUI

@main
struct EdgeRunnerChatApp: App {
    @State private var runtime = ChatRuntime(
        modelPath: LaunchConfiguration.defaultModelPath
    )

    var body: some Scene {
        WindowGroup("EdgeRunner Chat Bench") {
            ChatWindow(runtime: runtime)
                .frame(minWidth: 1180, minHeight: 760)
        }
        .windowResizability(.contentSize)
        .commands {
            CommandGroup(replacing: .newItem) {}
        }
    }
}

private enum LaunchConfiguration {
    static var defaultModelPath: String {
        let arguments = CommandLine.arguments.dropFirst()
        if let modelArgumentIndex = arguments.firstIndex(of: "--model") {
            let offset = arguments.distance(from: arguments.startIndex, to: modelArgumentIndex)
            let nextIndex = arguments.index(arguments.startIndex, offsetBy: offset + 1, limitedBy: arguments.endIndex)
            if let nextIndex, nextIndex < arguments.endIndex {
                return arguments[nextIndex]
            }
        }

        return arguments.first ?? ""
    }
}
