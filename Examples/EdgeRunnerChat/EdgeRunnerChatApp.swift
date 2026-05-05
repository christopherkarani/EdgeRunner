import EdgeRunnerChatAppCore
import Foundation
import SwiftUI

@main
struct EdgeRunnerChatApp: App {
    @State private var runtime = ChatRuntime()
    @State private var didStartAutomatedBenchmark = false

    var body: some Scene {
        WindowGroup {
            ChatWindow(runtime: runtime)
                .task {
                    await runAutomatedBenchmarkIfRequested()
                }
        }
    }

    @MainActor
    private func runAutomatedBenchmarkIfRequested() async {
        guard !didStartAutomatedBenchmark else { return }
        didStartAutomatedBenchmark = true

        guard let documentsDirectory = FileManager.default.urls(
            for: .documentDirectory,
            in: .userDomainMask
        ).first else {
            return
        }

        guard let configuration = BenchmarkAutomationConfiguration.make(
            environment: ProcessInfo.processInfo.environment,
            documentsDirectory: documentsDirectory,
            resultDirectory: documentsDirectory
        ) else {
            return
        }

        runtime.benchmarkStatusMessage = "Benchmark running..."
        let result = await runtime.runAutomatedBenchmark(configuration)

        do {
            try BenchmarkAutomationWriter.write(result, to: configuration.resultURL)
            runtime.benchmarkStatusMessage = result.errorMessage == nil
                ? "Benchmark completed."
                : "Benchmark failed."
        } catch {
            runtime.errorMessage = String(describing: error)
            runtime.benchmarkStatusMessage = "Benchmark failed."
        }
    }
}
