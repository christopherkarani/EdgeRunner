import SwiftUI

public struct ChatWindow: View {
    @Bindable var runtime: ChatRuntime

    public init(runtime: ChatRuntime) {
        self.runtime = runtime
    }

    public var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 320, ideal: 360, max: 420)
        } detail: {
            transcriptPane
        }
        .background(
            LinearGradient(
                colors: [
                    Color(red: 0.95, green: 0.97, blue: 0.99),
                    Color(red: 0.91, green: 0.94, blue: 0.98),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )
        )
    }

    private var sidebar: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                headerCard
                modelCard
                metricsCard
                controlsCard
            }
            .padding(20)
        }
        .scrollContentBackground(.hidden)
        .background(.clear)
    }

    private var transcriptPane: some View {
        VStack(spacing: 0) {
            transcriptHeader
            Divider()
            transcriptScroll
            Divider()
            composer
        }
        .background(
            RoundedRectangle(cornerRadius: 28, style: .continuous)
                .fill(.ultraThinMaterial)
                .overlay(
                    RoundedRectangle(cornerRadius: 28, style: .continuous)
                        .stroke(Color.white.opacity(0.55), lineWidth: 1)
                )
                .padding(18)
        )
    }

    private var headerCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("EdgeRunner Chat Bench")
                .font(.system(size: 28, weight: .bold, design: .rounded))
            Text("\(runtime.modelDisplayName) at a \(contextWindowLabel) context window with live TTFT and decode throughput.")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HStack(spacing: 10) {
                statusPill(
                    title: runtime.isGenerating ? "Streaming" : "Idle",
                    color: runtime.isGenerating ? .green : .blue
                )
                statusPill(
                    title: "Target 30 tok/s",
                    color: throughputColor(for: runtime.metrics.finalDecodeTokensPerSecond)
                )
                statusPill(title: "Context \(runtime.contextWindowSize)", color: .brown)
            }
        }
        .cardStyle()
    }

    private var modelCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Model")
                .sectionTitle()

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Local models")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Refresh") {
                        runtime.refreshDiscoveredModels()
                    }
                    .font(.caption.weight(.semibold))
                    .buttonStyle(.borderless)
                }

                if runtime.availableLocalModels.isEmpty {
                    Text("No `.gguf` files found in the app Documents folder yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    VStack(spacing: 8) {
                        ForEach(runtime.availableLocalModels) { model in
                            Button {
                                runtime.selectModel(model)
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(model.displayName)
                                            .font(.callout.weight(.semibold))
                                            .foregroundStyle(.primary)
                                        Text(model.path)
                                            .font(.caption.monospaced())
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)
                                    }
                                    Spacer()
                                    if runtime.modelPath == model.path {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundStyle(.blue)
                                    }
                                }
                                .padding(12)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(
                                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                                        .fill(
                                            runtime.modelPath == model.path
                                                ? Color.blue.opacity(0.12)
                                                : Color.black.opacity(0.05)
                                        )
                                )
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }

            DisclosureGroup("Advanced") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Manual GGUF Path Override")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    TextField("/path/to/model.gguf", text: $runtime.modelPath)
                        .textFieldStyle(.roundedBorder)
                }
                .padding(.top, 8)
            }
            .font(.caption.weight(.semibold))

            VStack(alignment: .leading, spacing: 8) {
                Text("System Prompt")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                TextField("Optional system prompt", text: $runtime.systemPrompt, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(3...6)
            }
        }
        .cardStyle()
    }

    private var metricsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Metrics")
                .sectionTitle()

            MetricRow(label: "Generated", value: "\(runtime.metrics.generatedTokenCount) tok")
            MetricRow(
                label: "TTFT",
                value: runtime.metrics.timeToFirstTokenSeconds.map(Self.secondsString) ?? "—"
            )
            MetricRow(
                label: "Live decode",
                value: runtime.metrics.rollingDecodeTokensPerSecond > 0
                    ? Self.tokensPerSecondString(runtime.metrics.rollingDecodeTokensPerSecond)
                    : "—"
            )
            MetricRow(
                label: "Final decode",
                value: runtime.metrics.finalDecodeTokensPerSecond > 0
                    ? Self.tokensPerSecondString(runtime.metrics.finalDecodeTokensPerSecond)
                    : "—"
            )
            MetricRow(
                label: "End to end",
                value: runtime.metrics.endToEndTokensPerSecond > 0
                    ? Self.tokensPerSecondString(runtime.metrics.endToEndTokensPerSecond)
                    : "—"
            )

            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Progress to target")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(targetProgressText)
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                ProgressView(value: targetProgress, total: 1)
                    .tint(throughputColor(for: runtime.metrics.finalDecodeTokensPerSecond))
            }
            .padding(.top, 4)
        }
        .cardStyle()
    }

    private var controlsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Controls")
                .sectionTitle()

            Stepper(value: $runtime.maxResponseTokens, in: 32...2048, step: 32) {
                HStack {
                    Text("Max response tokens")
                    Spacer()
                    Text("\(runtime.maxResponseTokens)")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Benchmark mode")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                Picker("Benchmark mode", selection: $runtime.benchmarkMode) {
                    ForEach(BenchmarkMode.allCases, id: \.self) { mode in
                        Text(mode.displayName).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("Benchmark prompt")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                TextField(
                    "Explain quantum computing in simple terms.",
                    text: $runtime.benchmarkPrompt,
                    axis: .vertical
                )
                .textFieldStyle(.roundedBorder)
                .lineLimit(2...5)
            }

            HStack(spacing: 12) {
                Button(runtime.isGenerating ? "Benchmarking…" : "Run Benchmark") {
                    Task {
                        await runtime.runBenchmark()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(runtime.isGenerating)

                Button("Reset Chat") {
                    runtime.resetConversation()
                }
                .buttonStyle(.bordered)

                if runtime.isGenerating {
                    Button("Stop") {
                        runtime.cancelGeneration()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.orange)
                }
            }

            if let benchmarkStatusMessage = runtime.benchmarkStatusMessage {
                Text(benchmarkStatusMessage)
                    .font(.callout)
                    .foregroundStyle(.blue)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(Color.blue.opacity(0.08))
                    )
            }

            if let errorMessage = runtime.errorMessage {
                Text(errorMessage)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(Color.red.opacity(0.08))
                    )
            }
        }
        .cardStyle()
    }

    private var transcriptHeader: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Conversation")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                Text("Send uses the streamed chat path. Run Benchmark can measure either the raw decode ceiling or the streamed path.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text("\(runtime.messages.count) messages")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 32)
        .padding(.top, 30)
        .padding(.bottom, 20)
    }

    private var transcriptScroll: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    if runtime.messages.isEmpty {
                        emptyState
                    } else {
                        ForEach(runtime.messages) { message in
                            MessageBubble(entry: message)
                                .id(message.id)
                        }
                    }
                }
                .padding(.horizontal, 32)
                .padding(.vertical, 24)
            }
            .scrollContentBackground(.hidden)
            .background(.clear)
            .onChange(of: runtime.messages.count) { _, _ in
                if let lastMessage = runtime.messages.last {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(lastMessage.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Ready to benchmark")
                .font(.system(size: 22, weight: .semibold, design: .rounded))
            Text("Point the app at a local GGUF, then either run the benchmark or ask a question. Raw decode mode measures the model ceiling. Chat stream mode includes streaming and UI overhead.")
                .foregroundStyle(.secondary)
            Text("Tip: pick a local model above, choose a benchmark mode, tap Run Benchmark, or send a prompt manually.")
                .font(.callout.monospaced())
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .fill(Color.black.opacity(0.05))
                )
        }
        .padding(28)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 24, style: .continuous)
                .fill(Color.white.opacity(0.72))
        )
    }

    private var composer: some View {
        VStack(spacing: 12) {
            TextField("Ask something to measure throughput on-device…", text: $runtime.currentInput, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(3...8)

            HStack {
                Text("Chat and benchmark runs use the current local model path.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button(runtime.isGenerating ? "Streaming…" : "Send") {
                    runtime.sendCurrentInput()
                }
                .buttonStyle(.borderedProminent)
                .disabled(runtime.isGenerating)
            }
        }
        .padding(24)
    }

    private var targetProgress: Double {
        min(runtime.metrics.finalDecodeTokensPerSecond / 30.0, 1.0)
    }

    private var targetProgressText: String {
        let value = runtime.metrics.finalDecodeTokensPerSecond
        guard value > 0 else { return "0%" }
        return "\(Int((targetProgress * 100).rounded()))%"
    }

    private var contextWindowLabel: String {
        if runtime.contextWindowSize % 1024 == 0 {
            return "\(runtime.contextWindowSize / 1024)k"
        }
        return "\(runtime.contextWindowSize)"
    }

    private func statusPill(title: String, color: Color) -> some View {
        HStack(spacing: 8) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(title)
                .font(.caption.weight(.semibold))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(
            Capsule(style: .continuous)
                .fill(color.opacity(0.12))
        )
    }

    private func throughputColor(for tokensPerSecond: Double) -> Color {
        guard tokensPerSecond > 0 else { return .gray }
        if tokensPerSecond >= 30 { return .green }
        if tokensPerSecond >= 20 { return .orange }
        return .red
    }

    private static func tokensPerSecondString(_ value: Double) -> String {
        String(format: "%.1f tok/s", value)
    }

    private static func secondsString(_ value: Double) -> String {
        if value < 1 {
            return String(format: "%.0f ms", value * 1000)
        }
        return String(format: "%.2f s", value)
    }
}

private struct MetricRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.system(.body, design: .monospaced))
        }
    }
}

private struct MessageBubble: View {
    let entry: ChatTranscriptEntry

    var body: some View {
        VStack(alignment: alignment, spacing: 8) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(entry.content.isEmpty ? "…" : entry.content)
                .textSelection(.enabled)
                .frame(maxWidth: 620, alignment: .leading)
                .padding(16)
                .background(
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .fill(backgroundColor)
                )
        }
        .frame(maxWidth: .infinity, alignment: frameAlignment)
    }

    private var title: String {
        switch entry.role {
        case .system:
            return "System"
        case .user:
            return "You"
        case .assistant:
            return "EdgeRunner"
        }
    }

    private var alignment: HorizontalAlignment {
        entry.role == .user ? .trailing : .leading
    }

    private var frameAlignment: Alignment {
        entry.role == .user ? .trailing : .leading
    }

    private var backgroundColor: Color {
        switch entry.role {
        case .system:
            return Color.black.opacity(0.06)
        case .user:
            return Color(red: 0.86, green: 0.93, blue: 1.0)
        case .assistant:
            return Color.white.opacity(0.88)
        }
    }
}

private extension Text {
    func sectionTitle() -> some View {
        self
            .font(.system(size: 16, weight: .bold, design: .rounded))
    }
}

private extension View {
    func cardStyle() -> some View {
        self
            .padding(18)
            .background(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 24, style: .continuous)
                            .stroke(Color.white.opacity(0.55), lineWidth: 1)
                    )
            )
    }
}
