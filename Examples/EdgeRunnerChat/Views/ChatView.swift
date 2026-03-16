import SwiftUI
import EdgeRunner

struct ChatView: View {
    @State private var viewModel = ChatViewModel()
    @State private var showModelPicker = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                MemoryUsageView(
                    usedMB: viewModel.state.memoryUsedMB,
                    totalMB: viewModel.state.memoryTotalMB
                )
                .padding(.horizontal)
                .padding(.top, 4)

                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(viewModel.state.messages) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }
                        }
                        .padding()
                    }
                    .onChange(of: viewModel.state.messages.count) {
                        if let last = viewModel.state.messages.last {
                            withAnimation {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                }

                if let error = viewModel.state.error {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(.horizontal)
                }

                HStack(spacing: 12) {
                    TextField("Message...", text: $viewModel.state.currentInput)
                        .textFieldStyle(.roundedBorder)
                        .disabled(viewModel.state.isGenerating)

                    Button {
                        Task { await viewModel.send() }
                    } label: {
                        Image(systemName: viewModel.state.isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                            .font(.title2)
                    }
                    .disabled(viewModel.state.currentInput.trimmingCharacters(in: .whitespaces).isEmpty && !viewModel.state.isGenerating)
                }
                .padding()
            }
            .navigationTitle(viewModel.state.selectedModel?.name ?? "EdgeRunner Chat")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showModelPicker = true
                    } label: {
                        Image(systemName: "cpu")
                    }
                }
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        viewModel.state.clearMessages()
                    } label: {
                        Image(systemName: "trash")
                    }
                }
            }
            .sheet(isPresented: $showModelPicker) {
                ModelPickerView(selectedModel: $viewModel.state.selectedModel)
            }
        }
    }
}
