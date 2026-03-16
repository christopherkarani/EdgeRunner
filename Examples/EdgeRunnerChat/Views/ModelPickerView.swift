import SwiftUI
import EdgeRunner

struct ModelPickerView: View {
    @Binding var selectedModel: ModelInfo?
    @Environment(\.dismiss) private var dismiss
    @State private var models: [ModelInfo] = []

    var body: some View {
        NavigationStack {
            List(models) { model in
                Button {
                    selectedModel = model
                    dismiss()
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(model.name)
                                .font(.headline)
                            Text("\(model.parameterCount) params - \(model.quantization)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text(model.fileSizeFormatted)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        if selectedModel?.id == model.id {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.blue)
                        }
                    }
                }
            }
            .navigationTitle("Select Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
    }
}
