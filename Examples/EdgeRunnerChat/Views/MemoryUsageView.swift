import SwiftUI

struct MemoryUsageView: View {
    let usedMB: Double
    let totalMB: Double

    private var usagePercent: Double {
        guard totalMB > 0 else { return 0 }
        return usedMB / totalMB
    }

    private var usageColor: Color {
        switch usagePercent {
        case ..<0.5: return .green
        case ..<0.75: return .yellow
        case ..<0.9: return .orange
        default: return .red
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text("Memory")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.0f / %.0f MB", usedMB, totalMB))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: usagePercent)
                .tint(usageColor)
        }
    }
}
