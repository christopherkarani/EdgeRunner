import Foundation

/// Metadata about an available model file.
public struct ModelInfo: Identifiable, Sendable {
    public let id: UUID
    public let name: String
    public let path: URL
    public let format: String
    public let parameterCount: String
    public let quantization: String
    public let fileSizeBytes: Int64

    public init(
        name: String, path: URL, format: String,
        parameterCount: String, quantization: String, fileSizeBytes: Int64
    ) {
        self.id = UUID()
        self.name = name; self.path = path; self.format = format
        self.parameterCount = parameterCount; self.quantization = quantization
        self.fileSizeBytes = fileSizeBytes
    }

    public var fileSizeFormatted: String {
        let gb = Double(fileSizeBytes) / 1_073_741_824.0
        if gb >= 1.0 { return String(format: "%.1f GB", gb) }
        let mb = Double(fileSizeBytes) / 1_048_576.0
        return String(format: "%.1f MB", mb)
    }
}
