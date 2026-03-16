import Foundation

public struct ModelConfig: Sendable, Equatable {
    public let architectureName: String
    public let metadata: [String: MetadataValue]

    public init(architectureName: String, metadata: [String: MetadataValue]) {
        self.architectureName = architectureName
        self.metadata = metadata
    }

    public func string(forKey key: String) -> String? {
        metadata[key]?.stringValue
    }

    public func int(forKey key: String) -> Int? {
        metadata[key]?.intValue
    }

    public func float(forKey key: String) -> Float? {
        metadata[key]?.floatValue
    }

    public func bool(forKey key: String) -> Bool? {
        metadata[key]?.boolValue
    }

    public func array(forKey key: String) -> [MetadataValue]? {
        metadata[key]?.arrayValue
    }
}
