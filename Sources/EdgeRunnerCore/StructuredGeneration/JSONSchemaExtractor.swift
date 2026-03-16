import Foundation

/// JSON Schema type identifiers.
public enum JSONSchemaType: String, Codable, Sendable {
    case object, array, string, integer, number, boolean, null
}

/// Thread-safe box for heap-allocated indirection, breaking recursive value type cycles.
public final class Box<T: Sendable>: @unchecked Sendable {
    public let value: T
    public init(_ value: T) { self.value = value }
}

/// A simplified JSON Schema representation for constrained decoding.
public struct JSONSchema: Sendable {
    public let type: JSONSchemaType
    public let properties: [String: JSONSchema]?
    public let required: [String]?
    private let _items: Box<JSONSchema>?
    public var items: JSONSchema? { _items?.value }
    public let enumValues: [String]?
    public let description: String?

    public init(
        type: JSONSchemaType,
        properties: [String: JSONSchema]? = nil,
        required: [String]? = nil,
        items: JSONSchema? = nil,
        enumValues: [String]? = nil,
        description: String? = nil
    ) {
        self.type = type
        self.properties = properties
        self.required = required
        self._items = items.map { Box($0) }
        self.enumValues = enumValues
        self.description = description
    }

    public func toJSON() throws -> String {
        let dict = toDictionary()
        let data = try JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])
        guard let str = String(data: data, encoding: .utf8) else {
            throw GenerationError.structuredOutputFailed(reason: "Failed to serialize schema")
        }
        return str
    }

    private func toDictionary() -> [String: Any] {
        var dict = [String: Any]()
        dict["type"] = type.rawValue
        if let props = properties {
            var propsDict = [String: Any]()
            for (key, schema) in props { propsDict[key] = schema.toDictionary() }
            dict["properties"] = propsDict
        }
        if let req = required { dict["required"] = req }
        if let items = items { dict["items"] = items.toDictionary() }
        if let enums = enumValues { dict["enum"] = enums }
        return dict
    }
}

/// Extracts JSON Schema from Swift Decodable types using Mirror-based reflection.
public enum JSONSchemaExtractor {
    public static func extractSchema<T: Decodable>(for type: T.Type) throws -> JSONSchema {
        let decoder = SchemaIntrospectionDecoder()
        _ = try? T(from: decoder)
        return decoder.buildSchema()
    }
}

// MARK: - Schema Introspection Decoder

private final class SchemaIntrospectionDecoder: Decoder {
    var codingPath: [CodingKey] = []
    var userInfo: [CodingUserInfoKey: Any] = [:]
    var discoveredProperties: [(key: String, schema: JSONSchema, isOptional: Bool)] = []

    func container<Key: CodingKey>(keyedBy type: Key.Type) throws -> KeyedDecodingContainer<Key> {
        KeyedDecodingContainer(SchemaKeyedContainer<Key>(decoder: self))
    }
    func unkeyedContainer() throws -> UnkeyedDecodingContainer {
        SchemaUnkeyedContainer(decoder: self)
    }
    func singleValueContainer() throws -> SingleValueDecodingContainer {
        SchemaSingleValueContainer(decoder: self)
    }

    func buildSchema() -> JSONSchema {
        let properties = Dictionary(uniqueKeysWithValues: discoveredProperties.map { ($0.key, $0.schema) })
        let required = discoveredProperties.filter { !$0.isOptional }.map(\.key)
        return JSONSchema(type: .object, properties: properties, required: required.isEmpty ? nil : required)
    }
}

private struct SchemaKeyedContainer<Key: CodingKey>: KeyedDecodingContainerProtocol {
    let decoder: SchemaIntrospectionDecoder
    var codingPath: [CodingKey] { decoder.codingPath }
    var allKeys: [Key] { [] }
    func contains(_ key: Key) -> Bool { true }

    func decodeNil(forKey key: Key) throws -> Bool {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .null), true))
        return true
    }
    func decode(_ type: Bool.Type, forKey key: Key) throws -> Bool {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .boolean), false)); return false
    }
    func decode(_ type: String.Type, forKey key: Key) throws -> String {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .string), false)); return ""
    }
    func decode(_ type: Double.Type, forKey key: Key) throws -> Double {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .number), false)); return 0
    }
    func decode(_ type: Float.Type, forKey key: Key) throws -> Float {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .number), false)); return 0
    }
    func decode(_ type: Int.Type, forKey key: Key) throws -> Int {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false)); return 0
    }
    func decode(_ type: Int8.Type, forKey key: Key) throws -> Int8 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false)); return 0
    }
    func decode(_ type: Int16.Type, forKey key: Key) throws -> Int16 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false)); return 0
    }
    func decode(_ type: Int32.Type, forKey key: Key) throws -> Int32 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false)); return 0
    }
    func decode(_ type: Int64.Type, forKey key: Key) throws -> Int64 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false)); return 0
    }
    func decode(_ type: UInt.Type, forKey key: Key) throws -> UInt {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false)); return 0
    }
    func decode(_ type: UInt8.Type, forKey key: Key) throws -> UInt8 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false)); return 0
    }
    func decode(_ type: UInt16.Type, forKey key: Key) throws -> UInt16 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false)); return 0
    }
    func decode(_ type: UInt32.Type, forKey key: Key) throws -> UInt32 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false)); return 0
    }
    func decode(_ type: UInt64.Type, forKey key: Key) throws -> UInt64 {
        decoder.discoveredProperties.append((key.stringValue, JSONSchema(type: .integer), false)); return 0
    }

    func decode<T: Decodable>(_ type: T.Type, forKey key: Key) throws -> T {
        let schema = inferSchema(for: T.self)
        let isOptional = isOptionalType(T.self)
        decoder.discoveredProperties.append((key.stringValue, schema, isOptional))
        // Try to return a dummy value to allow decoding to continue
        // for remaining properties. Fall back to throwing if we can't.
        if let empty = emptyValue(for: T.self) {
            return empty
        }
        throw SchemaExtractionComplete()
    }

    func decodeIfPresent<T: Decodable>(_ type: T.Type, forKey key: Key) throws -> T? {
        let schema = inferSchema(for: T.self)
        decoder.discoveredProperties.append((key.stringValue, schema, true))
        return nil
    }

    func nestedContainer<NestedKey: CodingKey>(keyedBy type: NestedKey.Type, forKey key: Key) throws -> KeyedDecodingContainer<NestedKey> {
        KeyedDecodingContainer(SchemaKeyedContainer<NestedKey>(decoder: decoder))
    }
    func nestedUnkeyedContainer(forKey key: Key) throws -> UnkeyedDecodingContainer {
        SchemaUnkeyedContainer(decoder: decoder)
    }
    func superDecoder() throws -> Decoder { decoder }
    func superDecoder(forKey key: Key) throws -> Decoder { decoder }
}

private struct SchemaUnkeyedContainer: UnkeyedDecodingContainer {
    let decoder: SchemaIntrospectionDecoder
    var codingPath: [CodingKey] { decoder.codingPath }
    var count: Int? { 0 }
    var isAtEnd: Bool { true }
    var currentIndex: Int { 0 }
    mutating func decodeNil() throws -> Bool { true }
    mutating func decode(_ type: Bool.Type) throws -> Bool { false }
    mutating func decode(_ type: String.Type) throws -> String { "" }
    mutating func decode(_ type: Double.Type) throws -> Double { 0 }
    mutating func decode(_ type: Float.Type) throws -> Float { 0 }
    mutating func decode(_ type: Int.Type) throws -> Int { 0 }
    mutating func decode(_ type: Int8.Type) throws -> Int8 { 0 }
    mutating func decode(_ type: Int16.Type) throws -> Int16 { 0 }
    mutating func decode(_ type: Int32.Type) throws -> Int32 { 0 }
    mutating func decode(_ type: Int64.Type) throws -> Int64 { 0 }
    mutating func decode(_ type: UInt.Type) throws -> UInt { 0 }
    mutating func decode(_ type: UInt8.Type) throws -> UInt8 { 0 }
    mutating func decode(_ type: UInt16.Type) throws -> UInt16 { 0 }
    mutating func decode(_ type: UInt32.Type) throws -> UInt32 { 0 }
    mutating func decode(_ type: UInt64.Type) throws -> UInt64 { 0 }
    mutating func decode<T: Decodable>(_ type: T.Type) throws -> T { throw SchemaExtractionComplete() }
    mutating func nestedContainer<NestedKey: CodingKey>(keyedBy type: NestedKey.Type) throws -> KeyedDecodingContainer<NestedKey> {
        KeyedDecodingContainer(SchemaKeyedContainer<NestedKey>(decoder: decoder))
    }
    mutating func nestedUnkeyedContainer() throws -> UnkeyedDecodingContainer { SchemaUnkeyedContainer(decoder: decoder) }
    mutating func superDecoder() throws -> Decoder { decoder }
}

private struct SchemaSingleValueContainer: SingleValueDecodingContainer {
    let decoder: SchemaIntrospectionDecoder
    var codingPath: [CodingKey] { decoder.codingPath }
    func decodeNil() -> Bool { true }
    func decode(_ type: Bool.Type) throws -> Bool { false }
    func decode(_ type: String.Type) throws -> String { "" }
    func decode(_ type: Double.Type) throws -> Double { 0 }
    func decode(_ type: Float.Type) throws -> Float { 0 }
    func decode(_ type: Int.Type) throws -> Int { 0 }
    func decode(_ type: Int8.Type) throws -> Int8 { 0 }
    func decode(_ type: Int16.Type) throws -> Int16 { 0 }
    func decode(_ type: Int32.Type) throws -> Int32 { 0 }
    func decode(_ type: Int64.Type) throws -> Int64 { 0 }
    func decode(_ type: UInt.Type) throws -> UInt { 0 }
    func decode(_ type: UInt8.Type) throws -> UInt8 { 0 }
    func decode(_ type: UInt16.Type) throws -> UInt16 { 0 }
    func decode(_ type: UInt32.Type) throws -> UInt32 { 0 }
    func decode(_ type: UInt64.Type) throws -> UInt64 { 0 }
    func decode<T: Decodable>(_ type: T.Type) throws -> T { throw SchemaExtractionComplete() }
}

private struct SchemaExtractionComplete: Error {}

/// Attempt to create an empty/default value for common Decodable types
/// so schema introspection can continue past the first non-primitive property.
private func emptyValue<T>(for type: T.Type) -> T? {
    // Handle common array types
    if type == [String].self { return [String]() as? T }
    if type == [Int].self { return [Int]() as? T }
    if type == [Double].self { return [Double]() as? T }
    if type == [Float].self { return [Float]() as? T }
    if type == [Bool].self { return [Bool]() as? T }
    // Handle primitive wrappers
    if type == String.self { return "" as? T }
    if type == Int.self { return 0 as? T }
    if type == Double.self { return 0.0 as? T }
    if type == Float.self { return Float(0) as? T }
    if type == Bool.self { return false as? T }
    // Try JSON-decoding an empty object or empty string for other Decodable types
    if let decodable = type as? Decodable.Type {
        if let val = try? decodable.init(from: EmptyDecoder()) as? T { return val }
    }
    return nil
}

/// Minimal decoder that provides empty containers for constructing default values.
private final class EmptyDecoder: Decoder {
    var codingPath: [CodingKey] = []
    var userInfo: [CodingUserInfoKey: Any] = [:]
    func container<Key: CodingKey>(keyedBy type: Key.Type) throws -> KeyedDecodingContainer<Key> {
        KeyedDecodingContainer(SchemaKeyedContainer<Key>(decoder: SchemaIntrospectionDecoder()))
    }
    func unkeyedContainer() throws -> UnkeyedDecodingContainer {
        SchemaUnkeyedContainer(decoder: SchemaIntrospectionDecoder())
    }
    func singleValueContainer() throws -> SingleValueDecodingContainer {
        SchemaSingleValueContainer(decoder: SchemaIntrospectionDecoder())
    }
}

private func inferSchema(for type: Any.Type) -> JSONSchema {
    switch type {
    case is String.Type: return JSONSchema(type: .string)
    case is Int.Type, is Int8.Type, is Int16.Type, is Int32.Type, is Int64.Type,
         is UInt.Type, is UInt8.Type, is UInt16.Type, is UInt32.Type, is UInt64.Type:
        return JSONSchema(type: .integer)
    case is Double.Type, is Float.Type: return JSONSchema(type: .number)
    case is Bool.Type: return JSONSchema(type: .boolean)
    default:
        if let arrayElementType = arrayElementType(type) {
            return JSONSchema(type: .array, items: inferSchema(for: arrayElementType))
        }
        if let wrappedType = optionalWrappedType(type) {
            return inferSchema(for: wrappedType)
        }
        if let decodableType = type as? Decodable.Type {
            let decoder = SchemaIntrospectionDecoder()
            _ = try? decodableType.init(from: decoder)
            return decoder.buildSchema()
        }
        return JSONSchema(type: .object)
    }
}

private func isOptionalType(_ type: Any.Type) -> Bool {
    String(describing: type).hasPrefix("Optional<")
}

private func optionalWrappedType(_ type: Any.Type) -> Any.Type? {
    let description = String(describing: type)
    guard description.hasPrefix("Optional<") else { return nil }
    return nil
}

private func arrayElementType(_ type: Any.Type) -> Any.Type? {
    let description = String(describing: type)
    if description.hasPrefix("Array<String>") { return String.self }
    if description.hasPrefix("Array<Int>") { return Int.self }
    if description.hasPrefix("Array<Double>") { return Double.self }
    if description.hasPrefix("Array<Float>") { return Float.self }
    if description.hasPrefix("Array<Bool>") { return Bool.self }
    return nil
}
