import Testing
import Foundation
@testable import EdgeRunnerCore

// MARK: - Test types for structured generation

struct PersonInfo: Codable, Equatable, Sendable {
    let name: String
    let age: Int
}

struct WeatherReport: Codable, Equatable, Sendable {
    let city: String
    let temperature: Double
    let isRaining: Bool
}

struct NestedType: Codable, Equatable, Sendable {
    struct Address: Codable, Equatable, Sendable {
        let street: String
        let zip: String
    }
    let name: String
    let address: Address
}

struct ArrayType: Codable, Equatable, Sendable {
    let tags: [String]
    let scores: [Int]
}

struct OptionalType: Codable, Equatable, Sendable {
    let required: String
    let optional: String?
}

enum Status: String, Codable, Sendable {
    case active
    case inactive
    case pending
}

struct EnumType: Codable, Equatable, Sendable {
    let name: String
    let status: Status
}

@Suite("JSONSchemaExtractor")
struct JSONSchemaExtractorTests {

    @Test func extractSimpleStruct() throws {
        let schema = try JSONSchemaExtractor.extractSchema(for: PersonInfo.self)
        #expect(schema.type == .object)
        #expect(schema.properties?.count == 2)
        #expect(schema.properties?["name"]?.type == .string)
        #expect(schema.properties?["age"]?.type == .integer)
        #expect(schema.required?.contains("name") == true)
        #expect(schema.required?.contains("age") == true)
    }

    @Test func extractWithBoolAndDouble() throws {
        let schema = try JSONSchemaExtractor.extractSchema(for: WeatherReport.self)
        #expect(schema.properties?["city"]?.type == .string)
        #expect(schema.properties?["temperature"]?.type == .number)
        #expect(schema.properties?["isRaining"]?.type == .boolean)
    }

    @Test func extractNestedObject() throws {
        let schema = try JSONSchemaExtractor.extractSchema(for: NestedType.self)
        #expect(schema.properties?["name"]?.type == .string)
        let addressSchema = schema.properties?["address"]
        #expect(addressSchema?.type == .object)
        #expect(addressSchema?.properties?["street"]?.type == .string)
        #expect(addressSchema?.properties?["zip"]?.type == .string)
    }

    @Test func extractArrayTypes() throws {
        let schema = try JSONSchemaExtractor.extractSchema(for: ArrayType.self)
        #expect(schema.properties?["tags"]?.type == .array)
        #expect(schema.properties?["tags"]?.items?.type == .string)
        #expect(schema.properties?["scores"]?.type == .array)
        #expect(schema.properties?["scores"]?.items?.type == .integer)
    }

    @Test func extractOptionalFields() throws {
        let schema = try JSONSchemaExtractor.extractSchema(for: OptionalType.self)
        #expect(schema.required?.contains("required") == true)
        #expect(schema.required?.contains("optional") == false)
    }

    @Test func schemaToJSON() throws {
        let schema = try JSONSchemaExtractor.extractSchema(for: PersonInfo.self)
        let json = try schema.toJSON()
        #expect(json.contains("\"type\":\"object\"") || json.contains("\"type\": \"object\""))
        #expect(json.contains("name"))
        #expect(json.contains("age"))
    }
}

@Suite("GrammarState")
struct GrammarStateTests {

    @Test func initialStateExpectsOpenBrace() {
        let schema = JSONSchema(type: .object, properties: [
            "name": JSONSchema(type: .string),
        ], required: ["name"])
        let state = GrammarState(schema: schema)
        let allowed = state.allowedNextCharacters()
        #expect(allowed.contains("{"))
        #expect(!allowed.contains("}"))
    }

    @Test func afterOpenBraceExpectsKey() {
        let schema = JSONSchema(type: .object, properties: [
            "name": JSONSchema(type: .string),
        ], required: ["name"])
        var state = GrammarState(schema: schema)
        state.advance(with: "{")
        let allowed = state.allowedNextCharacters()
        #expect(allowed.contains("\""))
    }

    @Test func validateCompleteJSON() {
        let schema = JSONSchema(type: .object, properties: [
            "name": JSONSchema(type: .string),
            "age": JSONSchema(type: .integer),
        ], required: ["name", "age"])
        let json = #"{"name":"Alice","age":30}"#
        let isValid = GrammarState.validate(json: json, against: schema)
        #expect(isValid)
    }

    @Test func rejectInvalidJSON() {
        let schema = JSONSchema(type: .object, properties: [
            "name": JSONSchema(type: .string),
        ], required: ["name"])
        let json = #"{"name": 42}"# // name should be string
        let isValid = GrammarState.validate(json: json, against: schema)
        #expect(!isValid)
    }

    @Test func validateNestedJSON() {
        let schema = JSONSchema(type: .object, properties: [
            "name": JSONSchema(type: .string),
            "address": JSONSchema(type: .object, properties: [
                "street": JSONSchema(type: .string),
            ], required: ["street"]),
        ], required: ["name", "address"])
        let json = #"{"name":"Bob","address":{"street":"123 Main"}}"#
        let isValid = GrammarState.validate(json: json, against: schema)
        #expect(isValid)
    }
}

@Suite("ConstrainedDecoder")
struct ConstrainedDecoderTests {

    @Test func maskDisallowedTokens() {
        let vocab = ["a", "b", "{", "}", "\"", ":", ",", "1", " ", "\n"]
        let schema = JSONSchema(type: .object, properties: [
            "x": JSONSchema(type: .string),
        ], required: ["x"])
        let state = GrammarState(schema: schema)

        let decoder = ConstrainedDecoder(vocabulary: vocab)
        let mask = decoder.computeMask(for: state)

        #expect(mask.count == vocab.count)
        // At start, only "{" should be allowed
        #expect(mask[2] == true)  // "{"
        #expect(mask[0] == false) // "a"
    }

    @Test func applyMaskToLogits() {
        let vocab = ["{", "}", "a"]
        let decoder = ConstrainedDecoder(vocabulary: vocab)
        let mask = [true, false, false]
        let logits: [Float] = [5.0, 10.0, 3.0]
        let masked = decoder.applyMask(mask, to: logits)
        #expect(masked[0] == 5.0)
        #expect(masked[1] == -.infinity)
        #expect(masked[2] == -.infinity)
    }
}

@Suite("StructuredGenerator")
struct StructuredGeneratorTests {

    @Test func parseValidJSON() throws {
        let json = #"{"name":"Alice","age":30}"#
        let result: PersonInfo = try StructuredGenerator.parse(json: json)
        #expect(result.name == "Alice")
        #expect(result.age == 30)
    }

    @Test func parseNestedJSON() throws {
        let json = #"{"name":"Bob","address":{"street":"123 Main","zip":"12345"}}"#
        let result: NestedType = try StructuredGenerator.parse(json: json)
        #expect(result.name == "Bob")
        #expect(result.address.street == "123 Main")
        #expect(result.address.zip == "12345")
    }

    @Test func parseWithArrays() throws {
        let json = #"{"tags":["swift","metal"],"scores":[95,87]}"#
        let result: ArrayType = try StructuredGenerator.parse(json: json)
        #expect(result.tags == ["swift", "metal"])
        #expect(result.scores == [95, 87])
    }

    @Test func invalidJSONThrows() throws {
        let json = "not valid json"
        #expect(throws: (any Error).self) {
            let _: PersonInfo = try StructuredGenerator.parse(json: json)
        }
    }

    @Test func extractJSONFromModelOutput() throws {
        let output = """
        Here is the result:
        ```json
        {"name":"Alice","age":25}
        ```
        That's the answer.
        """
        let json = try StructuredGenerator.extractJSON(from: output)
        #expect(json == #"{"name":"Alice","age":25}"#)
    }

    @Test func extractJSONBracketMatching() throws {
        let output = #"Some text {"name":"Bob","age":30} more text"#
        let json = try StructuredGenerator.extractJSON(from: output)
        #expect(json.contains("Bob"))
    }
}
