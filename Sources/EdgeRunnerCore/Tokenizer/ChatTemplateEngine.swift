// MARK: - Error Types

/// Errors that can occur during chat template parsing or evaluation.
public enum ChatTemplateError: Error, Sendable {
    case parseError(String)
    case unsupportedFeature(String)
    case evaluationError(String)
}

// MARK: - AST Types

/// A value produced during template evaluation.
private enum TemplateValue: Sendable {
    case string(String)
    case int(Int)
    case bool(Bool)
    case array([TemplateValue])
    case dict([String: TemplateValue])
    case none

    var isTruthy: Bool {
        switch self {
        case .string(let s): !s.isEmpty
        case .int(let i): i != 0
        case .bool(let b): b
        case .array(let a): !a.isEmpty
        case .dict(let d): !d.isEmpty
        case .none: false
        }
    }

    var asString: String {
        switch self {
        case .string(let s): s
        case .int(let i): String(i)
        case .bool(let b): b ? "True" : "False"
        case .array: "[array]"
        case .dict: "[dict]"
        case .none: ""
        }
    }
}

/// An expression in the template AST.
private indirect enum Expression: Sendable {
    case stringLiteral(String)
    case intLiteral(Int)
    case boolLiteral(Bool)
    case variable(String)
    case memberAccess(Expression, String)
    case binaryOp(String, Expression, Expression)
    case unaryOp(String, Expression)
    case filter(Expression, String)
}

/// A node in the template AST.
private indirect enum Node: Sendable {
    case text(String)
    case output(Expression)
    case forLoop(variable: String, iterable: Expression, body: [Node])
    case ifBlock(branches: [(condition: Expression?, body: [Node])])
    case setVar(variable: String, value: Expression)
}

// MARK: - Lexer

/// Token types produced by the lexer.
private enum TokenKind: Sendable, Equatable {
    case text(String)
    case expressionStart       // {{
    case expressionEnd         // }}
    case statementStart        // {%
    case statementEnd          // %}
}

/// A raw token from lexing, carrying whitespace-strip flags.
private struct RawBlock: Sendable {
    enum Kind: Sendable {
        case text(String)
        case expression(String)   // content between {{ and }}
        case statement(String)    // content between {% and %}
        case comment              // {# ... #} — ignored
    }
    let kind: Kind
    let leftStrip: Bool    // {%-  or {{-
    let rightStrip: Bool   // -%}} or -%}
}

/// Lexes a template string into raw blocks.
private func lexTemplate(_ template: String) -> [RawBlock] {
    var blocks: [RawBlock] = []
    var remaining = template[...]

    while !remaining.isEmpty {
        // Find the next tag opening: {{ , {% , or {#
        guard let openIdx = findNextTagOpen(in: remaining) else {
            // Rest is plain text
            blocks.append(RawBlock(kind: .text(String(remaining)), leftStrip: false, rightStrip: false))
            break
        }

        // Text before the tag
        if openIdx > remaining.startIndex {
            let text = String(remaining[remaining.startIndex..<openIdx])
            blocks.append(RawBlock(kind: .text(text), leftStrip: false, rightStrip: false))
        }

        let tagStart = remaining[openIdx...]
        let second = tagStart.index(after: openIdx)

        if tagStart[second] == "{" {
            // Expression {{ ... }}
            let contentStart = tagStart.index(second, offsetBy: 1)
            let leftStrip = contentStart < tagStart.endIndex && tagStart[contentStart] == "-"
            let actualContentStart = leftStrip ? tagStart.index(after: contentStart) : contentStart

            if let closeIdx = findClose("}}", in: tagStart[actualContentStart...]) {
                let rightStrip = closeIdx > actualContentStart && tagStart[tagStart.index(before: closeIdx)] == "-"
                let contentEnd = rightStrip ? tagStart.index(before: closeIdx) : closeIdx
                let content = String(tagStart[actualContentStart..<contentEnd]).trimmingCharacters(in: .whitespaces)
                blocks.append(RawBlock(kind: .expression(content), leftStrip: leftStrip, rightStrip: rightStrip))
                remaining = tagStart[tagStart.index(closeIdx, offsetBy: 2)...]
            } else {
                // Malformed — treat rest as text
                blocks.append(RawBlock(kind: .text(String(remaining[openIdx...])), leftStrip: false, rightStrip: false))
                break
            }
        } else if tagStart[second] == "%" {
            // Statement {% ... %}
            let contentStart = tagStart.index(second, offsetBy: 1)
            let leftStrip = contentStart < tagStart.endIndex && tagStart[contentStart] == "-"
            let actualContentStart = leftStrip ? tagStart.index(after: contentStart) : contentStart

            if let closeIdx = findClose("%}", in: tagStart[actualContentStart...]) {
                let rightStrip = closeIdx > actualContentStart && tagStart[tagStart.index(before: closeIdx)] == "-"
                let contentEnd = rightStrip ? tagStart.index(before: closeIdx) : closeIdx
                let content = String(tagStart[actualContentStart..<contentEnd]).trimmingCharacters(in: .whitespaces)
                blocks.append(RawBlock(kind: .statement(content), leftStrip: leftStrip, rightStrip: rightStrip))
                remaining = tagStart[tagStart.index(closeIdx, offsetBy: 2)...]
            } else {
                blocks.append(RawBlock(kind: .text(String(remaining[openIdx...])), leftStrip: false, rightStrip: false))
                break
            }
        } else if tagStart[second] == "#" {
            // Comment {# ... #}
            let contentStart = tagStart.index(second, offsetBy: 1)
            if let closeIdx = findClose("#}", in: tagStart[contentStart...]) {
                blocks.append(RawBlock(kind: .comment, leftStrip: false, rightStrip: false))
                remaining = tagStart[tagStart.index(closeIdx, offsetBy: 2)...]
            } else {
                blocks.append(RawBlock(kind: .text(String(remaining[openIdx...])), leftStrip: false, rightStrip: false))
                break
            }
        } else {
            // Not a tag, just a lone {
            blocks.append(RawBlock(kind: .text("{"), leftStrip: false, rightStrip: false))
            remaining = remaining[second...]
        }
    }

    return blocks
}

/// Find the next `{%`, `{{`, or `{#` in the substring.
private func findNextTagOpen(in s: Substring) -> Substring.Index? {
    var i = s.startIndex
    while i < s.endIndex {
        if s[i] == "{" {
            let next = s.index(after: i)
            if next < s.endIndex && (s[next] == "{" || s[next] == "%" || s[next] == "#") {
                return i
            }
        }
        i = s.index(after: i)
    }
    return nil
}

/// Find closing tag `}}` or `%}` or `#}`.
private func findClose(_ tag: String, in s: Substring) -> Substring.Index? {
    let chars = Array(tag)
    var i = s.startIndex
    while i < s.endIndex {
        let next = s.index(after: i)
        if next < s.endIndex && s[i] == chars[0] && s[next] == chars[1] {
            return i
        }
        i = s.index(after: i)
    }
    return nil
}

// MARK: - Expression Parser

/// Parses an expression string into an Expression AST node.
private func parseExpression(_ input: String) throws -> Expression {
    var parser = ExpressionParser(input)
    let expr = try parser.parseOr()
    return expr
}

/// A recursive descent parser for Jinja2 expressions.
private struct ExpressionParser {
    private var tokens: [ExprToken]
    private var pos: Int

    init(_ input: String) {
        self.tokens = ExpressionParser.tokenize(input)
        self.pos = 0
    }

    private enum ExprToken: Equatable, Sendable {
        case string(String)
        case int(Int)
        case identifier(String)
        case dot
        case lBracket
        case rBracket
        case lParen
        case rParen
        case plus
        case eq        // ==
        case neq       // !=
        case pipe
        case comma
        case assign    // = (single)
    }

    private static func tokenize(_ input: String) -> [ExprToken] {
        var tokens: [ExprToken] = []
        let chars = Array(input)
        var i = 0

        while i < chars.count {
            let c = chars[i]

            if c.isWhitespace {
                i += 1
                continue
            }

            // String literal
            if c == "'" || c == "\"" {
                let quote = c
                i += 1
                var s = ""
                while i < chars.count && chars[i] != quote {
                    if chars[i] == "\\" && i + 1 < chars.count {
                        let escaped = chars[i + 1]
                        switch escaped {
                        case "n": s.append("\n")
                        case "t": s.append("\t")
                        case "\\": s.append("\\")
                        case "'": s.append("'")
                        case "\"": s.append("\"")
                        default:
                            s.append("\\")
                            s.append(escaped)
                        }
                        i += 2
                    } else {
                        s.append(chars[i])
                        i += 1
                    }
                }
                i += 1  // skip closing quote
                tokens.append(.string(s))
                continue
            }

            // Number literal
            if c.isNumber {
                var numStr = String(c)
                i += 1
                while i < chars.count && chars[i].isNumber {
                    numStr.append(chars[i])
                    i += 1
                }
                tokens.append(.int(Int(numStr)!))
                continue
            }

            // Identifier or keyword
            if c.isLetter || c == "_" {
                var ident = String(c)
                i += 1
                while i < chars.count && (chars[i].isLetter || chars[i].isNumber || chars[i] == "_") {
                    ident.append(chars[i])
                    i += 1
                }
                tokens.append(.identifier(ident))
                continue
            }

            // Two-char operators
            if c == "=" && i + 1 < chars.count && chars[i + 1] == "=" {
                tokens.append(.eq)
                i += 2
                continue
            }
            if c == "!" && i + 1 < chars.count && chars[i + 1] == "=" {
                tokens.append(.neq)
                i += 2
                continue
            }

            // Single-char tokens
            switch c {
            case ".": tokens.append(.dot); i += 1
            case "[": tokens.append(.lBracket); i += 1
            case "]": tokens.append(.rBracket); i += 1
            case "(": tokens.append(.lParen); i += 1
            case ")": tokens.append(.rParen); i += 1
            case "+": tokens.append(.plus); i += 1
            case "|": tokens.append(.pipe); i += 1
            case ",": tokens.append(.comma); i += 1
            case "=": tokens.append(.assign); i += 1
            default:
                i += 1  // skip unknown chars
            }
        }

        return tokens
    }

    private var current: ExprToken? {
        pos < tokens.count ? tokens[pos] : nil
    }

    private mutating func advance() {
        pos += 1
    }

    private func peek() -> ExprToken? {
        pos < tokens.count ? tokens[pos] : nil
    }

    // Precedence: or < and < not < comparison < addition < primary
    mutating func parseOr() throws -> Expression {
        var left = try parseAnd()
        while case .identifier("or") = current {
            advance()
            let right = try parseAnd()
            left = .binaryOp("or", left, right)
        }
        return left
    }

    private mutating func parseAnd() throws -> Expression {
        var left = try parseNot()
        while case .identifier("and") = current {
            advance()
            let right = try parseNot()
            left = .binaryOp("and", left, right)
        }
        return left
    }

    private mutating func parseNot() throws -> Expression {
        if case .identifier("not") = current {
            advance()
            let operand = try parseNot()
            return .unaryOp("not", operand)
        }
        return try parseComparison()
    }

    private mutating func parseComparison() throws -> Expression {
        var left = try parseIn()

        while true {
            if case .eq = current {
                advance()
                let right = try parseIn()
                left = .binaryOp("==", left, right)
            } else if case .neq = current {
                advance()
                let right = try parseIn()
                left = .binaryOp("!=", left, right)
            } else {
                break
            }
        }
        return left
    }

    private mutating func parseIn() throws -> Expression {
        let left = try parseAddition()
        if case .identifier("in") = current {
            advance()
            let right = try parseAddition()
            return .binaryOp("in", left, right)
        }
        // "not in"
        if case .identifier("not") = current {
            let savedPos = pos
            advance()
            if case .identifier("in") = current {
                advance()
                let right = try parseAddition()
                return .unaryOp("not", .binaryOp("in", left, right))
            } else {
                pos = savedPos
            }
        }
        return left
    }

    private mutating func parseAddition() throws -> Expression {
        var left = try parsePrimary()
        while case .plus = current {
            advance()
            let right = try parsePrimary()
            left = .binaryOp("+", left, right)
        }
        return left
    }

    private mutating func parsePrimary() throws -> Expression {
        guard let token = current else {
            throw ChatTemplateError.parseError("Unexpected end of expression")
        }

        var expr: Expression

        switch token {
        case .string(let s):
            advance()
            expr = .stringLiteral(s)

        case .int(let n):
            advance()
            expr = .intLiteral(n)

        case .identifier("true"):
            advance()
            expr = .boolLiteral(true)

        case .identifier("false"):
            advance()
            expr = .boolLiteral(false)

        case .identifier("none"), .identifier("None"):
            advance()
            expr = .variable("__none__")

        case .identifier(let name):
            advance()
            expr = .variable(name)

        case .lParen:
            advance()
            expr = try parseOr()
            if case .rParen = current {
                advance()
            }

        default:
            throw ChatTemplateError.parseError("Unexpected token in expression")
        }

        // Postfix: member access via `.` or `[...]`, and filter via `|`
        while true {
            if case .dot = current {
                advance()
                if case .identifier(let member) = current {
                    advance()
                    expr = .memberAccess(expr, member)
                } else {
                    throw ChatTemplateError.parseError("Expected identifier after '.'")
                }
            } else if case .lBracket = current {
                advance()
                let indexExpr = try parseOr()
                // We need the string value for member access
                if case .rBracket = current {
                    advance()
                }
                // Convert bracket access to member access if the index is a string literal
                if case .stringLiteral(let key) = indexExpr {
                    expr = .memberAccess(expr, key)
                } else {
                    // For dynamic indexing, use a binary op
                    expr = .binaryOp("[]", expr, indexExpr)
                }
            } else if case .pipe = current {
                advance()
                if case .identifier(let filterName) = current {
                    advance()
                    expr = .filter(expr, filterName)
                } else {
                    throw ChatTemplateError.parseError("Expected filter name after '|'")
                }
            } else {
                break
            }
        }

        return expr
    }
}

// MARK: - Parser (Blocks -> AST)

/// Parses raw blocks into an AST of Nodes.
private func parseBlocks(_ blocks: [RawBlock]) throws -> [Node] {
    // First, apply whitespace stripping to text blocks
    var processed: [RawBlock] = blocks
    applyWhitespaceStripping(&processed)

    var index = 0
    return try parseNodeList(&index, in: processed, until: nil)
}

/// Apply whitespace stripping based on `{%-` and `-%}` markers.
private func applyWhitespaceStripping(_ blocks: inout [RawBlock]) {
    for i in 0..<blocks.count {
        // If this block has leftStrip, strip trailing whitespace from preceding text
        if blocks[i].leftStrip && i > 0 {
            if case .text(let t) = blocks[i - 1].kind {
                let stripped = t.replacingOccurrences(
                    of: "[ \\t]*\\n?$",
                    with: "",
                    options: .regularExpression
                )
                blocks[i - 1] = RawBlock(
                    kind: .text(stripped),
                    leftStrip: blocks[i - 1].leftStrip,
                    rightStrip: blocks[i - 1].rightStrip
                )
            }
        }
        // If this block has rightStrip, strip leading whitespace from following text
        if blocks[i].rightStrip && i + 1 < blocks.count {
            if case .text(let t) = blocks[i + 1].kind {
                let stripped = t.replacingOccurrences(
                    of: "^\\n?[ \\t]*",
                    with: "",
                    options: .regularExpression
                )
                blocks[i + 1] = RawBlock(
                    kind: .text(stripped),
                    leftStrip: blocks[i + 1].leftStrip,
                    rightStrip: blocks[i + 1].rightStrip
                )
            }
        }
    }
}

/// Recursively parse a list of nodes until we hit an end tag.
private func parseNodeList(
    _ index: inout Int,
    in blocks: [RawBlock],
    until endTags: Set<String>?
) throws -> [Node] {
    var nodes: [Node] = []

    while index < blocks.count {
        let block = blocks[index]

        switch block.kind {
        case .text(let text):
            if !text.isEmpty {
                nodes.append(.text(text))
            }
            index += 1

        case .expression(let content):
            let expr = try parseExpression(content)
            nodes.append(.output(expr))
            index += 1

        case .statement(let content):
            let parts = splitStatementParts(content)
            guard let keyword = parts.first else {
                index += 1
                continue
            }

            // Check if this is an end tag we're looking for
            if let endTags, endTags.contains(keyword) {
                return nodes
            }

            switch keyword {
            case "for":
                let (varName, iterableExpr) = try parseForStatement(parts)
                index += 1
                let body = try parseNodeList(&index, in: blocks, until: ["endfor"])
                index += 1  // skip endfor
                nodes.append(.forLoop(variable: varName, iterable: iterableExpr, body: body))

            case "if":
                let condition = try parseExpression(parts.dropFirst().joined(separator: " "))
                index += 1
                var branches: [(condition: Expression?, body: [Node])] = []

                let ifBody = try parseNodeList(&index, in: blocks, until: ["elif", "else", "endif"])
                branches.append((condition: condition, body: ifBody))

                // Handle elif / else chains
                while index < blocks.count {
                    if case .statement(let nextContent) = blocks[index].kind {
                        let nextParts = splitStatementParts(nextContent)
                        if nextParts.first == "elif" {
                            let elifCondition = try parseExpression(nextParts.dropFirst().joined(separator: " "))
                            index += 1
                            let elifBody = try parseNodeList(&index, in: blocks, until: ["elif", "else", "endif"])
                            branches.append((condition: elifCondition, body: elifBody))
                        } else if nextParts.first == "else" {
                            index += 1
                            let elseBody = try parseNodeList(&index, in: blocks, until: ["endif"])
                            branches.append((condition: nil, body: elseBody))
                        } else if nextParts.first == "endif" {
                            index += 1
                            break
                        } else {
                            break
                        }
                    } else {
                        break
                    }
                }

                nodes.append(.ifBlock(branches: branches))

            case "set":
                let (varName, valueExpr) = try parseSetStatement(parts)
                nodes.append(.setVar(variable: varName, value: valueExpr))
                index += 1

            default:
                throw ChatTemplateError.unsupportedFeature("Unsupported template tag: \(keyword)")
            }

        case .comment:
            index += 1
        }
    }

    return nodes
}

/// Split a statement's content into whitespace-separated parts, respecting strings.
private func splitStatementParts(_ content: String) -> [String] {
    var parts: [String] = []
    var current = ""
    var inString: Character? = nil
    var escaped = false

    for c in content {
        if escaped {
            current.append(c)
            escaped = false
            continue
        }
        if c == "\\" {
            current.append(c)
            escaped = true
            continue
        }
        if let q = inString {
            current.append(c)
            if c == q { inString = nil }
            continue
        }
        if c == "'" || c == "\"" {
            current.append(c)
            inString = c
            continue
        }
        if c.isWhitespace {
            if !current.isEmpty {
                parts.append(current)
                current = ""
            }
            continue
        }
        current.append(c)
    }
    if !current.isEmpty {
        parts.append(current)
    }
    return parts
}

/// Parse `for variable in iterable` statement.
private func parseForStatement(_ parts: [String]) throws -> (String, Expression) {
    // Expected: ["for", varName, "in", ...rest...]
    guard parts.count >= 4, parts[2] == "in" else {
        throw ChatTemplateError.parseError("Invalid for statement")
    }
    let varName = parts[1]
    let iterableStr = parts[3...].joined(separator: " ")
    let iterable = try parseExpression(iterableStr)
    return (varName, iterable)
}

/// Parse `set variable = expression` statement.
private func parseSetStatement(_ parts: [String]) throws -> (String, Expression) {
    // Expected: ["set", varName, "=", ...rest...]
    guard parts.count >= 4, parts[2] == "=" else {
        throw ChatTemplateError.parseError("Invalid set statement: \(parts.joined(separator: " "))")
    }
    let varName = parts[1]
    let valueStr = parts[3...].joined(separator: " ")
    let value = try parseExpression(valueStr)
    return (varName, value)
}

// MARK: - Evaluator

/// Evaluation context holding variables.
private struct EvalContext {
    var variables: [String: TemplateValue]

    init(_ variables: [String: TemplateValue] = [:]) {
        self.variables = variables
    }

    func with(variable name: String, value: TemplateValue) -> EvalContext {
        var copy = self
        copy.variables[name] = value
        return copy
    }

    mutating func set(variable name: String, value: TemplateValue) {
        variables[name] = value
    }
}

/// Evaluate a list of AST nodes to produce output text.
private func evaluateNodes(_ nodes: [Node], context: inout EvalContext) throws -> String {
    var output = ""
    for node in nodes {
        switch node {
        case .text(let text):
            output += text

        case .output(let expr):
            let value = try evaluateExpression(expr, context: context)
            output += value.asString

        case .forLoop(let variable, let iterable, let body):
            let iterableValue = try evaluateExpression(iterable, context: context)
            guard case .array(let items) = iterableValue else {
                throw ChatTemplateError.evaluationError("For loop iterable is not an array")
            }
            let count = items.count
            for (idx, item) in items.enumerated() {
                var loopContext = context
                loopContext.variables[variable] = item
                loopContext.variables["loop"] = .dict([
                    "index": .int(idx + 1),
                    "index0": .int(idx),
                    "first": .bool(idx == 0),
                    "last": .bool(idx == count - 1),
                    "length": .int(count),
                ])
                output += try evaluateNodes(body, context: &loopContext)
            }

        case .ifBlock(let branches):
            for branch in branches {
                if let condition = branch.condition {
                    let value = try evaluateExpression(condition, context: context)
                    if value.isTruthy {
                        output += try evaluateNodes(branch.body, context: &context)
                        break
                    }
                } else {
                    // else branch
                    output += try evaluateNodes(branch.body, context: &context)
                    break
                }
            }

        case .setVar(let variable, let value):
            let evaluated = try evaluateExpression(value, context: context)
            context.set(variable: variable, value: evaluated)
        }
    }
    return output
}

/// Evaluate an expression in the given context.
private func evaluateExpression(_ expr: Expression, context: EvalContext) throws -> TemplateValue {
    switch expr {
    case .stringLiteral(let s):
        return .string(s)

    case .intLiteral(let n):
        return .int(n)

    case .boolLiteral(let b):
        return .bool(b)

    case .variable(let name):
        if name == "__none__" { return .none }
        if let value = context.variables[name] {
            return value
        }
        return .none

    case .memberAccess(let object, let member):
        let objectValue = try evaluateExpression(object, context: context)
        switch objectValue {
        case .dict(let dict):
            return dict[member] ?? .none
        default:
            return .none
        }

    case .binaryOp(let op, let left, let right):
        let leftVal = try evaluateExpression(left, context: context)
        let rightVal = try evaluateExpression(right, context: context)

        switch op {
        case "+":
            // String concatenation or integer addition
            switch (leftVal, rightVal) {
            case (.string(let l), .string(let r)):
                return .string(l + r)
            case (.int(let l), .int(let r)):
                return .int(l + r)
            case (.string(let l), .int(let r)):
                return .string(l + String(r))
            case (.int(let l), .string(let r)):
                return .string(String(l) + r)
            default:
                return .string(leftVal.asString + rightVal.asString)
            }

        case "==":
            return .bool(valuesEqual(leftVal, rightVal))

        case "!=":
            return .bool(!valuesEqual(leftVal, rightVal))

        case "and":
            return .bool(leftVal.isTruthy && rightVal.isTruthy)

        case "or":
            return leftVal.isTruthy ? leftVal : rightVal

        case "in":
            if case .string(let needle) = leftVal, case .string(let haystack) = rightVal {
                return .bool(haystack.contains(needle))
            }
            if case .array(let arr) = rightVal {
                for item in arr {
                    if valuesEqual(leftVal, item) {
                        return .bool(true)
                    }
                }
                return .bool(false)
            }
            return .bool(false)

        case "[]":
            // Dynamic bracket access
            if case .dict(let dict) = leftVal {
                return dict[rightVal.asString] ?? .none
            }
            if case .array(let arr) = leftVal, case .int(let idx) = rightVal {
                if idx >= 0 && idx < arr.count {
                    return arr[idx]
                }
            }
            return .none

        default:
            throw ChatTemplateError.evaluationError("Unknown operator: \(op)")
        }

    case .unaryOp(let op, let operand):
        let value = try evaluateExpression(operand, context: context)
        switch op {
        case "not":
            return .bool(!value.isTruthy)
        default:
            throw ChatTemplateError.evaluationError("Unknown unary operator: \(op)")
        }

    case .filter(let expression, let name):
        let value = try evaluateExpression(expression, context: context)
        switch name {
        case "trim":
            return .string(value.asString.trimmingCharacters(in: .whitespacesAndNewlines))
        case "lower":
            return .string(value.asString.lowercased())
        case "upper":
            return .string(value.asString.uppercased())
        case "length":
            switch value {
            case .string(let s): return .int(s.count)
            case .array(let a): return .int(a.count)
            default: return .int(0)
            }
        case "first":
            if case .array(let a) = value, let f = a.first {
                return f
            }
            return .none
        case "last":
            if case .array(let a) = value, let l = a.last {
                return l
            }
            return .none
        default:
            throw ChatTemplateError.evaluationError("Unknown filter: \(name)")
        }
    }
}

/// Compare two template values for equality.
private func valuesEqual(_ a: TemplateValue, _ b: TemplateValue) -> Bool {
    switch (a, b) {
    case (.string(let l), .string(let r)): l == r
    case (.int(let l), .int(let r)): l == r
    case (.bool(let l), .bool(let r)): l == r
    case (.none, .none): true
    default: false
    }
}

// MARK: - Public API

/// A minimal Jinja2 template engine for formatting chat messages.
///
/// Supports the subset of Jinja2 needed for chat templates found in GGUF model files:
/// `for` loops, `if`/`elif`/`else` conditionals, `set` variables, string concatenation,
/// comparisons, boolean operators, member access, and basic filters.
///
/// The engine is `Sendable` — the AST is immutable after initialization,
/// and `apply()` is a pure function.
public struct ChatTemplateEngine: Sendable {
    private let nodes: [Node]

    /// Parse a Jinja2 chat template at initialization time.
    /// - Parameter template: The raw Jinja2 template string.
    /// - Throws: `ChatTemplateError` if parsing fails or an unsupported feature is encountered.
    public init(template: String) throws {
        let blocks = lexTemplate(template)
        self.nodes = try parseBlocks(blocks)
    }

    /// Apply the template to the given messages and options.
    /// - Parameters:
    ///   - messages: The chat messages to format.
    ///   - addGenerationPrompt: Whether to include the generation prompt suffix.
    ///   - bosToken: Optional beginning-of-sequence token.
    ///   - eosToken: Optional end-of-sequence token.
    ///   - tools: Optional tool definitions for function-calling templates.
    /// - Returns: The formatted chat string.
    /// - Throws: `ChatTemplateError` if evaluation fails.
    public func apply(
        messages: [ChatMessage],
        addGenerationPrompt: Bool = true,
        bosToken: String? = nil,
        eosToken: String? = nil,
        tools: [ToolDefinition]? = nil
    ) throws -> String {
        // Build context
        let messageDicts: [TemplateValue] = messages.map { msg in
            .dict([
                "role": .string(msg.role),
                "content": .string(msg.content),
            ])
        }

        var variables: [String: TemplateValue] = [
            "messages": .array(messageDicts),
            "add_generation_prompt": .bool(addGenerationPrompt),
        ]

        if let bosToken {
            variables["bos_token"] = .string(bosToken)
        }
        if let eosToken {
            variables["eos_token"] = .string(eosToken)
        }

        if let tools {
            let toolDicts: [TemplateValue] = tools.map { tool in
                .dict([
                    "name": .string(tool.name),
                    "description": .string(tool.description),
                    "parameters": .string(tool.parametersJSON),
                ])
            }
            variables["tools"] = .array(toolDicts)
        }

        var context = EvalContext(variables)
        return try evaluateNodes(nodes, context: &context)
    }
}
