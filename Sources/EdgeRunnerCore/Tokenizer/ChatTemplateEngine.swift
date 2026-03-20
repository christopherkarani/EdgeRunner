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
    case namespace([String: TemplateValue])
    case none

    var isTruthy: Bool {
        switch self {
        case .string(let s): !s.isEmpty
        case .int(let i): i != 0
        case .bool(let b): b
        case .array(let a): !a.isEmpty
        case .dict(let d): !d.isEmpty
        case .namespace(let d): !d.isEmpty
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
        case .namespace: "[namespace]"
        case .none: ""
        }
    }

    /// Serialize this value to a JSON string representation.
    var toJSON: String {
        switch self {
        case .string(let s):
            let escaped = s
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
                .replacingOccurrences(of: "\n", with: "\\n")
                .replacingOccurrences(of: "\t", with: "\\t")
            return "\"\(escaped)\""
        case .int(let i):
            return String(i)
        case .bool(let b):
            return b ? "true" : "false"
        case .array(let a):
            let items = a.map { $0.toJSON }
            return "[\(items.joined(separator: ", "))]"
        case .dict(let d):
            let pairs = d.sorted(by: { $0.key < $1.key }).map { "\"\($0.key)\": \($0.value.toJSON)" }
            return "{\(pairs.joined(separator: ", "))}"
        case .namespace(let d):
            let pairs = d.sorted(by: { $0.key < $1.key }).map { "\"\($0.key)\": \($0.value.toJSON)" }
            return "{\(pairs.joined(separator: ", "))}"
        case .none:
            return "null"
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
    case filter(Expression, String, [Expression])
    case isTest(Expression, String, Bool)       // expr, testName, negated
    case methodCall(Expression, String, [Expression])
    case functionCall(String, [(String?, Expression)])  // name, [(argLabel?, value)]
    case slice(Expression, Expression?, Expression?, Expression?)  // obj, start, end, step
    case arrayLiteral([Expression])
}

/// A node in the template AST.
private indirect enum Node: Sendable {
    case text(String)
    case output(Expression)
    case forLoop(variable: String, iterable: Expression, body: [Node])
    case ifBlock(branches: [(condition: Expression?, body: [Node])])
    case setVar(variable: String, value: Expression)
    case setMember(object: String, member: String, value: Expression)
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
        case minus
        case eq        // ==
        case neq       // !=
        case pipe
        case comma
        case assign    // = (single)
        case colon
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
            case "-": tokens.append(.minus); i += 1
            case "|": tokens.append(.pipe); i += 1
            case ",": tokens.append(.comma); i += 1
            case "=": tokens.append(.assign); i += 1
            case ":": tokens.append(.colon); i += 1
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

    private func peekAt(_ offset: Int) -> ExprToken? {
        let idx = pos + offset
        return idx < tokens.count ? tokens[idx] : nil
    }

    // Precedence: or < and < not < comparison < is-test < in < addition < unary-minus < primary
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
        var left = try parseIsTest()

        while true {
            if case .eq = current {
                advance()
                let right = try parseIsTest()
                left = .binaryOp("==", left, right)
            } else if case .neq = current {
                advance()
                let right = try parseIsTest()
                left = .binaryOp("!=", left, right)
            } else {
                break
            }
        }
        return left
    }

    /// Parse `is [not] testName` after comparison level.
    private mutating func parseIsTest() throws -> Expression {
        var left = try parseIn()

        while case .identifier("is") = current {
            advance()
            // Check for "is not"
            var negated = false
            if case .identifier("not") = current {
                negated = true
                advance()
            }
            guard case .identifier(let testName) = current else {
                throw ChatTemplateError.parseError("Expected test name after 'is'")
            }
            advance()
            left = .isTest(left, testName, negated)
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
        var left = try parseUnaryMinus()
        while true {
            if case .plus = current {
                advance()
                let right = try parseUnaryMinus()
                left = .binaryOp("+", left, right)
            } else if case .minus = current {
                advance()
                let right = try parseUnaryMinus()
                left = .binaryOp("-", left, right)
            } else {
                break
            }
        }
        return left
    }

    private mutating func parseUnaryMinus() throws -> Expression {
        if case .minus = current {
            advance()
            let operand = try parsePrimary()
            return .unaryOp("-", operand)
        }
        return try parsePrimary()
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

        case .identifier("namespace"):
            // Check if it's namespace(...) function call
            if peekAt(1) == .lParen {
                advance() // consume 'namespace'
                advance() // consume '('
                var args: [(String?, Expression)] = []
                while current != .rParen && current != nil {
                    // Parse keyword argument: name=value
                    if case .identifier(let argName) = current, peekAt(1) == .assign {
                        advance() // consume identifier
                        advance() // consume '='
                        let value = try parseOr()
                        args.append((argName, value))
                    } else {
                        let value = try parseOr()
                        args.append((nil, value))
                    }
                    if case .comma = current {
                        advance()
                    }
                }
                if case .rParen = current {
                    advance()
                }
                expr = .functionCall("namespace", args)
            } else {
                advance()
                expr = .variable("namespace")
            }

        case .identifier(let name):
            // Check if it's a function call: name(...)
            if peekAt(1) == .lParen {
                advance() // consume identifier
                advance() // consume '('
                var args: [(String?, Expression)] = []
                while current != .rParen && current != nil {
                    if case .identifier(let argName) = current, peekAt(1) == .assign {
                        advance() // consume identifier
                        advance() // consume '='
                        let value = try parseOr()
                        args.append((argName, value))
                    } else {
                        let value = try parseOr()
                        args.append((nil, value))
                    }
                    if case .comma = current {
                        advance()
                    }
                }
                if case .rParen = current {
                    advance()
                }
                expr = .functionCall(name, args)
            } else {
                advance()
                expr = .variable(name)
            }

        case .lParen:
            advance()
            expr = try parseOr()
            if case .rParen = current {
                advance()
            }

        case .lBracket:
            // Array literal: [expr, expr, ...]
            advance()
            var elements: [Expression] = []
            while current != .rBracket && current != nil {
                let element = try parseOr()
                elements.append(element)
                if case .comma = current {
                    advance()
                }
            }
            if case .rBracket = current {
                advance()
            }
            expr = .arrayLiteral(elements)

        default:
            throw ChatTemplateError.parseError("Unexpected token in expression")
        }

        // Postfix: member access via `.`, `[...]` (with slicing), filter via `|`, method calls
        while true {
            if case .dot = current {
                advance()
                if case .identifier(let member) = current {
                    advance()
                    // Check if it's a method call: expr.method(args)
                    if case .lParen = current {
                        advance()
                        var args: [Expression] = []
                        while current != .rParen && current != nil {
                            let arg = try parseOr()
                            args.append(arg)
                            if case .comma = current {
                                advance()
                            }
                        }
                        if case .rParen = current {
                            advance()
                        }
                        expr = .methodCall(expr, member, args)
                    } else {
                        expr = .memberAccess(expr, member)
                    }
                } else {
                    throw ChatTemplateError.parseError("Expected identifier after '.'")
                }
            } else if case .lBracket = current {
                advance()
                // Check for slice notation: [start:end] or [start:end:step] or [::step]
                // Detect if this is a slice by checking for colon as first or second token
                let sliceResult = try parseBracketAccess()
                expr = applyBracketResult(to: expr, result: sliceResult)
            } else if case .pipe = current {
                advance()
                if case .identifier(let filterName) = current {
                    advance()
                    // Check for filter arguments: | filterName(arg1, arg2)
                    var filterArgs: [Expression] = []
                    if case .lParen = current {
                        advance()
                        while current != .rParen && current != nil {
                            let arg = try parseOr()
                            filterArgs.append(arg)
                            if case .comma = current {
                                advance()
                            }
                        }
                        if case .rParen = current {
                            advance()
                        }
                    }
                    expr = .filter(expr, filterName, filterArgs)
                } else {
                    throw ChatTemplateError.parseError("Expected filter name after '|'")
                }
            } else {
                break
            }
        }

        return expr
    }

    /// The result of parsing a bracket access: either a single index or a slice.
    private enum BracketResult {
        case index(Expression)
        case slice(start: Expression?, end: Expression?, step: Expression?)
    }

    /// Parse the content inside `[...]`, detecting slices vs single index.
    private mutating func parseBracketAccess() throws -> BracketResult {
        // If we immediately see a colon, it's a slice with no start
        if case .colon = current {
            advance()
            return try parseSliceRemainder(start: nil)
        }

        // Parse the first expression
        let first = try parseOr()

        // If followed by colon, it's a slice
        if case .colon = current {
            advance()
            return try parseSliceRemainder(start: first)
        }

        // Otherwise, it's a single index
        if case .rBracket = current {
            advance()
        }
        return .index(first)
    }

    /// After seeing `start:`, parse the rest of the slice: `end]` or `end:step]` or `:step]` or `]`
    private mutating func parseSliceRemainder(start: Expression?) throws -> BracketResult {
        var end: Expression? = nil
        var step: Expression? = nil

        // Check if there's an end value before colon or close bracket
        if current != .colon && current != .rBracket && current != nil {
            end = try parseOr()
        }

        // Check for second colon (step)
        if case .colon = current {
            advance()
            if current != .rBracket && current != nil {
                step = try parseOr()
            }
        }

        if case .rBracket = current {
            advance()
        }

        return .slice(start: start, end: end, step: step)
    }

    /// Convert a BracketResult into the appropriate Expression.
    private func applyBracketResult(to expr: Expression, result: BracketResult) -> Expression {
        switch result {
        case .index(let indexExpr):
            // Convert bracket access to member access if the index is a string literal
            if case .stringLiteral(let key) = indexExpr {
                return .memberAccess(expr, key)
            }
            // For dynamic indexing, use a binary op
            return .binaryOp("[]", expr, indexExpr)
        case .slice(let start, let end, let step):
            return .slice(expr, start, end, step)
        }
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
                let node = try parseSetStatementNode(parts)
                nodes.append(node)
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

/// Parse `set variable = expression` or `set obj.member = expression` statement, returning a Node.
private func parseSetStatementNode(_ parts: [String]) throws -> Node {
    // Check for dot notation: set ns.member = value
    // parts[1] would be "ns.member" (splitStatementParts doesn't split on dots)
    guard parts.count >= 4, parts[2] == "=" else {
        throw ChatTemplateError.parseError("Invalid set statement: \(parts.joined(separator: " "))")
    }

    let target = parts[1]
    let valueStr = parts[3...].joined(separator: " ")
    let value = try parseExpression(valueStr)

    // Check for dot notation
    if let dotIdx = target.firstIndex(of: ".") {
        let objName = String(target[target.startIndex..<dotIdx])
        let memberName = String(target[target.index(after: dotIdx)...])
        return .setMember(object: objName, member: memberName, value: value)
    }

    return .setVar(variable: target, value: value)
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
            // Collect all namespace names from context before the loop
            let namespaceNames = context.variables.compactMap { (key, value) -> String? in
                if case .namespace = value { return key }
                return nil
            }
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
                // Propagate namespace mutations back to parent context
                for nsName in namespaceNames {
                    if let nsValue = loopContext.variables[nsName] {
                        context.variables[nsName] = nsValue
                    }
                }
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

        case .setMember(let object, let member, let value):
            let evaluated = try evaluateExpression(value, context: context)
            // Mutate namespace or dict in-place
            if let nsDict = context.variables[object] {
                switch nsDict {
                case .namespace(var dict):
                    dict[member] = evaluated
                    context.variables[object] = .namespace(dict)
                case .dict(var dict):
                    dict[member] = evaluated
                    context.variables[object] = .dict(dict)
                default:
                    throw ChatTemplateError.evaluationError("Cannot set member '\(member)' on non-dict/namespace '\(object)'")
                }
            } else {
                throw ChatTemplateError.evaluationError("Variable '\(object)' not found for member assignment")
            }
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
        case .namespace(let dict):
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

        case "-":
            switch (leftVal, rightVal) {
            case (.int(let l), .int(let r)):
                return .int(l - r)
            default:
                throw ChatTemplateError.evaluationError("Cannot subtract non-integer values")
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
            if case .namespace(let dict) = leftVal {
                return dict[rightVal.asString] ?? .none
            }
            if case .array(let arr) = leftVal, case .int(let idx) = rightVal {
                let resolvedIdx = idx < 0 ? arr.count + idx : idx
                if resolvedIdx >= 0 && resolvedIdx < arr.count {
                    return arr[resolvedIdx]
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
        case "-":
            if case .int(let n) = value {
                return .int(-n)
            }
            throw ChatTemplateError.evaluationError("Cannot negate non-integer value")
        default:
            throw ChatTemplateError.evaluationError("Unknown unary operator: \(op)")
        }

    case .filter(let expression, let name, let args):
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
        case "tojson":
            return .string(value.toJSON)
        case "join":
            guard case .array(let items) = value else {
                return value
            }
            let separator: String
            if let firstArg = args.first {
                let sepVal = try evaluateExpression(firstArg, context: context)
                separator = sepVal.asString
            } else {
                separator = ""
            }
            let strings = items.map { $0.asString }
            return .string(strings.joined(separator: separator))
        default:
            throw ChatTemplateError.evaluationError("Unknown filter: \(name)")
        }

    case .isTest(let expression, let testName, let negated):
        let value = try evaluateExpression(expression, context: context)
        let result: Bool
        switch testName {
        case "defined":
            // A value is "defined" if it's not .none AND if the variable actually
            // exists in context (not just defaulting to .none).
            result = !isNoneValue(value)
        case "string":
            if case .string = value { result = true } else { result = false }
        case "mapping":
            switch value {
            case .dict: result = true
            case .namespace: result = true
            default: result = false
            }
        case "iterable":
            switch value {
            case .array: result = true
            case .string: result = true
            default: result = false
            }
        default:
            throw ChatTemplateError.evaluationError("Unknown test: \(testName)")
        }
        return .bool(negated ? !result : result)

    case .methodCall(let receiver, let methodName, let args):
        let receiverVal = try evaluateExpression(receiver, context: context)
        switch methodName {
        case "strip":
            return .string(receiverVal.asString.trimmingCharacters(in: .whitespacesAndNewlines))
        case "split":
            let separator: String
            if let firstArg = args.first {
                let sepVal = try evaluateExpression(firstArg, context: context)
                separator = sepVal.asString
            } else {
                separator = " "
            }
            let parts = receiverVal.asString.components(separatedBy: separator)
            return .array(parts.map { .string($0) })
        case "startswith":
            guard let firstArg = args.first else {
                throw ChatTemplateError.evaluationError("startswith() requires an argument")
            }
            let prefixVal = try evaluateExpression(firstArg, context: context)
            return .bool(receiverVal.asString.hasPrefix(prefixVal.asString))
        case "endswith":
            guard let firstArg = args.first else {
                throw ChatTemplateError.evaluationError("endswith() requires an argument")
            }
            let suffixVal = try evaluateExpression(firstArg, context: context)
            return .bool(receiverVal.asString.hasSuffix(suffixVal.asString))
        default:
            throw ChatTemplateError.evaluationError("Unknown method: \(methodName)")
        }

    case .functionCall(let name, let args):
        switch name {
        case "namespace":
            var dict: [String: TemplateValue] = [:]
            for (label, valueExpr) in args {
                let val = try evaluateExpression(valueExpr, context: context)
                if let label {
                    dict[label] = val
                }
            }
            return .namespace(dict)
        default:
            throw ChatTemplateError.evaluationError("Unknown function: \(name)")
        }

    case .slice(let objectExpr, let startExpr, let endExpr, let stepExpr):
        let objectVal = try evaluateExpression(objectExpr, context: context)
        guard case .array(let arr) = objectVal else {
            throw ChatTemplateError.evaluationError("Slice can only be applied to arrays")
        }

        let count = arr.count
        let step: Int
        if let stepExpr {
            let stepVal = try evaluateExpression(stepExpr, context: context)
            if case .int(let s) = stepVal { step = s } else { step = 1 }
        } else {
            step = 1
        }

        guard step != 0 else {
            throw ChatTemplateError.evaluationError("Slice step cannot be zero")
        }

        let start: Int
        let end: Int

        if step > 0 {
            if let startExpr {
                let sv = try evaluateExpression(startExpr, context: context)
                if case .int(let s) = sv {
                    start = s < 0 ? max(count + s, 0) : min(s, count)
                } else { start = 0 }
            } else {
                start = 0
            }

            if let endExpr {
                let ev = try evaluateExpression(endExpr, context: context)
                if case .int(let e) = ev {
                    end = e < 0 ? max(count + e, 0) : min(e, count)
                } else { end = count }
            } else {
                end = count
            }

            var result: [TemplateValue] = []
            var i = start
            while i < end {
                result.append(arr[i])
                i += step
            }
            return .array(result)
        } else {
            // Negative step
            if let startExpr {
                let sv = try evaluateExpression(startExpr, context: context)
                if case .int(let s) = sv {
                    start = s < 0 ? count + s : min(s, count - 1)
                } else { start = count - 1 }
            } else {
                start = count - 1
            }

            if let endExpr {
                let ev = try evaluateExpression(endExpr, context: context)
                if case .int(let e) = ev {
                    end = e < 0 ? count + e : e
                } else { end = -1 }
            } else {
                end = -1
            }

            var result: [TemplateValue] = []
            var i = start
            while i > end {
                if i >= 0 && i < count {
                    result.append(arr[i])
                }
                i += step
            }
            return .array(result)
        }

    case .arrayLiteral(let elements):
        var values: [TemplateValue] = []
        for element in elements {
            let val = try evaluateExpression(element, context: context)
            values.append(val)
        }
        return .array(values)
    }
}

/// Check if a value is the none/undefined sentinel.
private func isNoneValue(_ value: TemplateValue) -> Bool {
    if case .none = value { return true }
    return false
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
/// comparisons, boolean operators, member access, basic filters, `tojson`, `join`,
/// `is defined/string/mapping/iterable` tests, `namespace()`, string methods,
/// array slicing, and negative indexing.
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
