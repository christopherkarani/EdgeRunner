import Foundation

public enum PreTokenizerPattern: Sendable {
    public static func resolve(_ preTokenizerName: String?) -> PreTokenizer {
        switch preTokenizerName?.lowercased() {
        case nil, "default", "gpt-2", "granite-docling":
            return gpt2()
        case "qwen2":
            return qwen2()
        case "llama3", "llama-v3", "llama4":
            return llama3()
        case "tekken":
            return tekken()
        case "starcoder", "command-r", "refact", "smollm", "codeshell", "exaone":
            return starcoder()
        case "deepseek-llm":
            return deepseekLLM()
        case "deepseek-coder":
            return deepseekCoder()
        case "chatglm-bpe":
            return gpt2()
        case "viking":
            return gpt2()
        default:
            return gpt2()
        }
    }

    private static func gpt2() -> RegexPreTokenizer {
        RegexPreTokenizer(pattern: try! Regex(
            #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        ))
    }

    private static func qwen2() -> RegexPreTokenizer {
        RegexPreTokenizer(pattern: try! Regex(
            #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        ))
    }

    private static func llama3() -> RegexPreTokenizer {
        RegexPreTokenizer(pattern: try! Regex(
            #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        ))
    }

    private static func tekken() -> RegexPreTokenizer {
        RegexPreTokenizer(pattern: try! Regex(
            #"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        ))
    }

    private static func starcoder() -> RegexPreTokenizer {
        RegexPreTokenizer(pattern: try! Regex(
            #"\p{N}|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        ))
    }

    private static func deepseekLLM() -> RegexPreTokenizer {
        RegexPreTokenizer(patterns: [
            try! Regex(#"[\r\n]|\s?\p{L}+|\s?\p{N}+|\s?[^\s\p{L}\p{N}]+|[一-龥ࠀ-一가-퟿]+"#)
        ])
    }

    private static func deepseekCoder() -> RegexPreTokenizer {
        RegexPreTokenizer(patterns: [
            try! Regex(#"[\r\n]|\s?\p{L}+|\s?\p{P}+|[一-龥ࠀ-一가-퟿]+|\p{N}"#)
        ])
    }
}
