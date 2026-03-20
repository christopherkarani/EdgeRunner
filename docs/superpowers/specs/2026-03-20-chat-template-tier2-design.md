# ChatTemplateEngine Tier 2 Design

**Date:** 2026-03-20
**Status:** Approved
**Scope:** Extend existing ChatTemplateEngine with Tier 2 Jinja2 features for Qwen3 tool calling and DeepSeek R1 templates

## Features

### New Filters
- `| tojson` — serialize TemplateValue to JSON string
- `| join(sep)` — join array elements with separator (filter with argument)

### Type Tests (`is` keyword)
- `is defined` / `is not defined` — variable existence check
- `is string` — type check for string values
- `is mapping` — type check for dict values
- `is iterable` — type check for array or string values

### `namespace()` Built-in
- `{% set ns = namespace(found=false) %}` — creates mutable container
- `{% set ns.found = true %}` — mutates namespace member in parent scope
- Persists across loop iterations (unlike regular `set`)

### String Methods
- `.strip()` — trim whitespace
- `.split(sep)` — split by separator
- `.startswith(prefix)` — prefix check
- `.endswith(suffix)` — suffix check

### Array Operations
- Negative indexing: `messages[-1]`
- Slice notation: `messages[1:]`, `messages[:3]`, `messages[1:3]`
- Reverse: `messages[::-1]`

### Expression Parser Extensions
- `is` keyword in comparison chain
- Method call syntax: `expr.method(args)`
- Slice syntax in bracket access: `[start:end:step]`
- Filter arguments: `| join(', ')`

## Files
- Modify: `Sources/EdgeRunnerCore/Tokenizer/ChatTemplateEngine.swift`
- Modify: `Tests/EdgeRunnerCoreTests/ChatTemplateEngineTests.swift`
