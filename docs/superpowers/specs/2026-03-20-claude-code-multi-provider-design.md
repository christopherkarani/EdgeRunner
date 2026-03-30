# Claude Code Multi-Provider Setup: Kimi K2.5 + MiniMax M2.7

## Summary

Two launch modes for Claude Code via shell functions:
- `claude` — direct to Anthropic (Opus/Sonnet/Haiku, no proxy)
- `claude-kimi` — through `claude-code-mux` proxy (Kimi K2.5 as main/think agent, MiniMax M2.7 for default + background tasks)

## Architecture

```
Mode 1: claude (normal)
  You → Claude Code → Anthropic API
  └── Opus / Sonnet / Haiku (native switching via /model)

Mode 2: claude-kimi (multi-provider)
  You → Claude Code → claude-code-mux (localhost:13456)
        ├── default tasks   → MiniMax M2.7   (api.minimax.io/anthropic)
        ├── think/plan mode → Kimi K2.5      (api.moonshot.ai/anthropic)
        ├── background tasks → MiniMax M2.7  (api.minimax.io/anthropic)
        └── auto-map: all claude-* model names get routed through above
```

## How claude-code-mux Routing Works

The proxy intercepts ALL Claude Code requests and routes them based on **task type**, not model name slots. The routing flow is:

```
Auto-map (regex transform) → WebSearch > Subagent > Think > Background > Default
```

1. **Auto-mapping**: Any request with model name matching `^claude-` gets transformed to the default model
2. **Think mode**: If request has `thinking.type = "enabled"` (triggered by `/model opus` or plan mode), routes to Think model
3. **Background**: If ORIGINAL model name matches `(?i)claude.*haiku`, routes to Background model
4. **Default**: Everything else goes to Default model

This means:
- When you use `/model opus` or plan mode → Kimi K2.5 (think model)
- When Claude Code spawns subagents (usually haiku) → MiniMax M2.7 (background model)
- Normal coding work → MiniMax M2.7 (default model)

## Components

### 1. claude-code-mux (Rust proxy)

- **What:** Lightweight routing proxy (~5MB RAM, <1ms overhead)
- **Version:** v0.6.3 (latest)
- **Config:** `~/.claude-code-mux/config.toml` (auto-created on first start)
- **Port:** 13456
- **Admin UI:** http://127.0.0.1:13456

**Install (macOS Apple Silicon):**
```bash
curl -L https://github.com/9j/claude-code-mux/releases/latest/download/ccm-macos-aarch64.tar.gz | tar xz
sudo mv ccm /usr/local/bin/
ccm --version  # verify
```

**Install (macOS Intel):**
```bash
curl -L https://github.com/9j/claude-code-mux/releases/latest/download/ccm-macos-x86_64.tar.gz | tar xz
sudo mv ccm /usr/local/bin/
```

**Install (via Cargo):**
```bash
cargo install claude-code-mux
```

### 2. Provider Details

#### Kimi K2.5 (Moonshot AI) — Think/Reasoning Model
- **Provider type in ccm:** `kimi`
- **Anthropic-compatible base URL:** `https://api.moonshot.ai/anthropic`
- **Model name:** `kimi-k2.5`
- **API key:** from https://platform.moonshot.ai/console
- **Strengths:** 256K context, thinking/reasoning mode, strong coding

#### MiniMax M2.7 — Default/Background Model
- **Provider type in ccm:** `minimax`
- **Anthropic-compatible base URL:** `https://api.minimax.io/anthropic`
- **Model name:** `MiniMax-M2.7`
- **API key:** from MiniMax Developer Platform
- **Strengths:** Ultra-fast, ultra-cheap ($0.30/$1.20 per M tokens)

### 3. Configuration via Admin UI (Recommended)

After `ccm start`, open http://127.0.0.1:13456 and configure:

**Providers tab — Add two providers:**

| Field | Kimi Provider | MiniMax Provider |
|-------|--------------|-----------------|
| Provider Type | Kimi | Minimax |
| Name | `kimi` | `minimax` |
| API Key | Your Moonshot API key | Your MiniMax API key |

**Models tab — Add two models:**

| Model Name | Provider | Actual Model | Priority |
|------------|----------|-------------|----------|
| `kimi-k2.5` | `kimi` | `kimi-k2.5` | 1 |
| `minimax-m2.7` | `minimax` | `MiniMax-M2.7` | 1 |

**Router tab — Configure routing:**

| Setting | Value |
|---------|-------|
| Default Model | `minimax-m2.7` |
| Think Model | `kimi-k2.5` |
| Background Model | `minimax-m2.7` |
| WebSearch Model | `kimi-k2.5` |
| Auto-map Regex | `^claude-` |
| Background Regex | `(?i)claude.*haiku` |

### 4. Shell Function (~/.zshrc)

```zsh
# --- Claude Code Multi-Provider ---

# Normal Claude — direct to Anthropic, no proxy, zero overhead
# Just use: claude

# Kimi + MiniMax mode — through claude-code-mux
claude-kimi() {
  # Check ccm is installed
  if ! command -v ccm > /dev/null 2>&1; then
    echo "Error: ccm (claude-code-mux) not installed."
    echo "Install: curl -L https://github.com/9j/claude-code-mux/releases/latest/download/ccm-macos-aarch64.tar.gz | tar xz && sudo mv ccm /usr/local/bin/"
    return 1
  fi

  # Start mux if not already running
  if ! ccm status > /dev/null 2>&1; then
    echo "Starting claude-code-mux..."
    if ! ccm start; then
      echo "Error: Failed to start claude-code-mux" >&2
      return 1
    fi
    # Wait for proxy to be ready
    local retries=0
    while ! ccm status > /dev/null 2>&1 && [ $retries -lt 5 ]; do
      sleep 1
      retries=$((retries + 1))
    done
    if [ $retries -eq 5 ]; then
      echo "Error: claude-code-mux failed to start within 5 seconds" >&2
      return 1
    fi
    echo "claude-code-mux ready. Kimi K2.5 (think) + MiniMax M2.7 (default)"
  fi

  ANTHROPIC_BASE_URL="http://127.0.0.1:13456" \
  ANTHROPIC_API_KEY="mux-passthrough" \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  claude "$@"
}

# Stop the proxy
alias ccm-stop='ccm stop'
```

### 5. Usage

```bash
# --- Normal Claude (Anthropic models) ---
claude
# /model opus, /model sonnet, /model haiku — all Anthropic, works as usual

# --- Kimi + MiniMax mode ---
claude-kimi
# Normal coding → MiniMax M2.7 (fast, cheap)
# /model opus or plan mode → Kimi K2.5 (deep thinking)
# Subagents (haiku) → MiniMax M2.7 (background)
# Web search tasks → Kimi K2.5

# --- Stop proxy when done ---
ccm-stop
```

### 6. Model Behavior in claude-kimi Mode

| What you do in Claude Code | Where it goes | Why |
|---------------------------|---------------|-----|
| Normal chat/coding | MiniMax M2.7 | Default model |
| `/model opus` or plan mode | Kimi K2.5 | Think model (thinking.type=enabled) |
| Agent tool spawns subagent | MiniMax M2.7 | Background model (haiku pattern match) |
| Web search tool used | Kimi K2.5 | WebSearch model |

## Setup Checklist

- [ ] Get Kimi API key from https://platform.moonshot.ai/console
- [ ] Get MiniMax API key from MiniMax Developer Platform
- [ ] Install `ccm` binary (see install commands above)
- [ ] Verify install: `ccm --version`
- [ ] Start proxy: `ccm start`
- [ ] Open admin UI: http://127.0.0.1:13456
- [ ] Add Kimi provider (type: Kimi, name: kimi, API key)
- [ ] Add MiniMax provider (type: Minimax, name: minimax, API key)
- [ ] Add model: `kimi-k2.5` → kimi provider → `kimi-k2.5`
- [ ] Add model: `minimax-m2.7` → minimax provider → `MiniMax-M2.7`
- [ ] Configure router (default: minimax-m2.7, think: kimi-k2.5, background: minimax-m2.7)
- [ ] Set auto-map regex: `^claude-`
- [ ] Set background regex: `(?i)claude.*haiku`
- [ ] Save config in admin UI
- [ ] Test via admin UI Test tab
- [ ] Add `claude-kimi` function to `~/.zshrc`
- [ ] Source zshrc: `source ~/.zshrc`
- [ ] Test: `claude-kimi` and verify routing works

## Security Notes

- API keys are stored in `~/.claude-code-mux/config.toml` (local file, not exposed)
- The proxy runs on localhost only (127.0.0.1:13456), not network-accessible
- `ANTHROPIC_API_KEY="mux-passthrough"` — the proxy ignores this value; it uses provider-specific keys from its config
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` prevents telemetry from leaking to third-party providers

## Cost Comparison

| Provider | Input (per M tokens) | Output (per M tokens) |
|----------|---------------------|-----------------------|
| MiniMax M2.7 | $0.30 | $1.20 |
| Kimi K2.5 | ~$0.60 | ~$2.40 |
| Claude Sonnet 4.6 | $3.00 | $15.00 |
| Claude Opus 4.6 | $5.00 | $25.00 |

**With this setup:** ~90% cost reduction vs Claude for most tasks, with Kimi K2.5's strong reasoning available on demand.

## Sources

- [claude-code-mux GitHub](https://github.com/9j/claude-code-mux)
- [MiniMax Anthropic API Docs](https://platform.minimax.io/docs/api-reference/text-anthropic-api)
- [MiniMax Claude Code Guide](https://platform.minimax.io/docs/guides/text-ai-coding-tools)
- [Kimi K2.5 Claude Code Integration](https://apidog.com/blog/kimi-k2-5-claude-code-integration/)
- [Moonshot AI Platform](https://platform.moonshot.ai/)
- [Claude Code Model Configuration](https://code.claude.com/docs/en/model-config)
