# Codex Autoresearch Integration Plan

## Objective
Enable a persistent, autonomous optimization loop within the Codex environment by binding the `run_loop.sh` experiment runner to an agent-driven process.

## Steps for Integration

### 1. Update Agent Persona (`.claude/agents/autoresearch.md`)
Extend the agent's logic to treat `./autoresearch/run_loop.sh` as the primary benchmarking engine.
- Instruct the agent to:
    - Periodically inspect `benchmarks/logs/results.json`.
    - Propose new parameter sets (via environment variables or config file updates).
    - Trigger `run_loop.sh` with the new configurations.
    - Commit results (or revert) based on throughput gains.

### 2. Configure Environment (`.claude/settings.local.json`)
Ensure the agent has appropriate permissions to execute the loop without user intervention for every step:
```json
{
  "permissions": {
    "run_shell_command": ["./autoresearch/run_loop.sh", "swift build", "swift test", "git commit", "git revert"],
    "read_file": ["benchmarks/logs/results.json", "Sources/EdgeRunnerMetal/*"],
    "write_file": ["benchmarks/logs/results.json"]
  }
}
```

### 3. Loop Orchestration
- Use a screen-based or persistent task runner:
  ```bash
  # Inside Codex session or shell
  screen -S autoresearch ./autoresearch/run_loop.sh 1000
  ```
- The agent will continuously monitor the output logs and the repository state to manage the experiments.

## Validation
- Monitor the agent's activity log for successful execution of `run_loop.sh`.
- Check `benchmarks/logs/results.json` for consistent throughput improvements over time.
- Verify `git` state for successful optimization commits.
