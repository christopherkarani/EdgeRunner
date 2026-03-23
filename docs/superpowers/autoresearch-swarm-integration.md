# Autoresearch Swarm Integration Plan

## Overview

Multi-agent swarm for autonomous decode throughput optimization on EdgeRunner.

**Goal:** Achieve publishable benchmark (>250 tok/s, target >280 tok/s) through rigorous scientific experimentation.

**Approach:** 6 specialized agents working in parallel, coordinated by Orchestrator.

---

## Agent Swarm

| Agent | File | Role |
|-------|------|------|
| Orchestrator | `autoresearch.md` | Coordinates swarm, dispatches tasks, manages iterations |
| Researcher | `autoresearch-researcher.md` | Deep research: whitepapers, code analysis |
| Designer | `autoresearch-designer.md` | Forms hypotheses, designs experiments |
| Implementer | `autoresearch-implementer.md` | Surgical code changes |
| Benchmarker | `autoresearch-benchmarker.md` | Statistical benchmarking |
| Analyst | `autoresearch-analyst.md` | Keep/rollback decisions |
| Logger | `autoresearch-logger.md` | Records all experiments |

---

## Workflow

### Phase 1: Research (Parallel)
```
Orchestrator ─┬─► Researcher (whitepapers)
              ├─► Researcher (llama.cpp analysis)
              └─► Researcher (MLX analysis)
```

### Phase 2: Design
```
Research Findings ──► Designer ──► 3 Hypotheses
```

### Phase 3: Execute (Per Experiment)
```
Hypothesis ──► Implementer ──► Benchmarker ──► Analyst ──► Logger
                  │                │              │           │
                  └────────────────┴──────────────┴───────────┘
                                    │
                              Commit/Branch
                                    │
                           Breakthrough? ──► Merge to main
```

---

## Key Principles

### 1. Commit Often for Safety
```bash
# Every experiment gets a branch
git checkout -b "autoresearch/exp-${n}-${name}"

# Every attempt committed
git commit -am "exp(${n}): ${hypothesis}"
git push -u origin "autoresearch/exp-${n}-${name}"
```

### 2. Only Commit Breakthroughs to Main
- Breakthrough: >3% improvement, p<0.05, perfect correctness
- Small gains: Kept on branch for cumulative strategy
- Failures: Logged, branch may be deleted

### 3. Log Everything
- Database: `benchmarks/experiment_database.json`
- Narrative: `benchmarks/experiment_log.md`
- Git history: Every attempt preserved

### 4. Scientific Rigor
- Falsifiable hypotheses
- Quantified predictions
- Statistical significance testing
- Controlled experiments (one variable)

---

## Starting the Swarm

### Option 1: Direct Orchestrator Launch
```bash
# In Claude Code
Agent("autoresearch") {
  "Begin optimization swarm. Current baseline: 228 tok/s.
   Dispatch researchers to find optimization opportunities."
}
```

### Option 2: Step-by-Step
```bash
# 1. Verify environment
swift build -c release
swift test --filter "PublishableBenchmark/fullBenchmark" 2>&1 | tail -20

# 2. Launch orchestrator
# (Agent dispatch from main Claude session)

# 3. Monitor progress
watch -n 30 'cat benchmarks/experiment_log.md | tail -50'
```

---

## Monitoring Progress

### Real-time Logs
```bash
# Watch experiment log
tail -f benchmarks/experiment_log.md

# Watch database updates
watch -n 10 'jq .metadata benchmarks/experiment_database.json'

# See active branches
git branch -a | grep autoresearch
```

### Progress Dashboard
```bash
# Count experiments
cat benchmarks/experiment_database.json | jq '.experiments | length'

# Breakthroughs
cat benchmarks/experiment_database.json | jq '.metadata.breakthrough_count'

# Current baseline
cat benchmarks/experiment_database.json | jq '.metadata.current_baseline'
```

---

## Breakthrough Criteria

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| Improvement | >3% | REQUIRED |
| Statistical Sig | p < 0.05 | REQUIRED |
| Correctness | All pass | REQUIRED |
| Memory | <5% increase | REQUIRED |

**On Breakthrough:**
```bash
git checkout main
git merge "autoresearch/exp-${n}-${name}"
git tag "breakthrough-${n}-${improvement}pct"
git push origin main --tags
```

---

## Safety Limits

STOP and escalate if:
- 10 consecutive experiments with no improvement
- Correctness check fails 3 times in a row
- Baseline regressed >10%
- No breakthrough after 50 experiments

---

## Target: Publishable Result

**Claimable:** >250 tok/s (+10% from baseline)
**World-Class:** >280 tok/s (+25% from baseline)
**Current:** 228 tok/s

Documented methodology, reproducible results, open source.

---

## Files Changed

| File | Purpose |
|------|---------|
| `.claude/agents/autoresearch.md` | Orchestrator agent |
| `.claude/agents/autoresearch-researcher.md` | Research specialist |
| `.claude/agents/autoresearch-designer.md` | Experiment designer |
| `.claude/agents/autoresearch-implementer.md` | Implementation specialist |
| `.claude/agents/autoresearch-benchmarker.md` | Benchmark specialist |
| `.claude/agents/autoresearch-analyst.md` | Decision analyst |
| `.claude/agents/autoresearch-logger.md` | Documentation specialist |
| `autoresearch/run_loop.sh` | Benchmark loop runner |
| `benchmarks/experiment_database.json` | Machine-readable log (created) |
| `benchmarks/experiment_log.md` | Human-readable log (created) |

---

## Next Steps

1. **Initialize database:**
   ```bash
   echo '{"metadata":{"baseline":228.1,"current_baseline":228.1,"total_experiments":0},"experiments":[],"patterns":{"what_works":[],"what_doesnt_work":[],"open_questions":[]}}' > benchmarks/experiment_database.json
   echo "# EdgeRunner Autoresearch Experiment Log\n\nCurrent baseline: 228 tok/s\n" > benchmarks/experiment_log.md
   ```

2. **Launch swarm:** Start orchestrator agent

3. **Monitor:** Watch logs and database

4. **Iterate:** Let swarm run autonomously

5. **Claim victory:** When breakthrough achieved, merge and tag
