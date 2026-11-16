# Paper reproduction

These experiments evaluate Harnessᴿ. They include the reference agent and chemical analyses needed to reproduce the paper; they are not extra components of the resource management layer.

## Offline entry points

Run these commands from the repository root using the installed virtual environment.

| Command | What it checks |
| --- | --- |
| `python reproduce.py verify` | Archived primary evidence, completion counts and model fingerprints; standard library only |
| `python reproduce.py tests` | The 69 selected resource-coordination and scientific-computation tests |
| `python reproduce.py controlled --suite main` | Lightweight 20-request resource comparison |
| `python reproduce.py fixed-summary` | Reaggregate the archived fixed-call records |
| `python reproduce.py science` | Recompute 12 archived predictions |
| `python experiments/train_models.py` | Reconstruct both fitted forests and compare predictions with the bundled models |
| `python experiments/check_installation.py` | Agent imports and real compute-worker checks for both implementation versions |
| `python reproduce.py mcp-preflight` | Real external-session lifecycle checks after installing the separate MCP/Node dependencies |

The controlled runner also accepts `--suite orders`, `capacity`, or `all`. These generate local validation output under `outputs/` and leave the archived formal results unchanged.

## Formal studies

The full entry/result mapping is [EXPERIMENT_MAP.json](../EXPERIMENT_MAP.json).

| Study | Runners | Archived results |
| --- | --- | --- |
| Controlled resource policies | [src/orion/benchmark](../src/orion/benchmark) | [results/controlled](../results/controlled) |
| Fixed calls and resource lifecycles | [src/orion/benchmark](../src/orion/benchmark) | [results/fixed_calls](../results/fixed_calls), [results/resource_lifecycle](../results/resource_lifecycle) |
| Screening, held-out formulations and recommendation | [src/orion/benchmark](../src/orion/benchmark) | [results/screening](../results/screening), [results/heldout](../results/heldout), [results/recommendation](../results/recommendation) |
| Shared use and continuations | [sharing](sharing) | [results/sharing](../results/sharing) |
| Linked chemical analysis | [integrated](integrated) | [results/integrated](../results/integrated) |
| Task-aware comparison | [task_aware](task_aware) | [results/task_aware](../results/task_aware) |
| External MCP sessions | [mcp_sessions](mcp_sessions) | [results/mcp_sessions](../results/mcp_sessions) |

The 26-run task-aware comparison uses the earlier implementation under [versions/task_aware](../versions/task_aware). The external-session study keeps its own original resource implementation under [provenance/frozen_sources/mcp_sessions](../provenance/frozen_sources/mcp_sessions). These versions must not be silently replaced by the current layer when reproducing historical results.

## Online experiments

Configure your own external services and credentials using [.env.example](../.env.example) and a private file selected by `ORION_ENV_FILE`. Do not put a filled configuration file in the repository.

```powershell
python reproduce.py integrated --allow-api-calls
python reproduce.py task-aware --run-id point-r0-lru --allow-api-calls
```

Task-aware run identifiers are in [task_aware/protocol.json](task_aware/protocol.json). Sharing/continuation runners preserve their frozen prompts and pause/resume history. Use new output directories for retries. Initial sharing records must not be relabeled as one uninterrupted experiment.

LLM/Jev calls require private API credentials. Retrieval additionally requires BGE-M3 weights, the evidence graph/vector index and retrieval service. Original GPU measurements require compatible hardware/software. See [DATA_SOURCES.md](../DATA_SOURCES.md). The clean-install checks did not rerun paid experiments or complete live retrieval.
