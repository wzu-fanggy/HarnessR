# Harness<sup>R</sup>

**Task-aware tool resource management for AI agent harnesses.**

Harnessᴿ coordinates the resources that support tool execution. It separates the tools an agent can select from the resources kept ready for execution, coordinates shared use across tools and agents, protects resources in active use, and uses subsequent task requirements to inform resource decisions.

The contribution implemented here is the **resource management layer**. The included scientific agent is a **reference integration and a vehicle for reproducing the paper**, with sodium-ion electrolyte studies as the application. Its language-model orchestration, context system and chemistry tools are supporting components.

## Start with the resource layer

| What to read | Location |
| --- | --- |
| Harnessᴿ implementation and source guide | [src/orion/capvm](src/orion/capvm/README.md) |
| Registration, resource states and capacity | [catalog.py](src/orion/capvm/catalog.py), [model.py](src/orion/capvm/model.py), [governor.py](src/orion/capvm/governor.py) |
| Shared access, active-use protection and admission | [shared_runtime.py](src/orion/capvm/shared_runtime.py) |
| Task information and retention decisions | [task_intent.py](src/orion/capvm/task_intent.py), [semantic_plan.py](src/orion/capvm/semantic_plan.py), [demand_coordinator.py](src/orion/capvm/demand_coordinator.py) |
| Resource lifecycle adapters | [adapters](src/orion/capvm/adapters) |
| Integration with a selected tool call | [langgraph_residency.py](src/orion/core/langgraph_residency.py), [resource_tool_node.py](src/orion/core/resource_tool_node.py) |

The source retains the tested `orion.capvm` / `capvm` import paths. This repository distributes the working research implementation; adapting it to another harness involves supplying resource adapters and connecting resource acquisition to tool execution. The integration guide identifies those boundaries.

## Repository layout

```text
HarnessR/
  src/
    orion/capvm/           HarnessR resource coordination implementation
    orion/core/            Resource integration and reference-agent infrastructure
    orion/main.py          Reference scientific agent
    orion/tools/           Tools used by that reference application
    orion/benchmark/       Paper runners and their required historical helpers
    kg_rag/               Retrieval service for the electrolyte application
    release_paths.py      Portable experiment-path resolution
    release_verify.py     Offline checks of archived evidence and predictions
  examples/scientific_agent/  Guide to the included reference integration
  experiments/            Paper reproduction entry points and protocols
  data/                   Required inputs, public training data and fitted models
  results/                Archived formal results and returned evidence
  versions/               Earlier implementation used by the task-aware study
  provenance/             Source fingerprints, versions and packaging history
  checks/                 Installation, test and evidence-validation records
  docs/                   Component boundaries and directory guide
  reproduce.py            Common reproduction entry point
  requirements-*.txt      Core and optional dependency groups
  constraints-tested.txt  Versions resolved in the clean installation
```

The [component guide](docs/REPOSITORY_MAP.md) distinguishes core code, reference integration and reproduction materials. The [experiment map](EXPERIMENT_MAP.json) links formal studies to their runners and results. The directory names and import paths have been preserved so that reorganizing the presentation does not change the validated implementation.

## Install and check the core

Python 3.11 is the tested version. On Windows:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-core.txt
.venv\Scripts\python reproduce.py tests
.venv\Scripts\python reproduce.py controlled --suite main
```

These commands run local checks without a model API. To inspect archived paper evidence with Python's standard library only:

```powershell
python reproduce.py verify
```

The `requirements` files specify dependencies; `constraints-tested.txt` also fixes the transitive versions used in the clean installation. Optional dependencies are installed only when their requirement group is selected.

## Reference scientific agent

The [reference integration guide](examples/scientific_agent/README.md) shows where Harnessᴿ enters the scientific agent. The complete agent and retrieval dependencies are optional:

```powershell
.venv\Scripts\python -m pip install -r requirements-agent.txt
.venv\Scripts\python experiments/check_installation.py
```

The installation check imports both implementation versions and starts their real prediction and ranking workers using bundled data. It does not invoke the LLM or Jev API. The original agent entry is [src/orion/main.py](src/orion/main.py); it is retained to reproduce the paper's orchestration and tool execution.

## Reproduce the paper

See [experiments/README.md](experiments/README.md) for the entry-point guide and [DATA_SOURCES.md](DATA_SOURCES.md) for data and model provenance.

```powershell
.venv\Scripts\python reproduce.py fixed-summary
.venv\Scripts\python reproduce.py science
.venv\Scripts\python experiments/train_models.py
```

These commands reaggregate the fixed-call records, recompute archived predictions and reconstruct the two fitted forests. New output is written under `outputs/`; archived `results/` are preserved.

For real MCP filesystem sessions, install the separate client and the locked Node service:

```powershell
.venv\Scripts\python -m pip install -r requirements-mcp.txt
npm --prefix experiments/mcp_sessions/server ci
.venv\Scripts\python reproduce.py mcp-preflight
```

The original MCP lifecycle/WorkingSet checks are Windows-specific. `requirements-platform.txt` supplies optional process telemetry; reproducing the original GPU measurements requires compatible hardware and a suitable PyTorch/CUDA installation.

Live agent experiments require your own private configuration file, selected with `ORION_ENV_FILE`. The shared entry point requires `--allow-api-calls` for live LLM/Jev commands. Complete retrieval experiments also require BGE-M3 weights, the evidence graph/vector index and retrieval service. Saved returned evidence supports auditing but does not replace the live database. These prerequisites are detailed in the reproduction guide.

## Validation and scope

A fresh Windows/Python 3.11.9 core environment and a separate complete agent/MCP environment were checked on 16 November 2026. The 69 selected tests, controlled main suite, fixed-call reaggregation, 12 exact prediction comparisons, two model reconstructions, both agent-version imports and compute-worker startup, and MCP lifecycle preflight passed. The extracted archive also passed relocation checks in a directory containing spaces. See [clean-environment.json](checks/clean-environment.json) and [roundtrip-check.json](checks/roundtrip-check.json).

Most archive bytes are original result records. Earlier source versions and imported helpers with historical names are retained where the paper depends on them; their presence does not add contributions or experiments to this work. The package excludes credentials, personal-session databases, virtual environments, installation caches, `node_modules`, bulk source PDFs and new runtime outputs. Data and formal-result files are unchanged. No paid experiments were rerun during packaging.

An author-code license remains to be selected before public distribution. Third-party software and datasets retain their own licenses. Original experiment environment information is in [provenance/verified_environment.json](provenance/verified_environment.json).
