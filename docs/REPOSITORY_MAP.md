# Component boundaries and retained material

## Core contribution

[src/orion/capvm](../src/orion/capvm/README.md) contains Harnessᴿ resource coordination. The source guide identifies registration, capacity, shared access, protection, task information and resource adapters. The `capvm` name is retained for compatibility with the tested research source.

## Reference integration

The integration boundary is implemented by `default_resource_runtime.py`, `langgraph_residency.py`, `resource_tool_node.py`, `resource_task_graph.py` and `task_board_intent.py` under [src/orion/core](../src/orion/core). The rest of that directory contains supporting agent infrastructure and must not all be described as Harnessᴿ.

[src/orion/main.py](../src/orion/main.py), its context/memory/task modules and [examples/scientific_agent](../examples/scientific_agent/README.md) provide the paper's reference agent. [src/orion/tools](../src/orion/tools) and [src/kg_rag](../src/kg_rag) provide the scientific tools and retrieval implementation. These components demonstrate and reproduce the integration.

## Reproduction and evidence

| Directory | Why it is retained |
| --- | --- |
| `experiments/` | Formal study entry points, input preparation, protocols and local validation |
| `src/orion/benchmark/` | Original runners and their imported helpers |
| `data/` | Public training inputs, fitted models, held-out partition and frozen task inputs |
| `results/` | Formal run records, returned evidence and analysis outputs supporting the paper |
| `versions/` | Earlier implementation actually used by the task-aware study |
| `provenance/` | Original and packaged fingerprints, source/version mapping and packaging changes |
| `checks/` | Validation records for this release; distinct from the paper's formal results |

There are 34 files under `data/` and 1,256 under `results/`; their bytes are unchanged from the checked release. Most archive bytes are formal records rather than source code.

## Historical names and compatibility code

Some retained helper names refer to older work. Their presence is determined by imports from the actual paper runners, not by claiming those old experiments as part of this paper. For example, the complete-case and paired agent runners use `_V2IsolatedContextManager` from `self_evolution_fresh_v2_development_runtime.py`. The fixed-call runner imports controlled catalog/trace helpers through `cwsr_electrolyte_workflow_policy_replay_v1.py`. The small workbench audit module and related support are imported through this closure; no workbench application or separate web interface is provided.

The core's existing package exports also retain plugin, deployment and routing compatibility interfaces. Removing those imports or extracting a new standalone public API would change the validated implementation. They remain documented support code, not extra paper contributions.

The previous audit removed an unreferenced shared-compute prototype and five unavailable unrelated tool exports. No further file was removed merely because it had an old name. This organization pass changes documentation and directory navigation only; executable code, requirements, models, inputs and formal results are unchanged. The per-file component inventory is in [checks/repository-scope.json](../checks/repository-scope.json).

## Excluded from the release

Virtual environments, installed dependencies, `node_modules`, package-manager caches, filled environment files, credentials, personal-session databases, bulk source PDFs, unrelated pilot result collections and newly generated output directories are excluded. The frozen prompt source required by a formal sharing protocol is retained as an input, even where its historical filename includes `pilot`.

The complete evidence graph/vector index, BGE-M3 weights and external model services remain external reproduction prerequisites; their absence is explicit in the installation and data documentation.
