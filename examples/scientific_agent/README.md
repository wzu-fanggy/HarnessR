# Scientific agent reference integration

This example guide points to the actual scientific agent used in the paper. It supplies an execution context for Harnessᴿ and reproduces the electrolyte studies. Its source remains in the tested `src/orion` layout; this guide does not contain a second copy of the agent.

| Component | Actual implementation | Role |
| --- | --- | --- |
| Resource management layer | [src/orion/capvm](../../src/orion/capvm/README.md) | Harnessᴿ coordination |
| Resource registration and application bindings | [default_resource_runtime.py](../../src/orion/core/default_resource_runtime.py) | Connect the application's resources to the layer |
| Selected-tool execution boundary | [langgraph_residency.py](../../src/orion/core/langgraph_residency.py), [resource_tool_node.py](../../src/orion/core/resource_tool_node.py) | Acquire required resources around an already selected call |
| Task information | [task_board_intent.py](../../src/orion/core/task_board_intent.py), [resource_task_graph.py](../../src/orion/core/resource_task_graph.py) | Supply task requirements to coordination |
| Reference agent | [main.py](../../src/orion/main.py) and its context, memory and task modules | LLM orchestration and task execution |
| Chemistry and retrieval tools | [electrolyte_design](../../src/orion/tools/electrolyte_design), [kg_rag](../../src/kg_rag) | Scientific application |

The agent decides what work to perform and selects tools. Harnessᴿ coordinates the resources used to execute those selected calls. The tool definitions, scientific calculations and LLM orchestration remain separate responsibilities.

## Check the integration locally

From the repository root, install `requirements-agent.txt` in the virtual environment, then run:

```powershell
.venv\Scripts\python experiments/check_installation.py
```

This imports the current and earlier reference implementations, checks their bundled model/training-data paths, and starts and closes real prediction and ranking workers. It performs no model API requests.

## Adapt the design

Use [catalog.py](../../src/orion/capvm/catalog.py) and the [resource adapters](../../src/orion/capvm/adapters) to describe the resources your tools need. The [shared-runtime tests](../../src/orion/tests/capvm/test_shared_resource_runtime.py) demonstrate coordinated acquisition, protection and admission. Connect resource acquisition to your own selected-tool execution boundary, and supply task requirements if task-aware decisions are needed. The bundled agent shows one integration; adoption by another agent requires that integration work.

Live paper experiments are documented in [experiments/README.md](../../experiments/README.md). The reference agent is not distributed with an LLM, the complete evidence graph or a live retrieval service.
