# Harnessᴿ resource coordination implementation

This directory contains the resource management implementation used in the paper. Its retained internal package name is `capvm`, also imported as `orion.capvm`.

| Design responsibility | Source |
| --- | --- |
| Resource descriptions, states, dependencies and capacity measures | [model.py](model.py), [catalog.py](catalog.py) |
| Resource preparation, residency and capacity decisions | [governor.py](governor.py), [resource_budget.py](resource_budget.py), [policies](policies) |
| Coordinated shared access, active-use protection and admission | [shared_runtime.py](shared_runtime.py) |
| Task requirements and decision guidance | [task_intent.py](task_intent.py), [task_demand.py](task_demand.py), [semantic_plan.py](semantic_plan.py), [demand_coordinator.py](demand_coordinator.py), [jev_task_intent.py](jev_task_intent.py) |
| Resource-specific lifecycle operations | [adapters](adapters) |
| Measurements and observations | [telemetry.py](telemetry.py), [resource_observation.py](resource_observation.py), [resource_profiles.py](resource_profiles.py) |

Resources are the managed objects; a loaded model, process or session is one possible runtime realization. These are distinct from the model-facing tool definitions an agent can select.

The scientific agent calls this layer through the [ToolNode bridge](../core/langgraph_residency.py). For another harness, start with the catalog, adapters and shared-access API, then connect them at the point where a selected tool is executed. The reference agent's memory, context and LLM orchestration are not required as a conceptual part of Harnessᴿ, although the current research package retains shared imports for reproduction.

Plugin/deployment/routing compatibility interfaces remain in this package because its existing exports and imported helpers depend on them. They are supporting implementation code, not additional methods or experimental claims of this paper. The original implementation snapshots under `versions/` and `provenance/` remain separate.
