"""Graph-native Change Domain V0.

该包只提供只读的 Harness 多层耦合图和局部演化边界推断。
V0 不修改生产 KG、不调用 LLM，也不执行真实 Harness 变更。
"""

from .capability_projection import (
    CapabilityContractProjection,
    CapabilityWorkingSetProjection,
    add_capability_contract_projection,
    add_task_role_requirements,
    project_capability_working_set,
    project_tool_capability_working_set,
)
from .contextual_bridge import ContextualBridgeProjection, add_contextual_scientific_bridge
from .extractors import (
    add_capability_dependencies,
    add_task_contract_projection,
    load_graphify_static_projection,
    link_source_symbol_views,
)
from .inference import recover_change_domain
from .kg_projection import (
    add_kg_evidence_scope_projection,
    add_kg_query_results_projection,
    add_kg_relation_provenance_projection,
    evidence_chunk_artifact_id,
    scientific_relation_artifact_id,
)
from .runtime_projection import add_task_artifact_runtime_projection
from .scientific_extractor import load_scientific_field_anchor_projection
from .state_extractor import load_python_state_access_projection
from .skill_projection import (
    SkillWorkflowProjection,
    SkillWorkflowStep,
    active_skill_tool_nodes,
    add_skill_workflow_projection,
    read_skill_frontmatter,
)
from .tool_identity_extractor import ToolIdentityProjection, load_tool_identity_projection
from .task_projection import (
    ScientificTaskProjection,
    TaskResponsibilityProjection,
    add_scientific_task_projection,
    project_task_responsibilities,
)
from .models import (
    ChangeDomainResult,
    ChangeRequest,
    CouplingEdge,
    CouplingLayer,
    HarnessArtifact,
    MultiplexHarnessGraph,
)

__all__ = [
    "add_capability_contract_projection",
    "add_task_role_requirements",
    "project_capability_working_set",
    "project_tool_capability_working_set",
    "CapabilityContractProjection",
    "CapabilityWorkingSetProjection",
    "add_contextual_scientific_bridge",
    "ContextualBridgeProjection",
    "add_capability_dependencies",
    "add_task_contract_projection",
    "add_kg_query_results_projection",
    "add_kg_evidence_scope_projection",
    "add_kg_relation_provenance_projection",
    "add_task_artifact_runtime_projection",
    "evidence_chunk_artifact_id",
    "scientific_relation_artifact_id",
    "load_graphify_static_projection",
    "link_source_symbol_views",
    "load_python_state_access_projection",
    "add_skill_workflow_projection",
    "active_skill_tool_nodes",
    "read_skill_frontmatter",
    "SkillWorkflowProjection",
    "SkillWorkflowStep",
    "load_scientific_field_anchor_projection",
    "load_tool_identity_projection",
    "ToolIdentityProjection",
    "add_scientific_task_projection",
    "project_task_responsibilities",
    "ScientificTaskProjection",
    "TaskResponsibilityProjection",
    "ChangeDomainResult",
    "ChangeRequest",
    "CouplingEdge",
    "CouplingLayer",
    "HarnessArtifact",
    "MultiplexHarnessGraph",
    "recover_change_domain",
]
