"""Capability residency and runtime-resource coordination primitives.

The core package keeps logical capability availability separate from the residency of
model-facing information, processes, models, sessions and other runtime objects. The
resource governor remains independent of planning; production LangGraph execution can
optionally enter it at the stable ToolNode seam through ``core.langgraph_residency``.
"""

from .capability_asset import (
    CapabilityAssetBundle,
    CapabilityAssetImportResult,
    CapabilityAssetManifest,
    CapabilityEvolutionRecord,
)
from .catalog import CapabilityCatalog
from .deployment import (
    CapabilityDeploymentQualifier,
    DeploymentInputProfile,
    DeploymentQualificationDecision,
    DeploymentQualificationStatus,
    DeploymentQualifiedInvoker,
    DeploymentScopeContract,
    ScopedInvocationResult,
)
from .evolvable_plugin import (
    CapabilityLearnedState,
    EvolvableLangGraphPlugin,
    JsonCapabilityLearnedStateStore,
    PluginExecutionState,
)
from .governor import CapabilityResidencyGovernor
from .lifecycle import CapabilityLifecycleCoordinates
from .local_capability_folder import (
    LocalCapabilityEvent,
    LocalCapabilityEventType,
    LocalCapabilityFolderManager,
    LocalCapabilityManifest,
    LocalCapabilityPackageContext,
    LocalCapabilityRecord,
    LocalCapabilityStatus,
)
from .plugin_bundle import CapabilityPluginBundle, CapabilityPluginBundleManifest
from .plugin_host import (
    HostExecutionState,
    PluginHotSwapResult,
    PluginRegistry,
    StableLangGraphPluginHost,
)
from .plugin_evolution import (
    PluginEvolutionResult,
    PluginValidationCase,
    ValidatedPluginEvolution,
)
from .regression_qualification import (
    RegressionCompatibilityPolicy,
    RegressionQualificationAssessment,
    RegressionQualificationGate,
    RegressionQualificationSnapshot,
)
from .resource_budget import physical_resource_budget
from .routing import (
    AdmissibilityDecision,
    CapabilityCompositionPlan,
    CapabilityControlPlane,
    CapabilityDescriptor,
    CapabilityDiscoveryBackend,
    CascadedDiscoveryBackend,
    DeploymentQualificationRule,
    DiscoveryCandidate,
    DiscoveryRankSelector,
    EmbeddingDiscoveryBackend,
    EmbeddingSemanticSelector,
    EvolvableCapabilityRouter,
    ExactTriggerDiscoveryBackend,
    GreedyQueryCoverageComposer,
    MetadataDiscoveryBackend,
    RecoveryAction,
    ReciprocalRankFusionDiscoveryBackend,
    RoutedCandidate,
    RoutedCapabilityCompositionLease,
    RoutedCapabilityLease,
    RoutedCapabilityRuntime,
    RoutingPlan,
    RoutingRequest,
    RuntimeAdmissionFailure,
    SemanticCapabilityComposer,
    SemanticCapabilitySelector,
)
from .model import (
    CapabilityDependency,
    CapabilityFault,
    CapabilityKind,
    CapabilityRuntimeState,
    MeasuredResidencyBudgetExceeded,
    ResidencyBudgetExceeded,
    ResidencyEvent,
    ResidencyLevel,
    ResourceVector,
    VirtualCapability,
)
from .policies import LRUCostEvictionPolicy, LayaDecisionModel, LayaReuseEvictionPolicy
from .stateful_graph_plugin import StatefulLangGraphCapabilityPlugin

__all__ = [
    "AdmissibilityDecision",
    "CapabilityAssetBundle",
    "CapabilityAssetImportResult",
    "CapabilityAssetManifest",
    "CapabilityCatalog",
    "CapabilityCompositionPlan",
    "CapabilityControlPlane",
    "CapabilityDeploymentQualifier",
    "CapabilityDependency",
    "CapabilityDescriptor",
    "CapabilityDiscoveryBackend",
    "CapabilityEvolutionRecord",
    "CapabilityFault",
    "CapabilityKind",
    "CapabilityLearnedState",
    "CapabilityLifecycleCoordinates",
    "CapabilityPluginBundle",
    "CapabilityPluginBundleManifest",
    "CapabilityResidencyGovernor",
    "CapabilityRuntimeState",
    "MeasuredResidencyBudgetExceeded",
    "CascadedDiscoveryBackend",
    "DeploymentInputProfile",
    "DeploymentQualificationDecision",
    "DeploymentQualificationRule",
    "DeploymentQualificationStatus",
    "DeploymentQualifiedInvoker",
    "DeploymentScopeContract",
    "DiscoveryCandidate",
    "DiscoveryRankSelector",
    "EmbeddingDiscoveryBackend",
    "EmbeddingSemanticSelector",
    "EvolvableCapabilityRouter",
    "EvolvableLangGraphPlugin",
    "ExactTriggerDiscoveryBackend",
    "GreedyQueryCoverageComposer",
    "HostExecutionState",
    "JsonCapabilityLearnedStateStore",
    "LRUCostEvictionPolicy",
    "LayaDecisionModel",
    "LayaReuseEvictionPolicy",
    "LocalCapabilityEvent",
    "LocalCapabilityEventType",
    "LocalCapabilityFolderManager",
    "LocalCapabilityManifest",
    "LocalCapabilityPackageContext",
    "LocalCapabilityRecord",
    "LocalCapabilityStatus",
    "MetadataDiscoveryBackend",
    "PluginExecutionState",
    "PluginEvolutionResult",
    "PluginHotSwapResult",
    "PluginRegistry",
    "PluginValidationCase",
    "physical_resource_budget",
    "RecoveryAction",
    "RegressionCompatibilityPolicy",
    "RegressionQualificationAssessment",
    "RegressionQualificationGate",
    "RegressionQualificationSnapshot",
    "ReciprocalRankFusionDiscoveryBackend",
    "ResidencyBudgetExceeded",
    "ResidencyEvent",
    "ResidencyLevel",
    "ResourceVector",
    "RoutedCandidate",
    "RoutedCapabilityCompositionLease",
    "RoutedCapabilityLease",
    "RoutedCapabilityRuntime",
    "RoutingPlan",
    "RoutingRequest",
    "RuntimeAdmissionFailure",
    "ScopedInvocationResult",
    "SemanticCapabilityComposer",
    "SemanticCapabilitySelector",
    "StableLangGraphPluginHost",
    "StatefulLangGraphCapabilityPlugin",
    "ValidatedPluginEvolution",
    "VirtualCapability",
]
