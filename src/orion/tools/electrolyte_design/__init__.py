"""电解液工具的公开入口，计算子模块导入时不提前加载其他工具。"""



from importlib import import_module





_TOOL_MODULES = {

    "predict_formulation_conductivity": ".conductivity_tool",

    "enrich_formulation_evidence": ".enrichment",

    "evaluate_formulation_constraints": ".evaluator",

    "extract_numeric_properties": ".properties",

    "formulate_candidates": ".formulate",

    "generate_candidate_space": ".candidate_space",

    "evaluate_candidate_batch": ".candidate_batch_tools",

    "inspect_candidate_artifact": ".candidate_batch_tools",

    "compare_conductivity_predictions": ".candidate_batch_tools",

    "rank_formulations_under_uncertainty": ".robust_ranking_tool",

    "fit_conductivity_temperature_law": ".temperature_fit_tool",

}



__all__ = list(_TOOL_MODULES)





def __getattr__(name: str):

    """按需导入公开工具，保留原有包级导入接口。"""



    module_name = _TOOL_MODULES.get(name)

    if module_name is None:

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name, __name__), name)

    globals()[name] = value

    return value





def __dir__() -> list[str]:

    return sorted(set(globals()) | set(__all__))

