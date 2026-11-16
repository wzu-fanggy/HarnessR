"""十二种独立留出电解液配方：自主 Agent、真实 ToolNode、统一资源预算 A/B。

先作完全离线的配方隔离及 48 点预测重放，再独立校准资源占用。
每臂使用原 Agent 和相同 RF 进程。原生缓存与 runtime 同受容量检查，
但只有 runtime 可以回收空闲资源。留出测量标签不会传入 Agent。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from orion.benchmark import cwsr_native_runtime_pair_v1 as base
from orion.benchmark import cwsr_complete_case_resource_v1 as complete

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
SPLIT = ROOT / "reports/cwsr_leca_rf_comparison_20261012_v1/split.json"
ARCHIVE = ROOT / "reports/cwsr_leca_rf_comparison_20261012_v1/predictions.csv"
MODEL = ROOT / "reports/cwsr_leca_rf_comparison_20261012_v1/models/rf_model.joblib"
TEMPERATURES = (-20.0, -10.0, 0.0, 20.0)
BASE_SNAPSHOT = complete.snapshot
BASE_SELECTED = base.selected_tasks


def load_split() -> dict:
    value = json.loads(SPLIT.read_text(encoding="utf-8"))
    train = value["training_formulations"]["ids"]
    held = value["held_out_formulations"]
    groups = [record["formulation_group"] for record in held]
    ids = [record["candidate_id"] for record in held]
    if len(train) != 60 or len(set(train)) != 60 or len(groups) != 12 or len(set(groups)) != 12:
        raise ValueError("冻结配方应为 60 种训练、12 种留出，且组身份唯一")
    if set(train) & set(groups) or len(set(ids)) != 12:
        raise ValueError("训练和留出配方重合或存在重复候选")
    expected = {(name, t) for name in ids for t in TEMPERATURES}
    cases = value["test_conditions"]
    actual = [(row["candidate_id"], float(row["target_temperature_c"])) for row in cases]
    if len(actual) != 48 or len(set(actual)) != 48 or set(actual) != expected:
        raise ValueError("四温度 48 点冻结条件不完整")
    if any(row.get("status") != "measured" or row["formulation_group"] != dict(zip(ids, groups))[row["candidate_id"]] for row in cases):
        raise ValueError("留出测量来源不一致")
    return value


def candidates_from_split(split: dict) -> list[dict]:
    """按既有特征定义反解全组分摩尔比例，不使用电导率标签。"""
    result = []
    for item in split["held_out_formulations"]:
        salt = float(item["x_NaSalt"])
        ec_relative = float(item["x_EC"])
        pc_relative = float(item["x_PC"])
        if not all(0.0 <= x <= 1.0 for x in (salt, ec_relative, pc_relative)):
            raise ValueError("配方特征超出摩尔比例范围")
        pc = (1.0 - salt) * pc_relative
        other = (1.0 - salt) * (1.0 - pc_relative)
        composition = {"NaPF6": salt, "EC": other * ec_relative,
                       "EMC": other * (1.0 - ec_relative), "PC": pc}
        if abs(sum(composition.values()) - 1.0) > 1e-12:
            raise ValueError("反解配方摩尔分数未归一")
        if abs(composition["EC"] / (composition["EC"] + composition["EMC"]) - ec_relative) > 1e-12:
            raise ValueError("EC/EMC 特征无法重建")
        if abs(composition["PC"] / sum(composition[k] for k in ("EC", "EMC", "PC")) - pc_relative) > 1e-12:
            raise ValueError("PC 特征无法重建")
        result.append({"candidate_id": item["candidate_id"], "salt": "NaPF6",
                       "solvents": ["EC", "EMC", "PC"], "additives": [],
                       "composition_inputs": {"molar_ratios": composition}})
    return result


def question_from_split(split: dict) -> str:
    candidates = candidates_from_split(split)
    return (
        "为低温钠离子电池电解液的后续实验选择优先测量候选。以下 12 种 NaPF6–EC/EMC/PC 配方"
        "均为公开数据中存在测量、但未用于本次冻结预测模型拟合的候选。"
        "molar_ratios 是包括盐在内的全组分摩尔比例。"
        "本任务为已有配方的模型性质初筛（candidate_screening），请加载 electrolyte_screening Skill，而不是配方设计 Skill。"
        "请使用科研电导率预测工具计算每种候选在 −20、−10、0、20 °C 的电导率。"
        "predict_formulation_conductivity 的 request.temperature_c 只接受单个温度数值，不能传列表；"
        "同一次工具调用可包含全部 12 种候选，四个温度须分别请求，调用顺序由你安排。"
        "共 48 个预测点；按 −20、−10、0 °C 三个温度预测值中的最低值由高到低排序，"
        "明确优先开展实验的四种候选；另比较 20 °C 的优先配方。"
        "请实际检索相关公开文献，说明电导率不能直接等同电池低温容量，"
        "并提出组成与电池构型相符的后续验证计划。"
        "最终通过 synthesize_design_report 编译 candidate_screening 报告，"
        "model_comparison 中提供全部 12 种候选的 ID，temperatures_c=[-20,-10,0]，"
        "reference_temperature_c=20，并设置 priority_count=4，要求报告按数值列出前四候选。"
        "在 request.next_steps 中写出至少两项具体的尚未执行实验验证计划："
        "候选的独立电导率复测，以及同等电极、电芯构型、倍率与电压范围下的电池性能测试。"
        "文献 claims 用完整可核的背景事实，并附真实逐字引文；不要把已有文献容量与本次预测混为实测。"
        "保留完整 48 项预测的数值和来源，不从文献或估算填充缺失的预测。"
        "文献检索仅需能够支持低温离子输运与后续电池验证的直接背景依据；"
        "完成全部预测并取得相关文献后，及时调用 synthesize_design_report 输出完整报告，"
        "不要为搜集与此初筛目标无关的材料继续反复检索。"
        "不要生成新配方，也不要将计算值误写成新测量；此轮输入不包含留出实测真值。\n\n"
        "候选配方（仅输入组成）：\n"
        + json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    )


def selected_tasks() -> dict:
    items = BASE_SELECTED()
    items["T1"] = (SPLIT, {"sample": {"question": question_from_split(load_split())},
                            "case_id": "CWSR_HELDOUT12_CONDUCTIVITY_SCREENING"})
    return items


def snapshot() -> dict:
    record = BASE_SNAPSHOT()
    for path in (SELF, SPLIT, ARCHIVE, MODEL, ROOT / "benchmark/cwsr_heldout12_shared_admission_v1.py"):
        record[str(path.relative_to(ROOT.parent)).replace("\\", "/")] = base.sha(path)
    return dict(sorted(record.items()))


def configure() -> None:
    complete.configure_base()
    base.source_snapshot = snapshot
    base.selected_tasks = selected_tasks
    base.__file__ = str(SELF)


def archived_rf() -> dict:
    """离线预测以已存盘的 RF 结果为唯一参照，不重新选择模型。"""
    values = {}
    with ARCHIVE.open(encoding="utf-8", newline="") as stream:
        for record in csv.DictReader(stream):
            if record["model"] != "rf":
                continue
            key = (record["candidate_id"], float(record["target_temperature_c"]))
            if key in values:
                raise ValueError("归档 RF 点重复")
            values[key] = float(record["predicted_ms_cm"])
    if len(values) != 48:
        raise ValueError("归档 RF 结果不是完整 48 点")
    return values


def preflight() -> dict:
    import joblib
    from tools.electrolyte_design.conductivity_ml import ConductivityFeatures, SodiumConductivityRandomForest
    split = load_split()
    candidates = candidates_from_split(split)
    prompt = question_from_split(split)
    if "measured_ms_cm" in prompt or "measured_log10" in prompt:
        raise ValueError("留出真值不得进入 Agent 问题")
    model = joblib.load(MODEL)
    if not isinstance(model, SodiumConductivityRandomForest) or not model.fitted:
        raise TypeError("预训练留出模型不是适用的随机森林")
    archive = archived_rf()
    observed = set()
    for row in split["test_conditions"]:
        point = (row["candidate_id"], float(row["target_temperature_c"]))
        estimate = model.predict_features(ConductivityFeatures(
            inverse_temperature=float(row["inverse_temperature"]),
            x_NaSalt=float(row["x_NaSalt"]),
            x_EC=float(row["x_EC"]),
            x_PC=float(row["x_PC"]))).conductivity_ms_cm
        if not math.isclose(estimate, archive[point], abs_tol=1e-9, rel_tol=1e-10):
            raise ValueError("模型预测与已冻结的 48 点 RF 数值不一致")
        observed.add(point)
    if len(observed) != 48 or len(candidates) != 12:
        raise ValueError("留出验证数据不完整")
    return {"status": "PASS", "train_formulations": 60, "heldout_formulations": 12,
            "validated_predictions": 48, "split_sha256": base.sha(SPLIT),
            "model_sha256": base.sha(MODEL), "archive_sha256": base.sha(ARCHIVE),
            "question_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "holdout_truth_exposed_in_prompt": False, "model_training_reexecuted": False}


def scientific_audit(result_path: Path) -> dict:
    """数值科研工作流和最终报告质量分开核验，不用编译成功代替科研论证。"""
    from scipy.stats import spearmanr
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    expected = archived_rf()
    details = (raw.get("task_artifact") or {}).get("model_comparison") or {}
    rows = details.get("records") or []
    issues = []
    points = {}
    for row in rows:
        prediction = row.get("prediction") or {}
        temperature = prediction.get("temperature_c")
        key = (row.get("candidate_id"), float(temperature) if temperature is not None else -999.0)
        if key in points:
            issues.append("重复预测：" + str(key))
            continue
        if key not in expected:
            issues.append("非预期预测：" + str(key))
            continue
        if prediction.get("status") != "predicted":
            issues.append("未得到数值：" + str(key))
            continue
        value = prediction.get("numeric_value")
        if value is None or not math.isclose(float(value), expected[key], abs_tol=1e-8, rel_tol=1e-9):
            issues.append("预测不匹配：" + str(key))
            continue
        points[key] = float(value)
    missing = set(expected) - set(points)
    if missing:
        issues.append("缺少 " + str(len(missing)) + " 个正确的温度-配方预测点")
    split = load_split()
    by_id = {row["candidate_id"]: row for row in split["held_out_formulations"]}
    low_pred = {}
    low_measured = {}
    for candidate_id in by_id:
        cold = [(candidate_id, t) for t in TEMPERATURES if t < 20]
        if all(point in points for point in cold):
            low_pred[candidate_id] = min(points[point] for point in cold)
        conditions = {float(c["target_temperature_c"]): float(c["measured_ms_cm"])
                      for c in split["test_conditions"] if c["candidate_id"] == candidate_id}
        low_measured[candidate_id] = min(conditions[t] for t in TEMPERATURES if t < 20)
    spearman = None
    overlap = None
    if len(low_pred) == 12:
        ids = list(by_id)
        spearman = float(spearmanr([low_pred[i] for i in ids], [low_measured[i] for i in ids]).statistic)
        overlap = len(set(sorted(ids, key=lambda i: -low_pred[i])[:4])
                      & set(sorted(ids, key=lambda i: -low_measured[i])[:4]))
    status = (raw.get("task_artifact") or {}).get("task_status")
    if status != "model_assessed":
        issues.append("报告状态不是 model_assessed")
    invoked = [call.get("name") for call in raw.get("tool_calls", [])]
    if "predict_formulation_conductivity" not in invoked:
        issues.append("未调用真实电导率预测工具")
    if not any(name in {"search_literature_multiview", "search_literature_chunks", "search_paper_chunks"} for name in invoked):
        issues.append("未完成文献检索")
    if "synthesize_design_report" not in invoked:
        issues.append("未调用结构化报告工具")
    if not (raw.get("task_artifact") or {}).get("claims"):
        issues.append("最终报告缺少有出处的文献事实")
    scientific_complete = bool(raw.get("ok") and not issues and len(points) == 48)
    from orion.benchmark.cwsr_heldout12_evidence_audit_v1 import report_audit, archived
    report_quality = report_audit(raw, archived())
    return {"scientific_complete": scientific_complete,
            "report_quality_pass": bool(scientific_complete and report_quality["full_report_quality_pass"]),
            "report_quality": report_quality,
            "agent_ok": bool(raw.get("ok")),
            "task_status": status, "valid_prediction_points": len(points),
            "expected_prediction_points": 48, "issues": issues[:30],
            "spearman_low_temperature": spearman, "top4_overlap": overlap,
            "predicted_top4": sorted(low_pred, key=lambda i: -low_pred[i])[:4],
            "measured_top4": sorted(low_measured, key=lambda i: -low_measured[i])[:4],
            "prediction_reference_sha256": base.sha(ARCHIVE)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "prepare", "run-arm", "audit"), default="preflight")
    parser.add_argument("--output", default="reports/cwsr_heldout12_agent_pair_20261014_v1")
    parser.add_argument("--budget-mode", choices=("tight", "roomy"), default="tight")
    parser.add_argument("--calibration-source", help="复用先前已冻结且有效的三轮校准 JSON；核验原始测量与公式后保存指纹")
    parser.add_argument("--service-port", type=int, default=8817)
    parser.add_argument("--arm", choices=base.ARMS)
    parser.add_argument("--pair", type=int, default=1)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--worker-task", help=argparse.SUPPRESS)
    parser.add_argument("--worker-arm", help=argparse.SUPPRESS)
    parser.add_argument("--service-url", help=argparse.SUPPRESS)
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure()
    if args.worker_task:
        folder = base.workspace_path(args.output)
        plan = json.loads((folder.parent / "manifest.json").read_text(encoding="utf-8"))
        from benchmark.cwsr_heldout12_shared_admission_v1 import run_worker
        run_worker(args.worker_arm, folder, args.service_url, plan)
        return

    report = preflight()
    if args.stage == "preflight":
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    output = base.workspace_path(args.output)
    if args.stage == "prepare":
        if output.exists():
            raise FileExistsError("不覆盖或复用已存在的实验目录")
        output.mkdir(parents=True)
        base.dump(output / "holdout-preflight.json", report)
        calibration_source = None
        if args.calibration_source:
            calibration_source = base.workspace_path(args.calibration_source)
            calibration = json.loads(calibration_source.read_text(encoding="utf-8"))
            rows = calibration.get("rows") or []
            if calibration.get("schema") != "cwsr_complete_case_calibration_v1" or len(rows) != 3:
                raise ValueError("仅允许复用有三轮原始测量的既有有效校准")
            bge_samples = [row["bge_resource"]["host_ram_mib"] for row in rows]
            rf_samples = [row["rf_memory"]["rss_mib"] for row in rows]
            if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                       for x in bge_samples + rf_samples):
                raise ValueError("旧校准存在缺失或非正数值，不得复用")
            expected_bge = math.ceil(max(bge_samples) * 1.15 + 16)
            expected_rf = math.ceil(max(rf_samples) * 1.15 + 16)
            expected_budget = math.ceil(max(expected_bge, expected_rf)
                                        + 0.30 * min(expected_bge, expected_rf))
            if (calibration["bge_declared_mib"], calibration["rf_declared_mib"],
                    calibration["managed_host_ram_mib"]) != (expected_bge, expected_rf, expected_budget):
                raise ValueError("校准值与事先定义的占用公式不一致")
            base.dump(output / "calibration.json", calibration)
        else:
            calibration = complete.measure_calibration(output, MODEL, args.service_port)
        b = calibration["bge_declared_mib"]
        rf = calibration["rf_declared_mib"]
        tight = calibration["managed_host_ram_mib"]
        roomy = math.ceil(1.2 * (b + rf))
        chosen = tight if args.budget_mode == "tight" else roomy
        if not (max(b, rf) < tight < b + rf <= roomy):
            raise RuntimeError("不满足各自可建但紧预算不能并存的预先校准条件")
        plan = base.make_plan(["T1"], args.pairs, 600, "deepseek-v4-flash", str(MODEL))
        plan["tasks"]["T1"]["task_role"] = "heldout_12_formulation_screening"
        plan["tasks"]["T1"]["task_type"] = "candidate_screening"
        plan["settings"]["calibration"] = calibration
        plan["settings"]["calibration_sha256"] = base.sha(output / "calibration.json")
        plan["settings"]["calibration_origin"] = (
            {"type": "reused_frozen_three_repeat_measurement",
             "source_path": str(calibration_source.relative_to(ROOT)),
             "source_sha256": base.sha(calibration_source)}
            if calibration_source else {"type": "new_three_repeat_measurement"})
        plan["settings"]["split_overfull_batches"] = True
        plan["settings"]["runtime_managed_budget"] = {
            "host_ram_mib": chosen, "gpu_vram_mib": 6144,
            "process_slots": 4, "connection_slots": 8}
        plan["comparison"].update({"holdout_split_frozen": True,
                                   "holdout_measurements_not_sent_to_agent": True,
                                   "common_admission_checks": True,
                                   "same_prediction_model_and_process_deployment": True,
                                   "benchmark_worker_task_type_override": "design -> candidate_screening",
                                   "original_cache_can_reclaim_idle": False,
                                   "budget_mode": args.budget_mode})
        base.dump(output / "manifest.json", plan)
        print(json.dumps({"stage": "prepared", "budget_mode": args.budget_mode,
                          "calibration_origin": plan["settings"]["calibration_origin"],
                          "budget_mib": chosen, "bge_mib": b, "rf_mib": rf,
                          "arms": [x["arm_order"] for x in plan["schedule"]]}, ensure_ascii=False), flush=True)
        return
    plan = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    base.verify_plan(plan)
    if plan["settings"]["calibration_sha256"] != base.sha(output / "calibration.json"):
        raise ValueError("校准文件指纹变化")
    if plan["settings"]["conductivity_model_sha256"] != report["model_sha256"]:
        raise ValueError("模型不符合原始离线预核验")
    if args.stage == "audit":
        for pair in plan["schedule"]:
            for arm in pair["arm_order"]:
                folder = output / f"{pair['pair_id']}-{arm}"
                if (folder / "result.json").exists():
                    audit = scientific_audit(folder / "result.json")
                    base.dump(folder / "scientific-audit.json", audit)
                    print(json.dumps({"pair": pair["pair_id"], "arm": arm,
                                      "scientific_complete": audit["scientific_complete"],
                                      "prediction_points": audit["valid_prediction_points"],
                                      "issues": audit["issues"]}, ensure_ascii=False))
        return
    if not args.arm or args.pair < 1 or args.pair > len(plan["schedule"]):
        raise ValueError("run-arm 必须指定 --arm 和有效的 --pair")
    pair = plan["schedule"][args.pair - 1]
    folder = output / f"{pair['pair_id']}-{args.arm}"
    if folder.exists():
        raise FileExistsError("禁止覆盖已保存的 A/B 结果")
    print(json.dumps({"stage": "running", "arm": args.arm, "pair": pair["pair_id"],
                      "budget": plan["settings"]["runtime_managed_budget"]}, ensure_ascii=False), flush=True)
    observation = base.run_arm("T1", args.arm, folder, plan, args.service_port)
    audit = scientific_audit(folder / "result.json")
    base.dump(folder / "scientific-audit.json", audit)
    observation["scientific_audit"] = audit
    base.dump(output / f"observation_{pair['pair_id']}_{args.arm}.json", observation)
    print(json.dumps({"stage": "finished", "arm": args.arm, "agent_ok": observation["ok"],
                      "scientific_complete": audit["scientific_complete"],
                      "prediction_points": audit["valid_prediction_points"],
                      "issues": audit["issues"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
