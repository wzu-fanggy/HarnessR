"""真实 KG 中验证 Driver 重建与按查询关闭 Session。"""

from __future__ import annotations

import json
from pathlib import Path

from orion.benchmark.hcvm_neo4j_session_integration_v1 import run as run_original
from orion.capvm.adapters.neo4j_driver import Neo4jDriverCapabilityAdapter


def run() -> dict:
    result = run_original(adapter_factory=Neo4jDriverCapabilityAdapter)
    result["success"] = bool(
        result["success"]
        and result["adapter"]["driver_open_count"] == 2
        and result["adapter"]["session_creation_count"] == 2
    )
    result["schema_version"] = "hcvm-neo4j-driver-integration-v2"
    result["status"] = "PASS_REAL_ELECTROLYTE_KG_DRIVER_REFAULT" if result["success"] else "FAIL_REAL_ELECTROLYTE_KG_DRIVER_REFAULT"
    result["physical_resource"] = "Neo4j Driver and connection pool"
    result["interpretation_guard"] = (
        "Read-only lifecycle check: the retained object is a Neo4j Driver/pool. "
        "Every query uses a newly created and closed Session; the connection slot "
        "is a configured quota rather than a physical connection count."
    )
    return result


if __name__ == "__main__":
    output = Path(__file__).with_name("hcvm_neo4j_driver_integration_v2_result.json")
    result = run()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result["status"])
    raise SystemExit(0 if result["success"] else 1)
