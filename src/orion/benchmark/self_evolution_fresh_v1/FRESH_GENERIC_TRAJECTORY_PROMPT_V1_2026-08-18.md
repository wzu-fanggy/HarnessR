# Fresh generic audited-evolution trajectory prompt v1

Status: **PREFREEZE — NO FORMAL EXECUTION AUTHORIZED**

You are completing one frozen scientific proxy task for an audited persistent-guidance study.

Use only the user task, generic task contract, evidence packet and optional sealed candidate supplied in this call.

The optional sealed candidate is **guidance, not scientific evidence**. It may improve what direction you investigate or what conditions you check, but it cannot establish a current-task fact and it cannot override the supplied evidence.

Do not infer hidden conditions, future evaluator labels, desired arm outcomes, or any scientific fact that is absent from the packet. Do not rewrite, narrow, broaden or repair the sealed candidate.

If the supplied evidence is insufficient for a requested scientific claim, preserve that uncertainty and request the missing evidence in `next_actions`.

Return exactly one JSON object with exactly these keys:

```json
{
  "task_completion": "non-empty source-bound task note",
  "evidence_refs_used": [],
  "explicit_boundaries": [],
  "next_actions": ["at least one concrete next action"],
  "unsupported_claims": []
}
```

Rules:

- `evidence_refs_used` may contain only evidence/source IDs explicitly supplied in the current packet.
- `explicit_boundaries` and `next_actions` must be JSON lists of strings.
- `unsupported_claims` must be an empty JSON list.
- Do not cite the persistent candidate as evidence.
- Return JSON only. Do not add markdown fences or commentary.
