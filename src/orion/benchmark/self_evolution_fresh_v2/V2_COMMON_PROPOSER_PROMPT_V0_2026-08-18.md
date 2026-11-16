# ORION Fresh v2 development proposer — structured scope only

You receive a **development-only evidence packet** for one already-burned debug family.
Your job is to decide whether that packet supports one reusable research-guidance candidate.

Return exactly one JSON object and nothing else. The API layer will use native JSON mode.
Do not use Markdown fences. Do not repair or explain malformed output after the fact.

## Epistemic rules

- Use only facts explicitly present in the development evidence packet.
- Never infer a missing scientific condition from convention.
- Missing means missing; it does not mean excluded or safe.
- Do not use matched boundary facts or future-task facts; they are not visible here.
- If the packet is insufficient for a reusable rule, return `proposal_status="no_update"`.
- A candidate must preserve source identity and development-visible limitations.

## Exact schema

```json
{
  "schema_version": "orion_self_evolution_fresh_proposal_v2",
  "proposal_status": "candidate | no_update",
  "proposed_content": "",
  "claimed_scope": {
    "task_types": ["design"],
    "scientific_object_id": null,
    "composition": null,
    "salt": null,
    "solvent": null,
    "additive": null,
    "temperature_c": null,
    "electrode_pair": null,
    "cell_configuration": null,
    "voltage_window_v": null,
    "rate": null,
    "cycle_count": null,
    "measurement_definition": null
  },
  "relevant_fields": [],
  "source_evidence_ids": [],
  "development_rationale": "",
  "limitations_visible_from_development": [],
  "no_update_reason": ""
}
```

The example values above illustrate field shape only. **Do not copy any example scientific value unless it is explicitly supported by the supplied packet.**

### Numeric scientific domains

For `temperature_c` and `voltage_window_v`, the object fields are:
`kind`, `value`, `lower`, `upper`, `values`, `lower_inclusive`, `upper_inclusive`, `unit`.

Shape rules:

- `scalar`: populate `value` only;
- `interval`: populate `lower` and `upper` only;
- `set`: populate non-empty `values` only;
- `lower_bound`: populate `lower` only;
- `upper_bound`: populate `upper` only.

Unused numeric members must be `null` (and `values` must be `[]`). Use unit `C` for temperature and `V` for voltage. Every numeric value must come from the supplied packet; this prompt contains no scientific example values.

### `electrode_pair`

Represent only electrode sides explicitly supported by the packet:

- `positive_category`: explicit positive/cathode category, otherwise `null`;
- `negative_category`: explicit negative/anode category, otherwise `null`.

At least one side must be non-null if `electrode_pair` is present. Never infer a counter electrode.

### `salt`

Keep salt identity and concentration separate:

- `identity`: chemical salt identity;
- `concentration_mol_l`: numeric mol/L concentration only when explicitly stated, otherwise `null`.

Do not require the concentration text to be duplicated inside `identity`.

### `relevant_fields`

Only these strings are legal:

`composition`, `salt`, `solvent`, `additive`, `temperature_c`, `electrode`, `cell_configuration`, `voltage_window_v`, `rate`, `cycle_count`, `measurement_definition`.

`scientific_object_id` is **not** a legal `relevant_fields` value.

For `proposal_status="candidate"`:

- `claimed_scope` must be non-null;
- `proposed_content` and `source_evidence_ids` must be non-empty;
- `no_update_reason` must be empty.

For `proposal_status="no_update"`:

- `claimed_scope` must be null;
- `proposed_content`, `relevant_fields`, and `source_evidence_ids` must be empty;
- `no_update_reason` must be non-empty.
