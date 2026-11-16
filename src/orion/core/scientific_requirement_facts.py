"""科学能力的完整范围摘要；不加载模型、不改变科学计算或结果。"""

from dataclasses import dataclass

from copy import deepcopy

from orion.capvm.requirement_coverage import digest





def _candidate(candidate):

    return {k: deepcopy(candidate[k]) for k in ('candidate_id', 'salt', 'solvents',

            'additives', 'conductivity_features', 'composition_inputs') if k in candidate}





@dataclass(frozen=True)

class ScientificRequirementFacts:

    kind: str



    def request_scope(self, arguments):

        request = deepcopy(arguments.get('request', arguments))

        if self.kind == 'predict':

            return {'kind': self.kind, 'candidates': [_candidate(c) for c in request['candidates']],

                    'temperature_c': request['temperature_c']}

        if self.kind == 'rank':

            return {'kind': self.kind, 'candidates': [_candidate(c) for c in request['candidates']],

                    'temperature_c': request['temperature_c'],

                    'min_conductivity_ms_cm': request['min_conductivity_ms_cm'],

                    'ensemble_budget': request.get('ensemble_budget')}

        return {'kind': self.kind, 'request': request}



    def matches(self, scope, value):

        """仅证明该调用范围产生了合法输出；语义上的要求覆盖仍独立判断。"""

        if self.kind == 'fit':

            return (isinstance(value, dict) and value.get('status') == 'fitted'

                    and value.get('measurement_count') == len(scope['request']['measurements'])

                    and {m.get('model') for m in value.get('models', [])} == {'arrhenius', 'vtf'})

        candidates = scope['candidates']

        ids = [c.get('candidate_id') for c in candidates]

        if len(ids) != len(set(ids)) or None in ids:

            return False

        if self.kind == 'rank':

            return (isinstance(value, dict)

                    and value.get('tool') == 'rank_formulations_under_uncertainty'

                    and value.get('temperature_c') == scope['temperature_c']

                    and value.get('min_conductivity_ms_cm') == scope['min_conductivity_ms_cm']

                    and len(value.get('candidates', []))==len(ids)

                    and {c.get('candidate_id') for c in value.get('candidates', [])} == set(ids)

                    and bool(value.get('ranking')))

        if not isinstance(value, list) or len(value) != len(candidates):

            return False

        if {c.get('candidate_id') for c in value} != set(ids):

            return False

        for c in value:

            predictions = c.get('model_predictions', [])

            if not predictions or any(p.get('status') != 'predicted'

                    or p.get('temperature_c') != scope['temperature_c'] for p in predictions):

                return False

        return True



    def outcome(self, value):

        if self.kind == 'rank':

            return {'ranking': value.get('ranking'), 'unranked_candidates': value.get('unranked_candidates'),

                    'evidence_status': value.get('evidence_status')}

        if self.kind == 'fit':

            return {'status': value.get('status'), 'preferred_by_aicc': value.get('preferred_by_aicc')}

        return {'model_ids': sorted({p.get('model_id', '') for c in value

                                    for p in c.get('model_predictions', [])})}





def scientific_requirement_registry():

    return {'predict_formulation_conductivity': ScientificRequirementFacts('predict'),

            'rank_formulations_under_uncertainty': ScientificRequirementFacts('rank'),

            'fit_conductivity_temperature_law': ScientificRequirementFacts('fit')}

