"""文献辅助推荐的语义判断接入层；确定性筛选结果保留为诊断信息。"""
from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any

from orion.core.design_decision import DecisionState, EvidenceGap, EvidenceLedgerItem
from orion.core.retrieved_evidence import _retrieved_sources
from orion.core.semantic_evidence_review import SemanticEvidenceReview, review_candidate, validate_review


def semantic_mode_enabled() -> bool:
    """保留历史回放行为；新判断路径通过显式配置验证后启用。"""
    return os.environ.get('ORION_DESIGN_REASONING_MODE', 'structured') == 'semantic'


async def apply_semantic_post_tool_policy(state, tool_update, policy, model):
    from orion.core.harness_policy import (
        _merge_state_with_tool_update, _combine_updates, apply_post_tool_policy)
    merged = _merge_state_with_tool_update(state, tool_update)
    prepared = await prepare_semantic_reviews(merged, model)
    return apply_post_tool_policy(state, _combine_updates(tool_update, prepared), policy)


def semantic_decision(state: Mapping[str, Any]) -> DecisionState:
    """根据已核验引文的模型审查形成决定，不伪造数值评估或联合见证。"""
    reviews = [SemanticEvidenceReview.model_validate(item['review'])
               for item in state['design_semantic_results']]
    sources = state['design_semantic_sources']
    from orion.core.recorded_predictions import recorded_predictions
    computations = recorded_predictions(state)
    ids = {item.candidate_id for item in state.get('design_candidates') or []}
    if {item.candidate_id for item in reviews} != ids or len(reviews) != len(ids):
        raise ValueError('语义审查与本轮候选不一一对应')
    for review in reviews:
        validate_review(review, review.candidate_id, sources, computations)
    approved = [item.candidate_id for item in reviews if item.recommendation_supported]
    gaps = []
    for review in reviews:
        if review.recommendation_supported:
            continue
        if review.recommendation_action == 'hold' and review.recommendation_reason:
            gaps.append(EvidenceGap(code='semantic_advice_hold', description=review.recommendation_reason,
                candidate_id=review.candidate_id, can_replan=False))
        for requirement in review.requirements:
            if requirement.status != 'supported':
                gaps.append(EvidenceGap(code='semantic_' + requirement.status,
                    description=requirement.requirement + '：' + requirement.reasoning,
                    candidate_id=review.candidate_id,
                    search_hint=requirement.requirement))
        for requirement in review.unaddressed_requirements:
            gaps.append(EvidenceGap(code='semantic_unaddressed_requirement',
                description=requirement, candidate_id=review.candidate_id, search_hint=requirement))
        if not review.original_goal_fully_addressed and not any(
                gap.candidate_id == review.candidate_id for gap in gaps):
            gaps.append(EvidenceGap(code='semantic_goal_unresolved',
                description=review.experimental_association, candidate_id=review.candidate_id,
                search_hint=review.experimental_association))
    attempts = int(state.get('design_attempts') or 0)
    maximum = int(state.get('design_max_attempts') or 2)
    assessment_complete = all(review.original_goal_fully_addressed for review in reviews)
    return DecisionState(status='recommend' if approved else ('search' if attempts < maximum and not assessment_complete else 'hold'),
        approved_candidate_ids=approved, evidence_gaps=[] if approved else gaps,
        next_queries=[] if approved else list(dict.fromkeys(gap.search_hint for gap in gaps)),
        attempts=attempts, max_attempts=maximum,
        semantic_evidence_reviews=reviews,
        reason_codes=['model_semantic_review_with_source_verification_v1'])


async def prepare_semantic_reviews(state: Mapping[str, Any], model: Any) -> dict[str, Any]:
    """每次筛选修订审查一次；调用或格式失败直接报错，不冒充科学暂缓。"""
    revision = int(state.get('design_state_revision') or 0)
    if (state.get('design_task_type') != 'design'
            or state.get('candidate_evaluations') is None
            or state.get('terminal_committed')
            or int(state.get('design_semantic_revision', -1)) == revision):
        return {}
    candidates = state.get('design_candidates') or []
    if not candidates:
        return {}
    question = str(state.get('design_original_question') or '').strip()
    if not question:
        raise ValueError('语义审查缺少本轮原始用户问题')
    retrieved = _retrieved_sources(state)
    if not retrieved:
        raise ValueError('语义审查缺少本轮真实检索来源')
    ledger = [item if isinstance(item, EvidenceLedgerItem) else EvidenceLedgerItem.model_validate(item)
              for item in state.get('evidence_ledger') or []]
    semaphore = asyncio.Semaphore(2)
    from orion.core.recorded_predictions import recorded_predictions
    computations = recorded_predictions(state)

    async def assess(candidate):
        # 只提供本轮已经检索到的相关论文上下文，不按论文名选择答案。
        papers = {item.chunk_id.split('::', 1)[0] for item in ledger
                  if item.candidate_id == candidate.candidate_id}
        # 给定/提出的候选未必出自论文。没有既有归属时提供本轮检索原文，
        # 由模型判断其适用性；已有论文归属的候选仍只读对应上下文。
        sources = {key: str(row['content']) for key, row in retrieved.items()
                   if not papers or key.split('::', 1)[0] in papers}
        if not sources:
            raise ValueError('候选缺少已检索的真实来源：' + candidate.candidate_id)
        async with semaphore:
            result = await asyncio.wait_for(review_candidate(question,
                {'candidate_id': candidate.candidate_id, **candidate.composition}, sources, model,
                computations=computations), 180)
        return result, sources

    pairs = await asyncio.gather(*(assess(candidate) for candidate in candidates))
    all_sources = {key: value for _, sources in pairs for key, value in sources.items()}
    # 将模型实际引用的检索原文保留到报告 ledger；不填猜测的实验条件。
    present = {(item.candidate_id, item.chunk_id) for item in ledger}
    from orion.core.design_runtime import _stable_id
    for result, _ in pairs:
        review = SemanticEvidenceReview.model_validate(result['review'])
        citations = [citation for item in review.requirements for citation in item.citations]
        for citation in [*citations, *review.association_citations]:
            key = (review.candidate_id, citation.chunk_id)
            if key in present:
                continue
            present.add(key)
            ledger.append(EvidenceLedgerItem(evidence_id=_stable_id('semantic-source', *key),
                candidate_id=key[0], chunk_id=key[1], paper_id=key[1].split('::', 1)[0],
                evidence_kind='background', excerpt=all_sources[key[1]]))
    return {'design_semantic_results': [result for result, _ in pairs],
            'design_semantic_sources': all_sources, 'design_semantic_revision': revision,
            'evidence_ledger': ledger}
