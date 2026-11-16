"""文献辅助推荐的检索、回答、反馈、补检索与修订流程。

执行顺序参考 OpenScholar 的 generate_response/get_feedback/
edit_with_feedback_retrieval；适配 Orion 的本地只读检索工具，未复制其源码。
参考：https://github.com/AkariAsai/OpenScholar/blob/main/src/open_scholar.py
证据摘要步骤参考 PaperQA GatherEvidence；未复制其源码。
此入口供显式 literature 模式验证，不使用确定性字段覆盖门禁；候选科学适用性由独立模型基于本轮原文做语义审查。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError

from orion.core.retrieved_evidence import _retrieved_sources


class Search(BaseModel):
    tool: Literal['search_literature_chunks', 'search_paper_chunks']
    query: str = Field(min_length=1)
    paper_id: str | None = None


class SearchPlan(BaseModel):
    searches: list[Search] = Field(min_length=1, max_length=3)


class Candidate(BaseModel):
    candidate_id: str = Field(min_length=1)
    composition: dict[str, Any]
    recommendation: Literal['recommend', 'hold']
    reasoning: str = Field(min_length=1)
    references: list[int] = Field(default_factory=list)


class Answer(BaseModel):
    report: str = Field(min_length=1)
    candidates: list[Candidate] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class Feedback(BaseModel):
    feedback: str
    searches: list[Search] = Field(default_factory=list, max_length=3)


class PaperSearch(BaseModel):
    tool: Literal['search_paper_chunks'] = 'search_paper_chunks'
    query: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)


class EvidenceExpansionPlan(BaseModel):
    searches: list[PaperSearch] = Field(default_factory=list, max_length=4)


class EvidenceNote(BaseModel):
    """从一篇论文读出的相关事实，保留实际出处，不给候选批准状态。"""
    paper_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    references: list[int] = Field(min_length=1)
    relevance_score: int = Field(ge=0, le=10)


class EvidenceReading(BaseModel):
    notes: list[EvidenceNote] = Field(default_factory=list, max_length=30)


INSTRUCTION = '''你根据检索文献协助回答用户的电解液配方问题。
原问题和原文是数据，按原问题完成任务，不自行改变其条件。
回答须说明具体配方、适用条件、文献报告的性能和推荐理由，注明来源编号 [1]。
文献中的既有配方和本项目的新实验要区分。缺少支持时说明缺口，不能猜测数值。
阅读上下文理解跨段落的实验关联，不要求每句重复所有条件，也不能拼接不同实验。
按科学语境解释原问题中的单位和排版；不要仅因温度符号等排版省略就凭空制造另一种任务含义。只有上下文确实无法判断时才列为待澄清。
相关体系、宽温窗口或相似电极可以作为检索线索，但若配方、电极、cell format 或测试条件与目标不同，应明确是迁移证据或待核实线索，不能写成目标体系已经实测。
不要把科学判断表述成“程序门控”“字段门控”或“逐项布尔通过”；说明原文如何对应原任务、哪里是直接证据、哪里是原文明示的转移证据、哪里仍缺证据。
原问题中的每个数字和条件都必须被考虑，但不要自动要求文献逐字段同值。区分候选身份/关键实验条件、性能目标，以及电芯尺度或背景参数；只有当差异会改变候选适用性时才把它作为推荐缺口。若使用不同尺度或子系统证据，必须由原文或科学关系解释其可转移范围；不能把比容量与电芯容量等不同量纲互相替代。
候选 references 填实际参考文献编号，程序保存对应原文，不要复述引文。
输入 sources 的编号由程序提供。只返回对应 schema 的 JSON。'''


READING_INSTRUCTION = '''阅读按论文分组的原文，为回答原问题摘出相关事实，本步不提出推荐。
每条摘要描述一个明确的研究对象及其相关结果，写清原文给出的实验条件、数值含义与比较基准。
保留具体组分及用量。原文没给出的信息写为未说明，不根据常识补齐。
同一论文的不同段落可结合阅读，说明它们的对象关联；不同实验的结果分开摘录。
记录与目标的实际对应关系，相关或相似的体系不自动当成目标体系。
每条摘要只使用所列 paper_id 的实际来源编号。不同论文分别摘录。
不相关的原文不摘录。relevance_score 只评价对原问题的相关程度，不表示推荐批准。'''


EXPANSION_INSTRUCTION = '''在写最终候选前，判断哪些已经命中的论文值得做一次论文内补证。
只允许对 evidence_notes 中已经出现的 paper_id 使用 search_paper_chunks；不要引入新论文名。
优先选择最接近原问题、只缺少一两个关键细节的论文，例如摘要已经匹配目标电极、cell format、温度或添加剂，但缺精确配方、倍率、循环条件或性能数值。
不要因为另一篇论文 relevance_score 更高，就忽略这种近乎直接命中的论文。
查询词应直接针对缺失细节。若现有原文已经足够回答，searches 返回空数组。'''


async def plan_evidence_expansion(question, evidence, model, runs):
    return await _ask(model, EvidenceExpansionPlan, EXPANSION_INSTRUCTION,
        {'original_question': question, **evidence}, runs)


async def gather_evidence_notes(question, refs, model, runs):
    """先读证据再写答案，按照相关程度选择摘要，不做条件字段裁决。"""
    papers = {}
    for row in refs:
        paper = row.get('paper_id') or row['chunk_id'].split('::', 1)[0]
        papers.setdefault(paper, []).append(row)
    reading_data = {'original_question': question,
                    'papers': [{'paper_id': key, 'sources': rows} for key, rows in papers.items()]}
    reading = await _ask(model, EvidenceReading, READING_INSTRUCTION, reading_data, runs)
    by_number = {row['reference']: row for row in refs}

    def normalize_and_validate_source_binding(value):
        for note in value.notes:
            source_papers = set()
            for number in note.references:
                source = by_number.get(number)
                if source is None:
                    return '证据摘要引用了未检索的来源编号'
                source_papers.add(source.get('paper_id') or source['chunk_id'].split('::', 1)[0])
            if len(source_papers) > 1:
                return '证据摘要混用了不同论文的来源编号'
            if source_papers:
                # paper_id 是模型生成的显示字段，不作为数据库身份主键；真实 reference
                # 已唯一绑定到检索来源，因此在单论文引用成立时规范化为真实 paper_id。
                note.paper_id = next(iter(source_papers))
        return None

    binding_error = normalize_and_validate_source_binding(reading)

    if binding_error:
        reading = await _ask(model, EvidenceReading,
            READING_INSTRUCTION + '\n上一轮只存在来源绑定协议错误。请保持原文事实不变，逐条确保 references 全部来自该 note.paper_id；不能确认的摘要直接删除，不得换用其它论文补齐。',
            {**reading_data, 'previous_reading': reading.model_dump(mode='json'),
             'binding_error': binding_error}, runs)
        binding_error = normalize_and_validate_source_binding(reading)
    if binding_error:
        raise ValueError(binding_error)
    # 检索上下文选择采用问题相关度，保留原文以便核查摘要。
    # 不设科学条件阈值，不把摘要得分当作推荐依据。
    selected, per_paper = [], {}
    for note in sorted(reading.notes, key=lambda row: row.relevance_score, reverse=True):
        if note.relevance_score == 0 or per_paper.get(note.paper_id, 0) >= 4:
            continue
        selected.append(note.model_dump(mode='json'))
        per_paper[note.paper_id] = per_paper.get(note.paper_id, 0) + 1
        if len(selected) == 12: break
    numbers = {number for note in selected for number in note['references']}
    return {'evidence_notes': selected,
            'sources': [row for row in refs if row['reference'] in numbers]}


async def _ask(model, schema, instruction, data, runs):
    prompt = (INSTRUCTION + '\n' + instruction + '\nSCHEMA\n'
              + json.dumps(schema.model_json_schema(), ensure_ascii=False)
              + '\nDATA\n' + json.dumps(data, ensure_ascii=False))
    last_error = None
    for attempt in range(2):
        retry_note = '' if attempt == 0 else (
            '\n上一轮输出不是合法的目标 JSON。只重新按上述 SCHEMA 输出完整 JSON；'
            '不要改变科学判断，也不要添加 DATA 中不存在的事实。')
        response = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=prompt + retry_note)]), 120)
        if response.response_metadata.get('finish_reason') == 'length':
            raise ValueError('模型输出截断；不能记为科学暂缓')
        try:
            result = schema.model_validate_json(response.content)
        except ValidationError as exc:
            last_error = exc
            runs.append({'stage': instruction + ' [structured-output retry]',
                         'output': {'parse_error': str(exc)},
                         'usage_metadata': response.usage_metadata,
                         'response_metadata': {key: response.response_metadata.get(key)
                                               for key in ('model_name', 'finish_reason')}})
            continue
        runs.append({'stage': instruction, 'output': result.model_dump(mode='json'),
                     'usage_metadata': response.usage_metadata,
                     'response_metadata': {key: response.response_metadata.get(key)
                                           for key in ('model_name', 'finish_reason')}})
        return result
    raise last_error


async def read_full_chunks(rows):
    """读取已经检索到的 chunk 完整文本，不选论文、配方或新增条件。"""
    from orion.tools.rag.service_client import get_root_kg_service_client
    ids = list(rows)
    for start in range(0, len(ids), 20):
        payload = await asyncio.to_thread(get_root_kg_service_client().call,
            '/v1/evidence/chunks', {'chunk_ids': ids[start:start + 20]})
        for row in payload.get('chunks') or []:
            key = row.get('chunk_id')
            if key in rows and row.get('text'):
                rows[key] = {**rows[key], **row, 'content': row['text']}
    return rows


def validate_answer(answer, sources):
    """检查实际来源编号；不把编号存在称作科学判断已通过。"""
    valid = set(range(1, len(sources) + 1))
    seen = set()
    for candidate in answer.candidates:
        if candidate.candidate_id in seen:
            raise ValueError('候选编号重复')
        seen.add(candidate.candidate_id)
        if candidate.recommendation == 'recommend' and not candidate.references:
            raise ValueError('模型推荐没有来源')
        if set(candidate.references) - valid:
            raise ValueError('模型引用了未检索的来源编号')
    if {int(x) for x in re.findall(r'\[(\d+)\]', answer.report)} - valid:
        raise ValueError('报告引用了未检索的来源编号')


async def ensure_answer_references(question, answer, evidence, model, runs, sources):
    """只修复推荐候选缺引用的输出协议；找不到真实来源就降为 hold。"""
    missing = [candidate.candidate_id for candidate in answer.candidates
               if candidate.recommendation == 'recommend' and not candidate.references]
    if not missing:
        validate_answer(answer, sources)
        return answer
    repaired = await _ask(model, Answer,
        '只修复候选引用协议，不重新做科学推理。保持 candidate_id 和 composition 不变，不新增候选。'
        '对 recommendation=recommend 且 references 为空的候选，只能填写 sources 中已经存在、且实际支持该候选的来源编号；'
        '如果不能从现有 sources 确定真实支持来源，就把该候选改为 hold，并在 unresolved_questions 说明缺少来源。'
        '报告同步移除任何仍无来源支撑的推荐表述。不得新增实验事实或数值。',
        {'original_question': question, **evidence,
         'answer_to_repair': answer.model_dump(mode='json'),
         'missing_reference_candidates': missing}, runs)
    before = {candidate.candidate_id: candidate.composition for candidate in answer.candidates}
    after = {candidate.candidate_id: candidate.composition for candidate in repaired.candidates}
    if set(after) - set(before):
        raise ValueError('引用协议修复引入了新候选')
    if any(after[key] != before[key] for key in after):
        raise ValueError('引用协议修复改变了候选组成')
    validate_answer(repaired, sources)
    return repaired


async def review_candidates(question, answer, refs, review_model, runs, reviewer=None):
    """对所有带真实来源的候选做第二视角原文语义审查；程序不替模型翻转 recommend/hold。"""
    if reviewer is None:
        from orion.core.semantic_evidence_review import review_candidate
        reviewer = review_candidate
    by_number = {row['reference']: row for row in refs}
    reviewable = [candidate for candidate in answer.candidates if candidate.references]
    semaphore = asyncio.Semaphore(2)

    def paper_identity(row):
        return row.get('paper_id') or row['chunk_id'].split('::', 1)[0]

    async def assess(candidate):
        cited_rows = [by_number[number] for number in candidate.references]
        cited_papers = {paper_identity(row) for row in cited_rows}
        # 候选引用只用于确定相关论文；独立审查读取这些论文在本轮已检索到的全部原文，
        # 避免只看回答自行挑选的支持片段而漏掉同文中的限定条件或反证。
        rows = [row for row in refs if paper_identity(row) in cited_papers]
        source_text = {row['chunk_id']: row['text'] for row in rows}
        if not source_text:
            raise ValueError('推荐候选缺少可供语义审查的原文')
        payload = {'candidate_id': candidate.candidate_id, **candidate.composition}
        async with semaphore:
            result = await asyncio.wait_for(
                reviewer(question, payload, source_text, review_model), 180)
        review = result.get('review') or {}
        if review.get('candidate_id') != candidate.candidate_id:
            raise ValueError('语义审查返回了其它候选')
        record = {'candidate_id': candidate.candidate_id,
                  'draft_recommendation': candidate.recommendation,
                  'recommendation_supported': bool(result.get('recommendation_supported')),
                  'review': review,
                  'source_references': list(candidate.references)}
        runs.append({'stage': 'independent semantic candidate review',
                     'output': record,
                     'usage_metadata': result.get('usage_metadata'),
                     'response_metadata': result.get('response_metadata')})
        return record

    if not reviewable:
        return []
    return list(await asyncio.gather(*(assess(candidate) for candidate in reviewable)))


def review_disagreements(answer, reviews):
    verdicts = {row['candidate_id']: row['recommendation_supported'] for row in reviews}
    unsupported_recommendations, supported_holds = [], []
    for candidate in answer.candidates:
        verdict = verdicts.get(candidate.candidate_id)
        if candidate.recommendation == 'recommend' and verdict is False:
            unsupported_recommendations.append(candidate.candidate_id)
        elif candidate.recommendation == 'hold' and verdict is True:
            supported_holds.append(candidate.candidate_id)
    return unsupported_recommendations, supported_holds


def validate_review_alignment(answer, reviews):
    """只检查最终结构与模型审查是否一致，不重新计算科学条件。"""
    verdicts = {row['candidate_id']: row['recommendation_supported'] for row in reviews}
    for candidate in answer.candidates:
        if candidate.recommendation != 'recommend':
            continue
        if not verdicts.get(candidate.candidate_id, False):
            raise ValueError('最终推荐未获得本轮独立语义审查支持：' + candidate.candidate_id)


async def run_workflow(question, model, tools, *, reader=read_full_chunks, max_feedback=2,
                       reviewer=None, review_model=None, expand_evidence=False):
    """先读证据，再按需要展开命中论文、回答并迭代补检索；没有逐字段批准矩阵。"""
    by_name = {tool.name: tool for tool in tools}
    runs, messages, sources, executed = [], [], {}, set()
    semantic_review_model = review_model or model
    active_review_model = semantic_review_model
    semantic_review_model_mode = 'separate_model' if review_model is not None and review_model is not model else 'same_model_second_pass'
    semantic_review_model_failures = []

    async def review_answer(answer):
        nonlocal active_review_model, semantic_review_model_mode
        try:
            return await review_candidates(
                question, answer, references(), active_review_model, runs, reviewer=reviewer)
        except Exception as exc:
            if active_review_model is model:
                raise
            semantic_review_model_failures.append({
                'error_type': type(exc).__name__,
                'message': str(exc)[:500],
            })
            active_review_model = model
            semantic_review_model_mode = 'same_model_second_pass_after_separate_reviewer_failure'
            return await review_candidates(
                question, answer, references(), active_review_model, runs, reviewer=reviewer)

    async def retrieve(searches):
        batch = []
        known_papers = {row.get('paper_id') or key.split('::', 1)[0]
                        for key, row in sources.items()}
        for request in searches:
            if request.tool not in by_name:
                raise ValueError('检索工具不在本次许可列表')
            if request.tool == 'search_paper_chunks' and request.paper_id not in known_papers:
                raise ValueError('论文内检索必须使用已检索到的 paper_id')
            args = {'query': request.query, 'top_k': 8}
            if request.tool == 'search_paper_chunks': args['paper_id'] = request.paper_id
            signature = (request.tool, json.dumps(args, sort_keys=True))
            if signature in executed: continue
            executed.add(signature)
            batch.append({'id': uuid4().hex, 'name': request.tool, 'args': args})
        if not batch: return False
        messages.append(AIMessage(content='', tool_calls=batch))
        for call in batch:
            result = await by_name[call['name']].ainvoke(call['args'])
            if (isinstance(result, dict) and result.get('error')) or (
                    isinstance(result, list) and any(row.get('error') for row in result if isinstance(row, dict))):
                raise RuntimeError('检索服务失败：' + json.dumps(result, ensure_ascii=False))
            messages.append(ToolMessage(name=call['name'], tool_call_id=call['id'],
                content=json.dumps(result, ensure_ascii=False, default=str)))
        current = _retrieved_sources({'messages': messages})
        # 所有读到的来源编号一经分配保持不变，修订时不重排。
        full = await reader(current)
        for key, row in full.items(): sources[key] = row
        return True

    def references():
        return [{'reference': i, 'chunk_id': key,
                 'paper_id': row.get('paper_id') or key.split('::', 1)[0],
                 'paper_title': row.get('paper_title'), 'text': row['content']}
                for i, (key, row) in enumerate(sources.items(), 1)]

    plan = await _ask(model, SearchPlan,
        '为原问题提出 3 个互补的初始检索，当前尚无已知论文。'
        '第一条尽量保留目标体系、原题给出的精确温度/倍率/电流/循环数等数值条件以及目标 observable，'
        '不要为了泛化而丢掉 900 mA、capacity retention、ICE、corrosion 等判别信息；'
        '第二条面向电解液配方/添加剂/盐溶剂等干预分支；第三条用同义表达做宽召回。'
        '若原问题要求 select and compare / compare branches，检索计划要支持发现多个可比较分支，不能只围绕一个候选。'
        '查询中不要写入你凭记忆猜到的论文名或隐藏答案。',
        {'original_question': question}, runs)
    if any(request.tool == 'search_paper_chunks' for request in plan.searches):
        plan = await _ask(model, SearchPlan,
            '修复初始检索计划：当前尚无本轮已发现的 paper_id，不能直接论文内检索。'
            '请改用普通文献检索发现论文；不要凭记忆填写 paper_id，也不要改变原任务。',
            {'original_question': question, 'invalid_initial_plan': plan.model_dump(mode='json')}, runs)
    await retrieve(plan.searches)
    if not sources: raise RuntimeError('初始检索未返回可阅读的文献')
    evidence = await gather_evidence_notes(question, references(), model, runs)
    if expand_evidence:
        expansion = await plan_evidence_expansion(question, evidence, model, runs)
        if await retrieve(expansion.searches):
            evidence = await gather_evidence_notes(question, references(), model, runs)
    answer = await _ask(model, Answer,
        '根据证据摘要和对应原文回答原问题。集中提出证据充分的具体配方，写明组分及用量。'
        '信息不完整的体系作为待核实线索，不列为已经选定的配方。不要扩展成全领域综述。'
        '任何 recommendation=recommend 的候选都必须在 references 中列出实际支持它的来源编号；不能确定来源就设为 hold。',
        {'original_question': question, **evidence}, runs)
    answer = await ensure_answer_references(question, answer, evidence, model, runs, sources)
    semantic_reviews = await review_answer(answer)
    for _ in range(max_feedback):
        unsupported, undercalled = review_disagreements(answer, semantic_reviews)
        feedback = await _ask(model, Feedback,
            '检查回答是否回应原问题、是否有原文依据。independent_semantic_reviews 是另一轮基于原文的模型审查，'
            '用它定位配方归属、实验条件和任务要求的证据缺口；它不是程序门禁，也不能代替原文。'
            '若 recommend 候选未获独立审查支持，优先补齐缺口，补不到就改为 hold。'
            '若 hold 候选被独立审查认为已经充分支持，重新检查自己是否新增了原题没有的要求或漏读了原文明示的实验关联；'
            '没有独立、原文支持的暂缓理由时应改为 recommend。'
            '若原问题要求 select and compare / compare branches，检查回答是否真的比较了本轮已发现的多个有意义分支；'
            '不能因已经找到一个可推荐候选就停止比较。若只有一个分支证据足够，也要说明其它已发现分支为什么只能 hold。'
            '没有需要修改或补检索的内容时，feedback 返回空字符串，searches 返回空数组。',
            {'original_question': question, **evidence,
             'answer': answer.model_dump(mode='json'),
             'independent_semantic_reviews': semantic_reviews}, runs)
        if not feedback.feedback and not feedback.searches and not unsupported and not undercalled:
            break
        feedback_text = feedback.feedback
        if unsupported and not feedback_text:
            feedback_text = ('独立语义审查仍认为部分拟推荐候选的原文支持不足。'
                             '根据审查中的具体缺口补证；若本轮没有新增证据，就将这些候选暂缓。')
        if undercalled:
            extra = ('独立语义审查认为以下 hold 候选已有足够原文支持：' + ', '.join(undercalled) + '。'
                     '重新核对是否把原题未要求的条件变成了必需项，或忽略了原文明示的跨段落/子系统转移关系。'
                     '若仍保持 hold，必须给出当前 sources 中独立且明确的科学理由；不要凭空新增要求。')
            feedback_text = (feedback_text + '\n' + extra).strip()
        if await retrieve(feedback.searches):
            evidence = await gather_evidence_notes(question, references(), model, runs)
        # 重新从证据组织回答，避免把前稿误读当作已成立的事实延续。
        answer = await _ask(model, Answer,
            '根据证据重新回答原问题，结合反馈和独立语义审查修正遗漏和误述。'
            '审查未支持的候选只有在新增原文确实补齐缺口后才可继续推荐，否则改为 hold。'
            '推荐具体且有文献支持的配方；不完整或适用条件尚未确认的体系只列为待核实线索。'
            '如果原问题要求比较分支，修订后的报告要保留有意义的已检索比较对象，并说明推荐与 hold 的依据。'
            '任何 recommendation=recommend 的候选都必须填写实际支持它的 references。'
            '不要把前稿内容或审查结论本身当成实验事实。',
            {'original_question': question, **evidence,
             'answer': answer.model_dump(mode='json'), 'feedback': feedback_text,
             'independent_semantic_reviews': semantic_reviews}, runs)
        answer = await ensure_answer_references(question, answer, evidence, model, runs, sources)
        semantic_reviews = await review_answer(answer)
    unsupported, undercalled = review_disagreements(answer, semantic_reviews)
    if unsupported:
        # 反馈预算耗尽后不伪装成“审查通过”；只允许模型据自己的独立审查降级候选。
        previous = {candidate.candidate_id: candidate.composition for candidate in answer.candidates}
        answer = await _ask(model, Answer,
            '这是最终一致性修订，不再新增检索。保持现有候选的 candidate_id 和 composition 不变，'
            '不得新增候选。independent_semantic_reviews 中 recommendation_supported=false 的候选改为 hold；'
            '报告同步删除把它们写成已推荐或已在目标条件验证的表述。其余内容只依据现有原文。',
            {'original_question': question, **evidence,
             'answer': answer.model_dump(mode='json'),
             'independent_semantic_reviews': semantic_reviews}, runs)
        answer = await ensure_answer_references(question, answer, evidence, model, runs, sources)
        if {candidate.candidate_id for candidate in answer.candidates} - set(previous):
            raise ValueError('最终一致性修订引入了未经审查的新候选')
        for candidate in answer.candidates:
            if candidate.candidate_id in previous and candidate.composition != previous[candidate.candidate_id]:
                raise ValueError('最终一致性修订改变了候选组成')
    # provenance 记录最终答案与独立审查之间仍残留的分歧，而不是修订前的中间状态。
    unsupported, undercalled = review_disagreements(answer, semantic_reviews)
    validate_review_alignment(answer, semantic_reviews)
    refs = references()
    artifact = {'schema_version': 'literature_design_workflow_v1', 'task_type': 'design',
        'task_status': 'recommended' if any(c.recommendation == 'recommend' for c in answer.candidates) else 'held',
        'candidates': [c.model_dump(mode='json') for c in answer.candidates],
        'gaps': answer.unresolved_questions, 'retrieved_sources': refs,
        'runtime_provenance': {'workflow': 'retrieve_read_expand_evidence_answer_semantic_review_feedback_retrieve_revise',
            'reference_implementation': ['OpenScholar', 'PaperQA GatherEvidence'],
            'evidence_notes': evidence['evidence_notes'],
            'semantic_candidate_reviews': semantic_reviews,
            'semantic_review_disagreements': {'unsupported_recommendations': unsupported,
                                              'supported_holds': undercalled},
            'model_runs': runs,
            'semantic_validation': 'source_grounded_second_view_requires_external_quality_audit',
            'semantic_review_model_mode': semantic_review_model_mode,
            'semantic_review_model_failures': semantic_review_model_failures,
            'structured_constraint_gate': False}}
    bibliography = '\n\n## 来源\n' + '\n'.join(
        f"[{r['reference']}] {r.get('paper_title') or r.get('paper_id')}（{r['chunk_id']}）" for r in refs)
    report = answer.report + bibliography
    return {'messages': [*messages, AIMessage(content=report)], 'last_report': report,
            'task_artifact': artifact, 'terminal_committed': True}


def build_literature_design_graph(model, tools, *, review_model=None):
    """显式试验入口；图内不调用旧筛选、决定或报告门禁。"""
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver
    from orion.core.state import GraphState
    async def workflow(state, config):
        if config.get('configurable', {}).get('task_type') != 'design':
            raise ValueError('literature 试验入口当前只支持显式 design 任务')
        if not state['messages'] or not isinstance(state['messages'][-1], HumanMessage):
            raise ValueError('缺少本次原始用户问题')
        return await run_workflow(state['messages'][-1].content, model, tools,
                                  review_model=review_model, expand_evidence=True)
    graph = StateGraph(GraphState)
    graph.add_node('literature_design', workflow)
    graph.add_edge(START, 'literature_design'); graph.add_edge('literature_design', END)
    return graph.compile(checkpointer=InMemorySaver())
