"""独立汇总保存的逐次记录；核对计数、文件哈希和来源是否变化。"""
from release_paths import release_path
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / 'runs/20261030_221402_811104_formal'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = load(ROOT / 'manifest.json')
    protocol = load(ROOT / 'formal_protocol.json')
    summary = load(FORMAL / 'summary.json')
    assert summary['phase'] == 'formal' and summary['blocks'] == 6
    assert sha(ROOT / 'manifest.json') == protocol['manifest_sha256']
    assert len(summary['results']) == 36
    checks = {}
    expected_read_hashes = {}
    for source in manifest['sources']:
        assert sha(Path(source['source_result'])) == source['source_result_sha256']
        assert sha(Path(source['source_report'])) == source['source_report_sha256']
        for filename, facts in source['files'].items():
            assert sha(ROOT / 'materials' / source['workspace'] / filename) == facts['sha256']
            expected_read_hashes[(source['workspace'], filename)] = facts['sha256']
    checks['original_and_frozen_materials_unchanged'] = True
    for relative, expected in manifest['runtime_python_sha256'].items():
        assert sha(Path(release_path('${ORION_SOURCE}')) / relative) == expected
    checks['runtime_source_files_unchanged'] = len(manifest['runtime_python_sha256'])
    for filename, expected in protocol['implementation_sha256'].items():
        assert sha(ROOT / filename) == expected
    checks['formal_implementation_unchanged'] = True
    groups = defaultdict(list)
    process_exits = 0
    calls = 0
    reads = 0
    protocol_versions = set()
    server_infos = []
    for result in summary['results']:
        directory = FORMAL / f"block{result['block']}_q{result['quota']}_{result['strategy']}"
        assert load(directory / 'result.json') == {k: v for k, v in result.items() if k != 'block'}
        logs = load(directory / 'adapter_events.json')
        creations = 0
        repeated = 0
        call_count = 0
        closures = 0
        for label, events in logs.items():
            opens = [e for e in events if e['type'] == 'open']
            closes = [e for e in events if e['type'] == 'close']
            commands = [e for e in events if e['type'] == 'call']
            assert len(opens) == len(closes)
            assert [e['generation'] for e in opens] == [e['generation'] for e in closes]
            assert all(e['process_exit_confirmed'] and e['exit_code'] == 0 for e in closes)
            assert all(not e['is_error'] for e in commands)
            for opened in opens:
                protocol_versions.add(opened['protocol_version'])
                server_infos.append(opened['server_info'])
                assert 'read_text_file' in opened['tool_names'] and 'list_directory' in opened['tool_names']
                assert opened['tool_names'] == opens[0]['tool_names']
            creations += len(opens)
            repeated += max(0, len(opens) - 1)
            closures += len(closes)
            call_count += len(commands)
        assert creations == result['creations']
        assert repeated == result['repeated_creations']
        assert closures == result['closures_including_cleanup']
        assert call_count == result['completed_calls']
        assert result['all_created_services_confirmed_closed']
        assert result['ready_boundaries_within_quota']
        for index, batch in enumerate(result['batches']):
            frozen = manifest['batches'][index]
            assert batch['workspace'] == frozen['workspace']
            assert batch['ready_session_count'] <= result['quota']
            assert len(batch['calls']) == len(frozen['calls'])
            for saved, (tool, relative) in zip(batch['calls'], frozen['calls']):
                assert saved['tool'] == tool and saved['relative_path'] == relative
                if tool == 'read_text_file':
                    assert saved['response_sha256'] == expected_read_hashes[(batch['workspace'], relative)]
                    reads += 1
        if result['strategy'] != 'retain_all' or result['quota'] == 2:
            assert result['completed_batches'] == 6 and result['completed_calls'] == 8
        else:
            assert result['completed_batches'] == 2 and result['stopped']['batch'] == 3
        groups[(result['quota'], result['strategy'])].append(result)
        process_exits += closures
        calls += call_count
    rows = []
    for (quota, strategy), trials in sorted(groups.items()):
        assert len(trials) == 6
        for field in ['completed_batches', 'completed_calls', 'creations', 'repeated_creations',
                      'closures_during_sequence', 'closures_including_cleanup']:
            assert len({r[field] for r in trials}) == 1, (quota, strategy, field)
        rows.append({'quota': quota, 'strategy': strategy, 'repeats': len(trials),
                     **{field: trials[0][field] for field in ['completed_batches', 'completed_calls',
                        'creations', 'repeated_creations', 'closures_during_sequence', 'closures_including_cleanup']}})
    preflight = load(Path(protocol['preflight_summary']))
    assert preflight['success'] and all(preflight['checks'].values())
    checks.update({'preflight': preflight['checks'], 'formal_sequences': len(summary['results']),
                   'verified_successful_calls': calls, 'byte_matching_file_reads': reads,
                   'confirmed_server_process_exits': process_exits,
                   'protocol_versions': sorted(protocol_versions),
                   'server_info': server_infos[0], 'groups': rows, 'success': True})
    (ROOT / 'audit.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (ROOT / 'results_by_group.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fields = ['block', 'quota', 'strategy', 'completed_batches', 'completed_calls', 'creations',
              'repeated_creations', 'closures_during_sequence', 'closures_including_cleanup', 'elapsed_s_descriptive_only']
    with (ROOT / 'all_formal_runs.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: result[k] for k in fields} for result in summary['results'])
    report = f'''真实MCP工具会话补充实验结果

结论：已完成真实接入、可行性检查和正式功能对照，可以增加一个真实工具会话资源的验证案例。

一、具体做了什么
使用官方Filesystem MCP服务和官方Python SDK，在已有运行时资源接口上增加独立实验适配器。工具执行列目录和读取证据/报告，会话由同一运行时建立、保留、保护、关闭和重建。
材料来自P2-r1-runtime与U2-r1-runtime的既有推荐任务记录。report.md逐字节复制，evidence.json只序列化原有contexts字段；没有新生成科学内容，没有调用LLM、RAG或写入知识图谱。
客户端MCP 1.30.0，服务包2.1.0，Node {manifest['node_version']}，实际协商协议{', '.join(sorted(protocol_versions))}。环境只新增此前缺失的依赖，原有依赖版本不变，pip check通过。

二、实验设计
前瞻性设置两个分别授权的任务工作区A、B。这是实验的任务隔离条件，不代表历史任务已经通过MCP运行，也不是所有工作流都必然需要两个会话。若一个服务被授权访问两组材料，可以共用一个会话。
固定六批顺序A、A、B、B、A、A，共八次只读工具调用。比较持续保留、每批结束释放、运行时按需回收三种策略，配额分别为1和2个受管会话实例。
六个顺序平衡区组，每个区组六种条件，共36次序列尝试。各策略先后位置出现次数相同，两种配额先后顺序平衡。六个区组用于验证重复性和平衡执行顺序，不用于宣称统计显著的速度优势。

三、正式结果
以下均为每次序列的结果，同组6次结果一致。

会话配额    策略           完成批次    建立总次数    重复建立次数
1           持续保留       2/6         1             0
1           每批结束释放   6/6         6             4
1           运行时         6/6         3             1
2           持续保留       6/6         2             0
2           每批结束释放   6/6         6             4
2           运行时         6/6         2             0

“建立总次数”包含各会话身份的首次建立；“重复建立”只计算首次建立之后的再次建立。未完成的持续保留组不能仅凭建立次数少而被判为更优。
单会话配额下，持续保留组在第三批切换到B时因空闲A仍占配额而停止。运行时关闭空闲A以建立B，第五批再次需要A时重新建立。双会话配额下，运行时保留A、B供后续复用。

四、正确性与实际释放证据
正式调用{calls}次均成功；其中{reads}次文件读取的返回文本重新编码后的哈希与冻结文件原始字节一致，目录结果也在执行时核对。
正式序列共建立{process_exits}个服务进程，全部通过独立Windows进程句柄确认实际退出，退出码均为0。任务内释放与最终清理在原始记录中分别计数。
可行性检查还验证了批内共享、批间复用、服务端授权范围限制、活动租约保护、关闭代次不可用、重建后内容一致，以及配额不足的双会话批次在创建前被拒绝。活动保护探针检验的是持有活动租约期间的冲突请求，没有宣称测量了耗时科学计算中的并行抢占。

五、对论文的作用
新增的是“真实MCP工具会话”的生命周期证据，补充此前模拟会话的验证。它与数据库驱动属于连接/会话这一宽泛家族，不能按软件对象数量直接宣称覆盖了所有独立资源大类。
这里验证无状态只读工具调用在会话重建后能正确继续，没有验证任意远程业务状态、长任务断点或事务状态的恢复。
本实验没有提出新回收策略，没有复现ThunderAgent，也没有比较二者性能。调用耗时保存在原始数据中，只作描述，不增加加速主张。
建议在方法中简要介绍对象、授权条件、配额和对照，在结果中增加一段与小表；实现与逐次日志可放补充材料。不必扩展成一个新的庞大主实验章节。

六、保留记录与当前状态
首次预检查因Python文本模式将CRLF转换为LF而误报内容不一致。核验程序改为直接解码原始字节后通过；首次失败、诊断返回和后续通过记录均保留。
capvm下{len(manifest['runtime_python_sha256'])}份运行时源码哈希与冻结前完全一致。原始任务记录和复制材料哈希不变。新增适配器、脚本和官方服务依赖均在独立实验目录。
当前英文稿未修改。本文档是实测结果记录，不将实验方案当作结果，不将功能验证当作所有资源或所有科学任务的普遍性能结论。

主要记录
正式方案：formal_protocol.json
来源与冻结材料：manifest.json
审计：audit.json
分组结果：results_by_group.csv
全部36次序列：all_formal_runs.csv
正式原始记录：{FORMAL}

设计依据沿用已读Experimental Design和Scientific Critical Thinking技能；设计来源见上一轮完整方案。实现参考为官方MCP Python SDK及Filesystem参考服务。
https://github.com/modelcontextprotocol/python-sdk
https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem
'''
    (ROOT / '真实MCP工具会话实验结果.txt').write_text(report, encoding='utf-8')
    proposed = '''Suggested Methods addition, pending manuscript placement and terminology checks

A supplementary experiment evaluated real Model Context Protocol tool sessions using the official filesystem server. The sessions provided read-only access to evidence and reports saved from two completed electrolyte recommendation tasks. Each task had a separately authorized workspace. A fixed sequence of six batches accessed the workspaces in the order A, A, B, B, A, A. Three strategies were compared under quotas of one and two managed sessions. The strategies retained all sessions, closed idle sessions after every batch, or used the runtime layer to retain and reclaim sessions as needed. Six blocks balanced strategy order and quota order. Completion, session creation, confirmed closure, and output consistency were recorded.

Suggested Results addition

Under the one-session quota, the runtime layer completed all six batches with three session creations, including one repeated creation. Closing sessions after every batch also completed the sequence but required six creations, including four repeated creations. Retaining all sessions completed only the first two batches because the session for workspace A still occupied the quota when workspace B was requested. Under the two-session quota, both retention and the runtime layer completed the sequence with two creations and no repeated creations. Counts were identical across the six blocks. All successful file reads matched the frozen input files, and every created server process was confirmed to have exited after closure or final cleanup.

Editorial note

These additions describe a prospective controlled file-access experiment using existing scientific task artifacts. They do not describe the historical recommendation traces as MCP traces. The quota counts managed session instances, not network connections or physical memory. The evidence supports real external-tool session lifecycle management and output preservation after reconnecting, rather than checkpointing arbitrary application state. No changes have yet been inserted into the manuscript.
'''
    (ROOT / 'proposed_manuscript_addition.txt').write_text(proposed, encoding='utf-8')
    print(json.dumps({k: checks[k] for k in ['success', 'formal_sequences', 'verified_successful_calls',
        'byte_matching_file_reads', 'confirmed_server_process_exits', 'runtime_source_files_unchanged', 'protocol_versions']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
