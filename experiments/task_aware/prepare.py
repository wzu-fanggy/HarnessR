"""冻结开发后评测协议；不按评测结果改任务、规则或运行顺序。"""
from release_paths import release_path
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import shutil

root=Path(__file__).resolve().parent
repo=Path(release_path('${ORION_SOURCE}'))
base=root/'async-comparison-v1'
base.mkdir(exist_ok=False)
pilot=json.loads((root/'generic-goal-pilot/protocol.json').read_text(encoding='utf-8'))
cases=pilot['cases']
data=json.dumps(cases,ensure_ascii=False)
common=('数据来自已归档Zenodo测量，保持全部10个温度点、组成及原值，不平均、不删点、不补值。'
        '不查文献，不调用子代理。明确区分测量值、拟合参数与模型估计。'
        '拟合比较Arrhenius与VTF的AICc和留一温度误差。0.1仅为本次工作条件，不是通用科学标准。'
        '这些配方可能出现在预测模型训练集中，不能称为独立留出验证。')
prompts={
 'point':('请分别分析下面两种电解液的温度测量。'+common+
           '对每种配方，若AICc优选模型的loo_rmse_log10低于0.1，再用独立电导率模型对20℃作计算核对；'
           '先看到拟合结果再判断条件。没有最低电导率门槛，不需作达标排序。'
           '最后比较两种配方并说明证据边界。树间差异只描述模型分歧，不解释成已校准预测区间。\n'+data),
 'ranking':('请评价下面两种电解液能否用于一个演示筛选条件：在0、20、40℃均达到3.0 mS/cm。'+common+
             '先检查每种配方的温度测量；若两种配方的AICc优选模型loo_rmse_log10均低于0.1，'
             '再按演示条件对两者作多温度稳健排序，使用完整可用集成成员。'
             '若条件不满足则报告不能继续的原因，不补造或修改输入。'
             '3.0 mS/cm是本任务明确给定的演示阈值，不是领域通用标准。'
             '最后报告排序、逐温度估计与适用边界；集成达标频率不是已校准的物理概率。\n'+data)
}
order=[]
arms=['lru','cost','sync','async']
# 轮换顺序避免固定的热机、网络或机器负载顺序偏向某一组。
for repeat in range(3):
    for case_index,case in enumerate(prompts):
        shifted=arms[(repeat+case_index)%4:]+arms[:(repeat+case_index)%4]
        for arm in shifted:
            order.append({'id':f'{case}-r{repeat}-{arm}','case':case,'repeat':repeat,
                          'arm':arm,'process_slots':2})
for case in prompts:
    order.append({'id':f'{case}-ample-async','case':case,'repeat':0,'arm':'async','process_slots':3})
sources=list(pilot['source_hashes'])
sources+=['tools/electrolyte_design/robust_ranking_tool.py',
          'tools/electrolyte_design/robust_ranking.py',
          'tools/electrolyte_design/robust_ranking_process.py',
          'tools/electrolyte_design/conductivity_ml.py']
source_hashes={}
for rel in sources:
    target=base/'frozen-sources'/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(repo/rel,target)
    source_hashes[rel]=hashlib.sha256(target.read_bytes()).hexdigest()
protocol={
 'scope':'post_development_real_agent_runtime_comparison_not_heldout_science',
 'cases':cases,'prompts':prompts,'order':order,
 'arms':{'lru':'原容量LRU；全部语义顾问关闭',
         'cost':'原执行器与实测重建成本顾问；不请求Jev',
         'sync':'任务语义、实测成本、最小最大后悔值；准入时同步解释',
         'async':'与sync相同；在已满容量且有至少两个已知成本的空闲实例时机会性预取语义'},
 'development_disclosure':['任务配方和点预测分支已用于开发试跑，不作为留出科学验证。',
     '最小最大后悔值是采用已有稳健决策准则，不主张其数学方法原创。',
     '预热库存模拟前任务遗留的空闲资源，预热成本另记；本实验不是冷启动比较。',
     '任务仅两次科学调用批次，不称长程科研任务；检验资源阶段转换。'],
 'max_primary_requests_per_run':16,'max_run_seconds':240,
 'max_live_jev_calls_total':12,'max_live_jev_calls_per_run':1,'request_retries':0,
 'warm_inventory':pilot['warm_inventory'],
 'allowed_tools':pilot['allowed_tools']+['rank_formulations_under_uncertainty'],
 'interface_mode':'完整生产任务工具说明；提示仅给科学目标与条件，不要求建卡、写意图、选择资源或固定调用序列。',
 'selection_rule':'minimax_regret',
 'latency_calibration':pilot['latency_calibration'],
 'source_hashes':source_hashes,
 'primary_metric':'实际graph.ainvoke任务交付耗时，含主模型、科学计算、Jev与检查点；全部运行和失败保留。',
 'secondary_metrics':['主模型请求数和等待时间','工具边界时间','Jev总调用与关键路径等待',
                      '实际进程建立/重载次数和耗时','科学参数与输出一致性','调用结束清理'],
 'stop_rule':'预算到达、鉴权失败或连续两次基础设施错误时停止；科学无优势不停止、不选跑。',
 'analysis_rule':'分支分别报告全部三次数据和中位数；不以局部控制开销替代端到端；不作小样本显著性声明。'
}
(base/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print('Frozen',len(order),'runs; Jev upper limit',protocol['max_live_jev_calls_total'])
