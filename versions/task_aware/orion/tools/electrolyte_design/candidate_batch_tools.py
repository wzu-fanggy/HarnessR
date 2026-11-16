"""候选文件与既有预测后端的原生工具入口，不维护第二套科学模型。"""

from langchain_core.runnables import RunnableConfig

from langchain_core.tools import tool



from . import candidate_batches, prediction_records





@tool

def evaluate_candidate_batch(request: dict, config: RunnableConfig) -> dict:

    """评价已保存候选批次在一个温度下的电导率，只返回覆盖统计及靠前项。



    request给出生成工具实际返回的batch_id、temperature_c，及

    objective="maximise_predicted_conductivity"。可选top_k为1–20，

    max_evaluations限制实际评价数；省略时评价整批。完整输出保存在文件中。

    预览未经评价；覆盖不足不能称为筛完全部空间。结果为模型估计，不能代替实验验证。

    """

    return candidate_batches.evaluate_candidate_batch(request, config=config)





@tool

def inspect_candidate_artifact(request: dict) -> dict:

    """分页读取完整候选或已保存评价，不重新计算。



    request给出batch_id或evaluation_id，二者仅选其一；可选offset、limit（1–20）。

    评价可用order="predicted_rank"或"generation"；生成批次仅用generation。

    candidate_ref包含批次身份，同名candidate_id不保证跨批组成相同。

    """

    return candidate_batches.inspect_candidate_artifact(request)





@tool

def compare_conductivity_predictions(request: dict) -> dict:

    """读取两条实际保存的预测记录，核对同配方、同模型的跨温度可比性。



    request给出left_record_id和right_record_id，不接受手工改写预测值。

    仅在组成、模型版本和其他显式条件一致且温度不同时返回compared及比值。

    not_comparable表示比较依据不成立，不表示配方性能差；不能重标旧结果充当新计算。

    """

    return prediction_records.compare_saved_predictions(request)

