"""可选的真实 AI 模块：用 LLM 理解修改意图，修改动作仍由受约束的 Python 程序执行。"""
from __future__ import annotations
import os
from pydantic import BaseModel, ConfigDict

class EditIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_day: str | None
    target_day: str | None
    exclude_foods: list[str]
    explanation: str


def parse_edit_with_llm(text, allowed_foods, model=None):
    """需要 OPENAI_API_KEY；未配置或 API 不可用时应在 UI 明确告知。"""
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY；不能调用真实大模型。")
    sys = ("你是健身饮食应用的编辑意图解析器，仅做信息提取，不提供医疗建议。"
           "对未明确提到的字段填写 null 或空列表。周几统一格式为周一至周日；"
           "食物禁忌必须从允许的 food_id 中选择；不要编造食物 ID。"
           f"允许的食品：{allowed_foods}")
    response = OpenAI().responses.parse(
        model=model or os.getenv("FITMIND_MODEL", "gpt-4o-mini"),
        input=[{"role":"system","content":sys}, {"role":"user","content":text}],
        text_format=EditIntent)
    if response.output_parsed is None:
        raise RuntimeError("模型没有返回可用的结构化编辑意图。")
    return response.output_parsed.model_dump()
