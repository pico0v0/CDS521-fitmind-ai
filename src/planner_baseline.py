"""基准实验组：Direct LLM 规划器 (Baseline)
用于实验评估：在相同大模型、相同输入画像和相同 JSON Schema 约束下，
直接由模型端到端生成，不提供外部候选池过滤，也不进行后置规则修复。
以此衡量单纯大语言模型在复杂健康约束下的真实遵循率与幻觉水平。
"""
from __future__ import annotations
import json
import time
from typing import Dict, Any, Tuple
from src.schemas import UserProfile, WeeklyPlan
from src.validator import validate_weekly_plan
from src.llm_client import call_structured_llm


def generate_weekly_plan_baseline(
    profile: UserProfile | dict,
    model: str | None = None
) -> Tuple[WeeklyPlan, Dict[str, Any]]:
    """
    基线方法 (Direct LLM Baseline):
    - 无外部动作库与食品库的预先检索过滤
    - 无后置规则校验拦截与自动修复
    - 记录模型的原始输出及违规情况
    """
    user_p = profile if isinstance(profile, UserProfile) else UserProfile.model_validate(profile)
    p_dict = user_p.model_dump()

    system_prompt = (
        "你是一个专业的健身与营养 AI 助手。\n"
        "请根据用户画像生成一份包含连续 7 天（周一至周日）的详细训练与三餐饮食方案。\n"
        "请严格遵守用户的可用训练天数、训练频次、器械条件与饮食禁忌/过敏要求。\n"
        "每个餐食的食材需注明克重并估算营养数值。"
    )

    user_prompt = (
        f"【用户画像】：\n"
        f"- 年龄：{user_p.age} 岁 | 身高：{user_p.height_cm} cm | 体重：{user_p.weight_kg} kg\n"
        f"- 健身目标：{user_p.goal} | 经验水平：{user_p.experience}\n"
        f"- 可用器械：{user_p.equipment}\n"
        f"- 每周计划训练频次：{user_p.sessions} 次\n"
        f"- 允许训练的日期：{user_p.available_days}\n"
        f"- 食物过敏：{user_p.allergies or '无'}\n"
        f"- 排除食物：{user_p.exclude_foods or '无'}\n\n"
        f"请直接生成符合 JSON 模式的 7 天周计划。"
    )

    raw_plan_obj, meta = call_structured_llm(
        response_model=WeeklyPlan,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model
    )
    plan_dict = raw_plan_obj.model_dump()
    plan_dict["mode"] = "baseline_direct_llm"
    plan_dict["profile"] = p_dict

    # 运行同样的校验器评测其违规表现，但不做自动修复
    val_result = validate_weekly_plan(plan_dict, p_dict)
    meta["raw_issues"] = val_result.issues
    meta["raw_passed"] = val_result.is_valid
    meta["repaired"] = False
    meta["final_issues"] = val_result.issues

    final_plan = WeeklyPlan.model_validate(plan_dict)
    return final_plan, meta

