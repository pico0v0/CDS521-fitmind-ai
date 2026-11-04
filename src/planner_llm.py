"""FitMind AI 大模型周计划生成器 (LLM Planner)
结合用户画像、确定性候选过滤与结构化 JSON Schema 输出，
生成科学个性化、且严格受约束的 7 天运动与营养方案。
"""
from __future__ import annotations
import json
import time
from typing import Dict, Any, Tuple
from src.schemas import UserProfile, WeeklyPlan
from src.retrieval import get_eligible_exercises, get_safe_foods_and_meals, load_catalog, compute_meal_detail
from src.validator import validate_weekly_plan, auto_repair_plan
from src.llm_client import call_structured_llm


def generate_weekly_plan_llm(
    profile: UserProfile | dict,
    model: str | None = None
) -> Tuple[WeeklyPlan, Dict[str, Any]]:
    """
    FitMind AI 完整方法 (Proposed Method)：
    1. 候选池过滤检索 (Eligible Exercises & Safe Meals)
    2. LLM 结构化规划生成 (Strict JSON Schema)
    3. 硬约束校验与自动修复 (Validator & Auto-Repair)
    返回 (WeeklyPlan, meta_info: {latency_ms, tokens_in, tokens_out, raw_issues, repaired})
    """
    user_p = profile if isinstance(profile, UserProfile) else UserProfile.model_validate(profile)
    p_dict = user_p.model_dump()

    # 1. 确定性候选检索
    eligible_ex = get_eligible_exercises(user_p.equipment, user_p.experience)
    safe_food_by_id, safe_meals = get_safe_foods_and_meals(user_p.allergies, user_p.exclude_foods)

    ex_summary = [{"id": e["id"], "name": e["name"], "muscle": e["muscle"], "equipment": e["equipment"]} for e in eligible_ex]
    meal_summary = [{"id": m["id"], "name": m["name"], "slot": m["slot"], "ingredients": m["ingredients"]} for m in safe_meals[:12]]

    # 2. 构建 Prompt
    system_prompt = (
        "你是一个精通运动训练学与临床营养学的专业 AI 规划专家（FitMind AI）。\n"
        "你的职责是为目标用户制定一份兼具科学性、个性化与可执行性的 7 天健身与三餐饮食方案。\n\n"
        "【硬性系统约束 - 必须 100% 遵守】：\n"
        f"1. 训练日约束：必须在用户的可用训练日中正好安排 {user_p.sessions} 天训练，其余天数为休息恢复日（workout 设为 null）。允许训练日：{user_p.available_days}。\n"
        f"2. 动作器械约束：用户器械为 '{user_p.equipment}'，所选动作 ID 必须全部来自下方的【合法候选动作列表】，绝对不可越界使用未持有的器械！\n"
        "3. 饮食安全约束：严禁包含用户的过敏原与排除食物。每餐可以选用【合法餐食模板】或搭配安全食材。\n"
        "4. 数值生成加速：三餐营养数据（卡路里/蛋白质/脂肪/碳水）填写合理预估值即可，后台确定性引擎会自动进行精确复算校验与物理守恒对齐，请专注结构搭配以加快响应速度。\n"
        "5. 7天完整性：必须且只能覆盖周一至周日共 7 天。\n"
        "6. 输出精炼规范：每餐搭配 2-3 种核心食材；explanation 字段控制在 250 字左右，使用 Markdown 标题和列表排版（训练策略、营养赤字、执行要点），避免长篇大论堆砌。\n"
    )

    user_prompt = (
        f"【用户画像】：\n"
        f"- 年龄：{user_p.age} 岁 | 身高：{user_p.height_cm} cm | 体重：{user_p.weight_kg} kg\n"
        f"- 健身目标：{user_p.goal} | 经验水平：{user_p.experience}\n"
        f"- 可用器械：{user_p.equipment} (最高可用级别)\n"
        f"- 每周计划训练频次：{user_p.sessions} 次\n"
        f"- 本周允许训练的日期：{user_p.available_days}\n"
        f"- 食物过敏：{user_p.allergies or '无'}\n"
        f"- 排除食物：{user_p.exclude_foods or '无'}\n\n"
        f"【可选合法动作候选池】（共 {len(ex_summary)} 个动作，仅可从此处选取 ID）：\n"
        f"{json.dumps(ex_summary, ensure_ascii=False)}\n\n"
        f"【安全合规餐食模板库】（供参考选用）：\n"
        f"{json.dumps(meal_summary, ensure_ascii=False)}\n\n"
        f"请生成结构化 7 天规划，并附带针对该用户特质的科学解释与指导建议。"
    )

    t0 = time.time()
    raw_plan_obj, meta = call_structured_llm(
        response_model=WeeklyPlan,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model
    )
    raw_dict = raw_plan_obj.model_dump()
    raw_dict["mode"] = "fitmind_llm"
    raw_dict["profile"] = p_dict

    # 3. 运行确定性规则审查
    val_result = validate_weekly_plan(raw_dict, p_dict)
    meta["raw_issues"] = val_result.issues
    meta["raw_passed"] = val_result.is_valid

    # 4. 自动修复机制
    if not val_result.is_valid:
        repaired_dict = auto_repair_plan(raw_dict, p_dict)
        final_val = validate_weekly_plan(repaired_dict, p_dict)
        meta["repaired"] = True
        meta["final_issues"] = final_val.issues
        final_dict = repaired_dict
    else:
        meta["repaired"] = False
        meta["final_issues"] = []
        final_dict = raw_dict

    final_plan = WeeklyPlan.model_validate(final_dict)
    return final_plan, meta

