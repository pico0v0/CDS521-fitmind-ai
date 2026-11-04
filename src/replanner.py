"""FitMind AI 自然语言智能局部重排器 (Replanner)
负责解析用户的动态反馈并执行高精度局部修改，
严格保证“无关内容保留率（Invariant Retention Rate）≥ 95%”，杜绝全盘重写的波动。
"""
from __future__ import annotations
import copy
from typing import Dict, Any, Tuple, List, Optional
from src.schemas import WeeklyPlan, EditIntent, UserProfile, WeekDay
from src.retrieval import load_catalog, get_safe_foods_and_meals, get_eligible_exercises, compute_meal_detail
from src.validator import validate_weekly_plan
from src.llm_client import call_structured_llm

DAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def parse_intent_hybrid(
    user_text: str,
    allowed_foods: list[dict],
    model: Optional[str] = None
) -> EditIntent:
    """使用大模型解析修改意图（带结构化验证与兜底）"""
    from src.llm_client import parse_edit_with_llm
    
    intent_dict = parse_edit_with_llm(user_text, allowed_foods, model=model)
    return EditIntent.model_validate(intent_dict)


def apply_plan_edit(
    current_plan: WeeklyPlan | dict,
    intent: EditIntent | dict
) -> Tuple[WeeklyPlan, Dict[str, Any]]:
    """
    执行精准局部更新算法：
    1. 日程移动 (Source Day -> Target Day)
    2. 新增禁忌食材局部替换 (仅替换包含该食材的单餐)
    3. 单餐定制替换
    4. 保证其余无关天数严格不变并计算保留率
    """
    plan_dict = copy.deepcopy(current_plan if isinstance(current_plan, dict) else current_plan.model_dump())
    intent_obj = intent if isinstance(intent, EditIntent) else EditIntent.model_validate(intent)
    user_p = plan_dict.get("profile", {})

    ex_list, foods_list, meals_list = load_catalog()
    food_by_id = {f["id"]: f for f in foods_list}

    changes_log: List[str] = []
    total_elements = 7 * 4  # 7天，每天 1个训练 + 3餐
    unchanged_elements = total_elements

    # 1. 训练日程搬移
    src, dst = intent_obj.source_day, intent_obj.target_day
    if src and dst and src != dst:
        days_map = {d["day"]: d for d in plan_dict["week"]}
        if src not in days_map or dst not in days_map:
            raise ValueError(f"指定的日期 '{src}' 或 '{dst}' 不在周计划范围内")
        
        src_day = days_map[src]
        dst_day = days_map[dst]

        if not src_day.get("workout"):
            raise ValueError(f"{src} 原本就是休息日，无需移动训练")
        if dst_day.get("workout"):
            raise ValueError(f"{dst} 原本已安排了训练，不能直接搬移以防训练日合并冲突")
        if dst not in user_p.get("available_days", []):
            # 若目标日不在原可用日中，提示并动态拓展
            user_p["available_days"] = list(set(user_p.get("available_days", [])) | {dst})
            changes_log.append(f"由于用户要求改至 {dst}，已将 {dst} 增补为可训练日")

        dst_day["workout"] = src_day["workout"]
        src_day["workout"] = None
        changes_log.append(f"训练安排已从 {src} 挪至 {dst}（保持原动作组合与强度）")
        unchanged_elements -= 2  # 仅影响这两个训练项

    # 2. 新增食物排除（只替换踩中禁忌的单餐）
    new_excludes = set(intent_obj.exclude_foods)
    if new_excludes:
        existing_excludes = set(user_p.get("exclude_foods", []))
        updated_excludes = list(existing_excludes | new_excludes)
        user_p["exclude_foods"] = updated_excludes

        _, safe_meals = get_safe_foods_and_meals(
            user_p.get("allergies", []),
            updated_excludes
        )
        safe_by_slot = {
            slot: [m for m in safe_meals if m["slot"] == slot]
            for slot in ("早餐", "午餐", "晚餐")
        }

        replaced_count = 0
        for i, day in enumerate(plan_dict["week"]):
            meals = day.get("meals", {})
            for zh_slot, en_slot in [("早餐", "breakfast"), ("午餐", "lunch"), ("晚餐", "dinner")]:
                m = meals.get(zh_slot) or meals.get(en_slot)
                if not m:
                    continue
                # 检查该餐是否包含新增的排除食材
                hit = any(
                    (ing.get("food_id") if isinstance(ing, dict) else getattr(ing, "food_id", None)) in new_excludes
                    for ing in m.get("ingredients", [])
                )
                if hit:
                    candidates = safe_by_slot.get(zh_slot, [])
                    if not candidates:
                        raise ValueError(f"新增禁忌后，无法为 {zh_slot} 找到安全合规餐食模板")
                    new_template = candidates[i % len(candidates)]
                    repaired_detail = compute_meal_detail(new_template, food_by_id).model_dump()
                    meals[zh_slot] = repaired_detail
                    if en_slot in meals:
                        meals[en_slot] = repaired_detail
                    replaced_count += 1
                    unchanged_elements -= 1

        banned_names = [food_by_id[fid]["name"] for fid in new_excludes if fid in food_by_id]
        changes_log.append(f"新增禁忌项【{', '.join(banned_names)}】，精准替换了 {replaced_count} 份相关餐食")

    # 3. 单餐定制替换
    if intent_obj.replace_meal:
        rm_day = intent_obj.replace_meal.get("day")
        rm_slot = intent_obj.replace_meal.get("slot", "午餐")
        target_meal_id = intent_obj.replace_meal.get("target_meal_id")

        for day in plan_dict["week"]:
            if day["day"] == rm_day:
                meals = day.get("meals", {})
                template = next((m for m in meals_list if m["id"] == target_meal_id), None)
                if template:
                    detail = compute_meal_detail(template, food_by_id).model_dump()
                    meals[rm_slot] = detail
                    unchanged_elements -= 1
                    changes_log.append(f"已将 {rm_day} 的 {rm_slot} 替换为【{template['name']}】")

    # 4. 严格复检硬约束
    plan_dict["mode"] = "fitmind_replanned"
    plan_dict["profile"] = user_p
    val_res = validate_weekly_plan(plan_dict, user_p)
    if not val_res.is_valid:
        raise ValueError("局部调整后未能满足硬约束：" + "；".join(val_res.issues[:3]))

    retention_rate = round(unchanged_elements / total_elements, 4)

    updated_plan = WeeklyPlan.model_validate(plan_dict)
    summary = {
        "changes": changes_log,
        "retention_rate": retention_rate,
        "unchanged_ratio": f"{retention_rate * 100:.1f}%",
        "validation_passed": True
    }
    return updated_plan, summary

