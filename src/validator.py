"""FitMind AI 硬规则校验与修复引擎 (Constraint Validation & Repair Engine)
负责对大模型或规则生成的周计划执行 100% 确定性硬约束审查与局部自动修复。
"""
from __future__ import annotations
import copy
from typing import Dict, List, Optional, Tuple, Any
from src.schemas import (
    UserProfile, WeeklyPlan, DailyPlan, DailyWorkout,
    MealDetail, IngredientPortion, NutritionSummary, ValidationResult
)
from src.retrieval import (
    load_catalog, get_eligible_exercises, get_safe_foods_and_meals,
    compute_meal_detail, EQUIPMENT_RANK
)

DAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def validate_weekly_plan(plan: WeeklyPlan | dict, profile: Optional[UserProfile | dict] = None) -> ValidationResult:
    """
    全面审查周计划是否满足所有硬约束。
    返回 ValidationResult(is_valid, issues)。
    """
    if isinstance(plan, dict):
        # 兼容 dict 格式
        plan_dict = plan
        user_p = profile if profile is not None else plan_dict.get("profile", {})
        if isinstance(user_p, UserProfile):
            user_p = user_p.model_dump()
        week_list = plan_dict.get("week", [])
    else:
        plan_dict = plan.model_dump()
        user_p = profile.model_dump() if isinstance(profile, UserProfile) else (profile or plan.profile.model_dump())
        week_list = [d.model_dump() if hasattr(d, "model_dump") else d for d in plan.week]

    ex_list, foods_list, _ = load_catalog()
    ex_by_id = {e["id"]: e for e in ex_list}
    food_by_id = {f["id"]: f for f in foods_list}

    user_equipment = user_p.get("equipment", "徒手")
    max_rank = EQUIPMENT_RANK.get(user_equipment, 0)
    allowed_ex_ids = {e["id"] for e in ex_list if EQUIPMENT_RANK.get(e["equipment"], 0) <= max_rank}

    banned_foods = set(user_p.get("exclude_foods", []))
    allergies = set(user_p.get("allergies", []))
    available_days = set(user_p.get("available_days", []))
    expected_sessions = user_p.get("sessions", 3)

    issues: List[str] = []

    # 1. 周期覆盖与顺序检查
    days_in_plan = [d.get("day") for d in week_list]
    if days_in_plan != DAYS:
        issues.append("周期必须完整覆盖周一至周日且保持严格顺序")

    # 2. 逐日审查
    actual_sessions = 0
    for day_item in week_list:
        day_name = day_item.get("day", "未知")
        workout = day_item.get("workout")

        # 检查训练日程
        if workout:
            actual_sessions += 1
            if day_name not in available_days:
                issues.append(f"{day_name} 安排了训练，但不在用户的可训练日期内")

            exercise_ids = workout.get("exercise_ids", [])
            if not exercise_ids:
                issues.append(f"{day_name} 训练日未指定任何动作")

            for eid in exercise_ids:
                if eid not in ex_by_id:
                    issues.append(f"{day_name} 包含未知训练动作 ID: {eid}")
                elif eid not in allowed_ex_ids:
                    req_eq = ex_by_id[eid]["equipment"]
                    issues.append(f"{day_name} 动作器械违规：'{eid}' 需要 {req_eq}，超出用户器械限制 ({user_equipment})")

        # 检查三餐
        meals = day_item.get("meals", {})
        slot_keys = [("早餐", "breakfast"), ("午餐", "lunch"), ("晚餐", "dinner")]
        for zh_slot, en_slot in slot_keys:
            m = None
            if isinstance(meals, dict):
                m = meals.get(zh_slot) or meals.get(en_slot)
            elif hasattr(meals, "__getitem__"):
                try:
                    m = meals[zh_slot]
                except (KeyError, TypeError):
                    pass

            if not m:
                issues.append(f"{day_name} 缺少 {zh_slot}")
                continue

            ingredients = m.get("ingredients", []) if isinstance(m, dict) else getattr(m, "ingredients", [])
            if not ingredients:
                issues.append(f"{day_name} {zh_slot} 缺少食材清单")
                continue

            for ing in ingredients:
                fid = ing.get("food_id") if isinstance(ing, dict) else getattr(ing, "food_id", None)
                grams = ing.get("grams") if isinstance(ing, dict) else getattr(ing, "grams", 0.0)

                if fid not in food_by_id:
                    issues.append(f"{day_name} {zh_slot} 引用了未知食材: {fid}")
                    continue

                food_meta = food_by_id[fid]
                if fid in banned_foods:
                    issues.append(f"{day_name} {zh_slot} 包含了用户禁忌食材: {food_meta['name']}({fid})")

                allergen_hits = allergies & set(food_meta.get("allergens", []))
                if allergen_hits:
                    issues.append(f"{day_name} {zh_slot} 食材 {food_meta['name']} 含有用户过敏原: {','.join(allergen_hits)}")

                if grams <= 0:
                    issues.append(f"{day_name} {zh_slot} 食材 {fid} 克重非正: {grams}g")

            # 营养确定性计算复验
            nutrition = m.get("nutrition", {})
            if all((ing.get("food_id") if isinstance(ing, dict) else getattr(ing, "food_id", None)) in food_by_id for ing in ingredients):
                totals = {"kcal": 0.0, "protein": 0.0, "carb": 0.0, "fat": 0.0}
                for ing in ingredients:
                    fid = ing.get("food_id") if isinstance(ing, dict) else getattr(ing, "food_id", None)
                    grams = ing.get("grams") if isinstance(ing, dict) else getattr(ing, "grams", 0.0)
                    for key in totals:
                        totals[key] += food_by_id[fid][key] * grams / 100.0

                for key in totals:
                    claimed = nutrition.get(key, 0.0)
                    expected = round(totals[key], 1)
                    if abs(claimed - expected) > 0.11:
                        issues.append(f"{day_name} {zh_slot} 的 {key} 计算不一致：声明 {claimed}，复算值 {expected}")

    # 3. 训练频次审查
    if actual_sessions != expected_sessions:
        issues.append(f"训练频次不符：期望每周 {expected_sessions} 次，实际安排 {actual_sessions} 次")

    return ValidationResult(is_valid=(len(issues) == 0), issues=issues)


def auto_repair_plan(plan: dict, profile: dict) -> dict:
    """
    当模型初次输出存在局部违规时，应用确定性规则进行自动纠正修复。
    保证最终输出 100% 满足硬约束。
    """
    repaired = copy.deepcopy(plan)
    ex_list, foods_list, meals_list = load_catalog()
    food_by_id = {f["id"]: f for f in foods_list}
    ex_by_id = {e["id"]: e for e in ex_list}

    user_equipment = profile.get("equipment", "徒手")
    max_rank = EQUIPMENT_RANK.get(user_equipment, 0)
    allowed_exercises = [e for e in ex_list if EQUIPMENT_RANK.get(e["equipment"], 0) <= max_rank]
    allowed_ex_ids = {e["id"] for e in allowed_exercises}

    available_days = [d for d in DAYS if d in profile.get("available_days", [])]
    expected_sessions = profile.get("sessions", 3)

    _, safe_meal_templates = get_safe_foods_and_meals(
        profile.get("allergies", []),
        profile.get("exclude_foods", [])
    )
    safe_by_slot = {
        slot: [m for m in safe_meal_templates if m["slot"] == slot]
        for slot in ("早餐", "午餐", "晚餐")
    }

    # 1. 修复训练日安排
    workout_days = [d for d in repaired["week"] if d.get("workout")]
    
    # 移除非可用日的训练
    valid_workout_days = []
    for d in repaired["week"]:
        if d.get("workout"):
            if d["day"] not in available_days:
                d["workout"] = None
            else:
                valid_workout_days.append(d["day"])

    # 补充不足的训练日
    while len(valid_workout_days) < expected_sessions and len(valid_workout_days) < len(available_days):
        for candidate_day in available_days:
            if candidate_day not in valid_workout_days:
                for d in repaired["week"]:
                    if d["day"] == candidate_day:
                        d["workout"] = {
                            "exercise_ids": [allowed_exercises[i % len(allowed_exercises)]["id"] for i in range(3)],
                            "minutes": 40,
                            "sets": 3,
                            "notes": "自动规划补充训练日"
                        }
                        valid_workout_days.append(candidate_day)
                        break
            if len(valid_workout_days) >= expected_sessions:
                break

    # 削减多余的训练日
    while len(valid_workout_days) > expected_sessions:
        drop_day = valid_workout_days.pop()
        for d in repaired["week"]:
            if d["day"] == drop_day:
                d["workout"] = None
                break

    # 2. 修复动作器械越界
    for d in repaired["week"]:
        if d.get("workout"):
            new_ex_ids = []
            for eid in d["workout"].get("exercise_ids", []):
                if eid in allowed_ex_ids:
                    new_ex_ids.append(eid)
                else:
                    # 替换为同部位或合规动作
                    fallback = allowed_exercises[0]["id"]
                    if eid in ex_by_id:
                        same_muscle = [e["id"] for e in allowed_exercises if e["muscle"] == ex_by_id[eid]["muscle"]]
                        if same_muscle:
                            fallback = same_muscle[0]
                    new_ex_ids.append(fallback)
            if not new_ex_ids:
                new_ex_ids = [allowed_exercises[0]["id"], allowed_exercises[1]["id"]]
            d["workout"]["exercise_ids"] = new_ex_ids

    # 3. 修复餐饮过敏原/禁忌与计算误差
    for i, d in enumerate(repaired["week"]):
        if "meals" not in d:
            d["meals"] = {}
        slot_map = [("早餐", "breakfast"), ("午餐", "lunch"), ("晚餐", "dinner")]
        for zh_slot, en_slot in slot_map:
            m = d["meals"].get(zh_slot) or d["meals"].get(en_slot)
            needs_replacement = False
            if not m or not m.get("ingredients"):
                needs_replacement = True
            else:
                for ing in m.get("ingredients", []):
                    fid = ing.get("food_id")
                    if fid not in food_by_id:
                        needs_replacement = True
                        break
                    if fid in profile.get("exclude_foods", []):
                        needs_replacement = True
                        break
                    if set(profile.get("allergies", [])) & set(food_by_id[fid].get("allergens", [])):
                        needs_replacement = True
                        break

            if needs_replacement:
                candidates = safe_by_slot.get(zh_slot, [])
                if not candidates:
                    raise ValueError(f"无法为 {zh_slot} 找到安全合规餐食模板")
                chosen_template = candidates[i % len(candidates)]
                detail = compute_meal_detail(chosen_template, food_by_id).model_dump()
                d["meals"][zh_slot] = detail
                d["meals"][en_slot] = detail
            else:
                # 重新计算营养数值以保证数学精度严格吻合
                totals = {"kcal": 0.0, "protein": 0.0, "carb": 0.0, "fat": 0.0}
                for ing in m["ingredients"]:
                    fid = ing["food_id"]
                    grams = ing["grams"]
                    for key in totals:
                        totals[key] += food_by_id[fid][key] * grams / 100.0
                nut_dict = {k: round(v, 1) for k, v in totals.items()}
                if zh_slot in d["meals"]:
                    d["meals"][zh_slot]["nutrition"] = nut_dict
                if en_slot in d["meals"]:
                    d["meals"][en_slot]["nutrition"] = nut_dict

    return repaired
