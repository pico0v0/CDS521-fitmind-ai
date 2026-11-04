"""FitMind AI 教学脚手架：默认只运行本地规则 Demo；无模型时不能称为 LLM 生成结果。"""
from __future__ import annotations
import copy, itertools, json, re
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def read_catalog():
    return tuple(json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))
                 for name in ("exercises", "foods", "meals"))


def eligible_exercises(equipment, exercises):
    rank = {"徒手": 0, "哑铃": 1, "健身房": 2}
    return [e for e in exercises if rank[e["equipment"]] <= rank[equipment]]


def safe_meals(profile, meals, food_by_id, slot):
    banned = set(profile.get("exclude_foods", []))
    allergies = set(profile.get("allergies", []))
    return [m for m in meals if m["slot"] == slot and not any(
        f in banned or (allergies & set(food_by_id[f]["allergens"])) for f, _ in m["ingredients"])]


def calculate_meal(template, food_by_id):
    totals = {key: 0.0 for key in ("kcal", "protein", "carb", "fat")}
    for food_id, grams in template["ingredients"]:
        for key in totals:
            totals[key] += food_by_id[food_id][key] * grams / 100.0
    return {"id": template["id"], "name": template["name"],
            "ingredients": [{"food_id": f, "grams": g} for f, g in template["ingredients"]],
            "nutrition": {k: round(v, 1) for k, v in totals.items()}}


def build_demo_plan(profile):
    """确定性本地版本，用来跑通 UI/单元测试；不代表模型生成，也不是个人健康处方。"""
    if not 18 <= profile["age"] <= 64:
        raise ValueError("演示项目限定 18–64 岁健康成年人。")
    available = [d for d in DAYS if d in profile["available_days"]]
    if len(available) < profile["sessions"]:
        raise ValueError("可训练日少于要求的每周训练次数。")
    ex, foods, meals = read_catalog()
    food_by_id = {f["id"]: f for f in foods}
    candidates = eligible_exercises(profile["equipment"], ex)
    if len(candidates) < 3:
        raise ValueError("器械条件下的候选训练动作不足。")
    meal_lists = {slot: safe_meals(profile, meals, food_by_id, slot)
                  for slot in ("早餐", "午餐", "晚餐")}
    if not all(meal_lists.values()):
        raise ValueError("有饮食限制无法满足：缺少合法餐食，需扩大食谱库。")
    # 在全部允许日期中选择相对分散的一组训练日；不偷偷扩大用户的可用日期。
    options = itertools.combinations(available, profile["sessions"])
    def spacing_score(ds):
        ix = [DAYS.index(x) for x in ds]
        gaps = [ix[i+1] - ix[i] for i in range(len(ix)-1)] + [7 + ix[0] - ix[-1]]
        return (min(gaps), -sum(g*g for g in gaps))
    selected = list(max(options, key=spacing_score))
    week = []
    for i, day in enumerate(DAYS):
        workout = None
        if day in selected:
            seq = [candidates[(i + j * 2) % len(candidates)] for j in range(3)]
            workout = {"exercise_ids": [e["id"] for e in seq], "minutes": 35,
                       "sets": 2 if profile["experience"] == "初学者" else 3}
        day_meals = {slot: calculate_meal(opts[i % len(opts)], food_by_id)
                     for slot, opts in meal_lists.items()}
        week.append({"day": day, "workout": workout, "meals": day_meals})
    return {"mode": "local_rule_demo", "profile": copy.deepcopy(profile), "week": week,
            "explanation": "示例规则组合方案；训练强度与饮食份量尚未经专业个体化评估。"}


def validate_plan(plan, profile=None):
    """返回硬约束违规详情；空列表表示当前覆盖的规则全部通过。"""
    profile = profile or plan["profile"]
    ex, foods, _ = read_catalog()
    ex_by_id = {e["id"]: e for e in ex}
    food_by_id = {f["id"]: f for f in foods}
    allowed_ex = {e["id"] for e in eligible_exercises(profile["equipment"], ex)}
    banned = set(profile.get("exclude_foods", []))
    allergies = set(profile.get("allergies", []))
    issues = []
    if [d["day"] for d in plan["week"]] != DAYS:
        issues.append("周计划必须覆盖周一至周日且顺序正确")
    count = 0
    for day in plan["week"]:
        workout = day.get("workout")
        if workout:
            count += 1
            if day["day"] not in profile["available_days"]:
                issues.append(day["day"] + " 不在可训练日内")
            for exercise_id in workout["exercise_ids"]:
                if exercise_id not in ex_by_id or exercise_id not in allowed_ex:
                    issues.append(day["day"] + " 器械不兼容：" + exercise_id)
        for slot in ("早餐", "午餐", "晚餐"):
            if slot not in day["meals"]:
                issues.append(day["day"] + " 缺少" + slot)
                continue
            meal = day["meals"][slot]
            for it in meal["ingredients"]:
                food_id = it["food_id"]
                if food_id not in food_by_id:
                    issues.append(day["day"] + " 未知食物：" + food_id)
                elif food_id in banned or allergies & set(food_by_id[food_id]["allergens"]):
                    issues.append(day["day"] + " 禁忌食物：" + food_id)
                if it["grams"] <= 0:
                    issues.append(day["day"] + " 食物重量不合法")
            for key in ("kcal", "protein", "carb", "fat"):
                if all(it["food_id"] in food_by_id for it in meal["ingredients"]):
                    reference = sum(food_by_id[it["food_id"]][key]*it["grams"]/100
                                    for it in meal["ingredients"])
                    if abs(reference - meal["nutrition"][key]) > 0.11:
                        issues.append(day["day"] + " " + slot + " 的" + key + "计算错误")
    if count != profile["sessions"]:
        issues.append(f"训练次数错误：期望{profile['sessions']}次，实际{count}次")
    return issues


def parse_edit_locally(text, foods):
    """仅支持两类预设规则，用于无 API Key 的演示。"""
    match = re.search(r"(周[一二三四五六日]).{0,12}?(?:改到|挪到|移到|调整到)(周[一二三四五六日])", text)
    banned = [f["id"] for f in foods if any(w in text for w in
              ("不吃"+f["name"], "不要"+f["name"], "不想吃"+f["name"]))]
    return {"source_day": match.group(1) if match else None,
            "target_day": match.group(2) if match else None,
            "exclude_foods": banned, "parsed_by": "local_regex_demo"}


def apply_edit(plan, intent):
    result = copy.deepcopy(plan)
    src, dst = intent.get("source_day"), intent.get("target_day")
    if src and dst:
        days = {d["day"]: d for d in result["week"]}
        if src not in days or dst not in days or not days[src]["workout"] or days[dst]["workout"]:
            raise ValueError("源训练日不存在或目标日期已有训练，不能自动调整。")
        if dst not in result["profile"]["available_days"]:
            raise ValueError("目标日期不在原本可训练日期内；请先修改可用时间。")
        days[dst]["workout"] = days[src]["workout"]
        days[src]["workout"] = None
    new_foods = set(intent.get("exclude_foods", []))
    if new_foods:
        result["profile"]["exclude_foods"] = list(set(result["profile"].get("exclude_foods", [])) | new_foods)
        _, foods, meals = read_catalog()
        food_by_id = {f["id"]: f for f in foods}
        for day in result["week"]:
            for slot, old in day["meals"].items():
                if any(it["food_id"] in new_foods for it in old["ingredients"]):
                    opts = safe_meals(result["profile"], meals, food_by_id, slot)
                    if not opts:
                        raise ValueError("调整饮食后缺少可用餐食。")
                    day["meals"][slot] = calculate_meal(opts[0], food_by_id)
    errors = validate_plan(result)
    if errors:
        raise ValueError("修改违反约束：" + "；".join(errors[:3]))
    result["mode"] = "local_rule_replan_demo"
    return result
