import pytest
from src.schemas import UserProfile
from src.retrieval import get_eligible_exercises, get_safe_foods_and_meals, EQUIPMENT_RANK
from src.validator import validate_weekly_plan, auto_repair_plan
from src.planner_llm import generate_weekly_plan_llm

SAMPLE_PROFILE = {
    "age": 25,
    "height_cm": 175,
    "weight_kg": 70.0,
    "goal": "减脂",
    "experience": "初学者",
    "equipment": "徒手",
    "sessions": 3,
    "available_days": ["周一", "周三", "周五"],
    "allergies": ["乳制品", "鸡蛋"],
    "exclude_foods": ["beef"]
}

def test_equipment_retrieval_filtering():
    """测试器械分级候选动作检索"""
    bodyweight_ex = get_eligible_exercises("徒手")
    assert all(e["equipment"] == "徒手" for e in bodyweight_ex)
    assert len(bodyweight_ex) >= 10

    dumbbell_ex = get_eligible_exercises("哑铃")
    assert all(e["equipment"] in ("徒手", "哑铃") for e in dumbbell_ex)
    assert any(e["equipment"] == "哑铃" for e in dumbbell_ex)

    gym_ex = get_eligible_exercises("健身房")
    assert len(gym_ex) > len(dumbbell_ex)


def test_safe_foods_and_meals_filtering():
    """测试过敏原与主动排除项过滤"""
    safe_foods, safe_meals = get_safe_foods_and_meals(
        allergies=["乳制品", "鸡蛋"],
        exclude_foods=["beef"]
    )
    # 确保禁忌食材不在安全库中
    assert "milk" not in safe_foods
    assert "yogurt" not in safe_foods
    assert "egg" not in safe_foods
    assert "beef" not in safe_foods

    # 确保安全餐食中无任何违规食材
    for m in safe_meals:
        for fid, _ in m["ingredients"]:
            assert fid in safe_foods


def test_validator_detects_violations_and_repairs():
    """测试校验器发现违规并成功自动修复"""
    # 构造一个有严重违规的计划
    bad_plan = {
        "mode": "test_bad",
        "profile": SAMPLE_PROFILE,
        "week": [
            {
                "day": "周一",
                "workout": {"exercise_ids": ["barbell_bench_press"], "minutes": 40, "sets": 3},  # 健身房器械违规
                "meals": {
                    "早餐": {"id": "m1", "name": "坏早餐", "ingredients": [{"food_id": "milk", "grams": 200}], "nutrition": {"kcal": 100, "protein": 0, "carb": 0, "fat": 0}}, # 乳制品过敏
                    "午餐": {"id": "m2", "name": "正常午餐", "ingredients": [{"food_id": "chicken", "grams": 150}], "nutrition": {"kcal": 247.5, "protein": 46.5, "carb": 0, "fat": 5.4}},
                    "晚餐": {"id": "m3", "name": "正常晚餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}}
                }
            },
            {"day": "周二", "workout": None, "meals": {
                "早餐": {"id": "m1", "name": "早餐", "ingredients": [{"food_id": "oats", "grams": 50}], "nutrition": {"kcal": 194.5, "protein": 8.4, "carb": 33.2, "fat": 3.4}},
                "午餐": {"id": "m2", "name": "午餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}},
                "晚餐": {"id": "m3", "name": "晚餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}}
            }},
            {"day": "周三", "workout": None, "meals": {
                "早餐": {"id": "m1", "name": "早餐", "ingredients": [{"food_id": "oats", "grams": 50}], "nutrition": {"kcal": 194.5, "protein": 8.4, "carb": 33.2, "fat": 3.4}},
                "午餐": {"id": "m2", "name": "午餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}},
                "晚餐": {"id": "m3", "name": "晚餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}}
            }},
            {"day": "周四", "workout": None, "meals": {
                "早餐": {"id": "m1", "name": "早餐", "ingredients": [{"food_id": "oats", "grams": 50}], "nutrition": {"kcal": 194.5, "protein": 8.4, "carb": 33.2, "fat": 3.4}},
                "午餐": {"id": "m2", "name": "午餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}},
                "晚餐": {"id": "m3", "name": "晚餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}}
            }},
            {"day": "周五", "workout": None, "meals": {
                "早餐": {"id": "m1", "name": "早餐", "ingredients": [{"food_id": "oats", "grams": 50}], "nutrition": {"kcal": 194.5, "protein": 8.4, "carb": 33.2, "fat": 3.4}},
                "午餐": {"id": "m2", "name": "午餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}},
                "晚餐": {"id": "m3", "name": "晚餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}}
            }},
            {"day": "周六", "workout": None, "meals": {
                "早餐": {"id": "m1", "name": "早餐", "ingredients": [{"food_id": "oats", "grams": 50}], "nutrition": {"kcal": 194.5, "protein": 8.4, "carb": 33.2, "fat": 3.4}},
                "午餐": {"id": "m2", "name": "午餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}},
                "晚餐": {"id": "m3", "name": "晚餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}}
            }},
            {"day": "周日", "workout": None, "meals": {
                "早餐": {"id": "m1", "name": "早餐", "ingredients": [{"food_id": "oats", "grams": 50}], "nutrition": {"kcal": 194.5, "protein": 8.4, "carb": 33.2, "fat": 3.4}},
                "午餐": {"id": "m2", "name": "午餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}},
                "真实": {"id": "m3", "name": "晚餐", "ingredients": [{"food_id": "tofu", "grams": 150}], "nutrition": {"kcal": 114.0, "protein": 12.2, "carb": 2.8, "fat": 7.2}}
            }}
        ]
    }
    val = validate_weekly_plan(bad_plan, SAMPLE_PROFILE)
    assert not val.is_valid
    assert len(val.issues) > 0

    # 自动修复
    repaired = auto_repair_plan(bad_plan, SAMPLE_PROFILE)
    val2 = validate_weekly_plan(repaired, SAMPLE_PROFILE)
    assert val2.is_valid, f"修复后仍有违规: {val2.issues}"
