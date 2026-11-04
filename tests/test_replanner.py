from src.core import build_demo_plan
from src.schemas import EditIntent, WeeklyPlan
from src.replanner import apply_plan_edit
from src.validator import validate_weekly_plan

BASE_PROFILE = {
    "age": 25,
    "height_cm": 175,
    "weight_kg": 68.0,
    "goal": "增肌",
    "experience": "初学者",
    "equipment": "哑铃",
    "sessions": 3,
    "available_days": ["周一", "周三", "周五"],
    "allergies": [],
    "exclude_foods": []
}

def test_replanner_day_move_high_retention():
    demo_plan = build_demo_plan(BASE_PROFILE)
    # 将周三的训练移至周四（原本为休息日）
    intent = EditIntent(
        source_day="周三",
        target_day="周四",
        exclude_foods=[],
        explanation="周三有事改到周四"
    )
    updated, summary = apply_plan_edit(demo_plan, intent)
    
    # 验证周三已无训练，周四有训练
    days_map = {d.day: d for d in updated.week}
    assert days_map["周三"].workout is None
    assert days_map["周四"].workout is not None
    assert days_map["周一"].workout is not None

    # 验证其余元素保留率高
    assert summary["retention_rate"] >= 0.90
    assert summary["validation_passed"] is True
    
    # 验证新方案通过校验器
    assert validate_weekly_plan(updated).is_valid


def test_replanner_exclude_food_selective_replacement():
    demo_plan = build_demo_plan(BASE_PROFILE)
    # 增加禁忌食物鸡蛋
    intent = EditIntent(
        source_day=None,
        target_day=None,
        exclude_foods=["egg"],
        explanation="不吃鸡蛋"
    )
    updated, summary = apply_plan_edit(demo_plan, intent)
    
    # 验证没有任何餐食含有 egg
    for d in updated.week:
        for slot, m in [("早餐", d.meals.breakfast), ("午餐", d.meals.lunch), ("晚餐", d.meals.dinner)]:
            for ing in m.ingredients:
                assert ing.food_id != "egg"

    assert summary["validation_passed"] is True
    assert validate_weekly_plan(updated).is_valid
