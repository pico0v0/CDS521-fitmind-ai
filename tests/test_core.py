from copy import deepcopy
from src.core import build_demo_plan, validate_plan, read_catalog, parse_edit_locally, apply_edit

BASE = {"age":23,"height_cm":170,"weight_kg":65,"goal":"减脂",
        "experience":"初学者","equipment":"哑铃", "sessions":3,
        "available_days":["周一","周三","周四","周五"],"allergies":[],"exclude_foods":[]}

def test_demo_output_passes_validator():
    plan = build_demo_plan(BASE)
    assert len(plan["week"]) == 7
    assert validate_plan(plan) == []

def test_food_allergy_filtered():
    p = deepcopy(BASE); p["allergies"] = ["鸡蛋", "乳制品"]
    plan = build_demo_plan(p)
    assert validate_plan(plan) == []
    assert all(x["food_id"] not in ("egg", "milk", "yogurt")
               for day in plan["week"] for meal in day["meals"].values()
               for x in meal["ingredients"])

def test_equipment_violation_detected():
    p = deepcopy(BASE); p["equipment"] = "徒手"
    plan = build_demo_plan(p)
    plan["week"][0]["workout"]["exercise_ids"][0] = "leg_press"
    assert any("器械不兼容" in x for x in validate_plan(plan))

def test_local_replan_preserves_other_days():
    plan = build_demo_plan(BASE)
    _, foods, _ = read_catalog()
    intent = parse_edit_locally("周三挪到周四，不要鸡蛋", foods)
    updated = apply_edit(plan, intent)
    assert updated["week"][2]["workout"] is None
    assert updated["week"][3]["workout"] is not None
    assert updated["week"][0]["workout"] == plan["week"][0]["workout"]
    assert validate_plan(updated) == []
