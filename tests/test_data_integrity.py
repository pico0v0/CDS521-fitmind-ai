import json
from pathlib import Path
from src.schemas import Exercise, FoodItem, MealTemplate, AllergenType

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def test_exercises_integrity():
    content = json.loads((DATA_DIR / "exercises.json").read_text(encoding="utf-8"))
    assert len(content) >= 35, f"训练动作数量应在 35 个以上，当前: {len(content)}"
    
    ids = set()
    for item in content:
        ex = Exercise.model_validate(item)
        assert ex.id not in ids, f"动作 ID 重复: {ex.id}"
        ids.add(ex.id)
        assert ex.equipment in ("徒手", "哑铃", "健身房")
        assert len(ex.name) > 0


def test_foods_integrity():
    content = json.loads((DATA_DIR / "foods.json").read_text(encoding="utf-8"))
    assert len(content) >= 35, f"食品食材数量应在 35 种以上，当前: {len(content)}"
    
    allowed_allergens = {"乳制品", "鸡蛋", "小麦", "大豆", "鱼类", "坚果"}
    ids = set()
    for item in content:
        food = FoodItem.model_validate(item)
        assert food.id not in ids, f"食品 ID 重复: {food.id}"
        ids.add(food.id)
        for al in food.allergens:
            assert al in allowed_allergens, f"未知过敏原类别: {al} in {food.id}"


def test_meals_integrity():
    foods = {f["id"]: f for f in json.loads((DATA_DIR / "foods.json").read_text(encoding="utf-8"))}
    content = json.loads((DATA_DIR / "meals.json").read_text(encoding="utf-8"))
    assert len(content) >= 20, f"餐食模板数量应在 20 份以上，当前: {len(content)}"
    
    meal_ids = set()
    slots = {"早餐": 0, "午餐": 0, "晚餐": 0}
    for item in content:
        meal = MealTemplate.model_validate(item)
        assert meal.id not in meal_ids, f"餐食 ID 重复: {meal.id}"
        meal_ids.add(meal.id)
        if meal.slot in slots:
            slots[meal.slot] += 1
        for food_id, grams in meal.ingredients:
            assert food_id in foods, f"餐食 {meal.id} 引用了不存在的食材 ID: {food_id}"
            assert grams > 0, f"餐食 {meal.id} 中的 {food_id} 克重必须大于0"

    assert slots["早餐"] >= 4, "早餐模板不足"
    assert slots["午餐"] >= 6, "午餐模板不足"
    assert slots["晚餐"] >= 6, "晚餐模板不足"

