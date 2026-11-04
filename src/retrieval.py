"""FitMind AI 候选检索与数据预处理模块 (Retrieval Layer)
负责在调用大模型前，基于用户画像做确定性硬约束过滤，
提供合法的动作候选池和绝对安全的餐食候选池，防止幻觉与越界。
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from src.schemas import (
    UserProfile, Exercise, FoodItem, MealTemplate,
    MealDetail, IngredientPortion, NutritionSummary
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EQUIPMENT_RANK = {"徒手": 0, "哑铃": 1, "健身房": 2}


def load_catalog() -> Tuple[List[dict], List[dict], List[dict]]:
    """加载原始基础数据"""
    ex = json.loads((DATA_DIR / "exercises.json").read_text(encoding="utf-8"))
    foods = json.loads((DATA_DIR / "foods.json").read_text(encoding="utf-8"))
    meals = json.loads((DATA_DIR / "meals.json").read_text(encoding="utf-8"))
    return ex, foods, meals


def get_eligible_exercises(equipment: str, experience: str = "初学者") -> List[dict]:
    """根据用户可用器械等级过滤动作"""
    ex, _, _ = load_catalog()
    user_rank = EQUIPMENT_RANK.get(equipment, 0)
    eligible = [e for e in ex if EQUIPMENT_RANK.get(e["equipment"], 0) <= user_rank]
    return eligible


def get_safe_foods_and_meals(
    allergies: List[str],
    exclude_foods: List[str]
) -> Tuple[Dict[str, dict], List[dict]]:
    """
    根据用户过敏原与主动排除项，过滤出安全食材字典与合规餐食模板列表
    """
    _, foods, meals = load_catalog()
    food_by_id = {f["id"]: f for f in foods}
    
    banned_ids: Set[str] = set(exclude_foods)
    user_allergies: Set[str] = set(allergies)
    
    # 查找所有因过敏原需要禁用的食材
    allergy_banned_ids = {
        fid for fid, f in food_by_id.items()
        if user_allergies & set(f.get("allergens", []))
    }
    all_banned = banned_ids | allergy_banned_ids
    
    safe_food_by_id = {fid: f for fid, f in food_by_id.items() if fid not in all_banned}
    
    safe_meals = []
    for m in meals:
        # 该餐食所用的全部食材必须都在 safe_food_by_id 中
        if all(ing[0] in safe_food_by_id for ing in m["ingredients"]):
            safe_meals.append(m)
            
    return safe_food_by_id, safe_meals


def compute_meal_detail(template: dict, food_by_id: Dict[str, dict]) -> MealDetail:
    """对单份餐食模板进行精确确定性营养计算"""
    totals = {"kcal": 0.0, "protein": 0.0, "carb": 0.0, "fat": 0.0}
    portions = []
    
    for food_id, grams in template["ingredients"]:
        food = food_by_id[food_id]
        portions.append(IngredientPortion(food_id=food_id, grams=float(grams)))
        factor = float(grams) / 100.0
        totals["kcal"] += food["kcal"] * factor
        totals["protein"] += food["protein"] * factor
        totals["carb"] += food["carb"] * factor
        totals["fat"] += food["fat"] * factor
        
    return MealDetail(
        id=template["id"],
        name=template["name"],
        ingredients=portions,
        nutrition=NutritionSummary(
            kcal=round(totals["kcal"], 1),
            protein=round(totals["protein"], 1),
            carb=round(totals["carb"], 1),
            fat=round(totals["fat"], 1)
        )
    )

