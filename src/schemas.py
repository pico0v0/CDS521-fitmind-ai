"""FitMind AI 数据结构契约定义 (基于 Pydantic V2)
确保系统各模块（检索、LLM、校验、重排、评测与前端）数据格式高度统一。
"""
from __future__ import annotations
from typing import Literal, List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator

EquipmentType = Literal["徒手", "哑铃", "健身房"]
GoalType = Literal["减脂", "增肌", "塑形", "保持健康"]
ExperienceLevel = Literal["初学者", "有经验"]
WeekDay = Literal["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
MealSlot = Literal["早餐", "午餐", "晚餐", "加餐"]
MuscleGroup = Literal["上肢", "下肢", "核心", "全身", "胸部", "背部", "肩部", "手臂"]
AllergenType = Literal["乳制品", "鸡蛋", "小麦", "大豆", "鱼类", "坚果"]


class UserProfile(BaseModel):
    """用户健康体征与个性化需求画像"""
    model_config = ConfigDict(extra="ignore")

    id: Optional[str] = Field(default=None, description="用户画像唯一ID，供测试与实验使用")
    age: int = Field(ge=18, le=64, description="年龄 (18-64 岁)")
    height_cm: int = Field(ge=130, le=220, description="身高 (cm)")
    weight_kg: float = Field(ge=30.0, le=250.0, description="体重 (kg)")
    goal: GoalType = Field(default="减脂", description="健康/塑形目标")
    experience: ExperienceLevel = Field(default="初学者", description="运动经验")
    equipment: EquipmentType = Field(default="哑铃", description="可用器械环境")
    available_days: List[WeekDay] = Field(min_length=1, max_length=7, description="每周允许训练的日期集合")
    sessions: int = Field(ge=1, le=7, description="每周规划的训练次数")
    allergies: List[str] = Field(default_factory=list, description="过敏原列表")
    exclude_foods: List[str] = Field(default_factory=list, description="用户主动避开的食物 ID 列表")
    notes: Optional[str] = Field(default=None, description="特殊习惯或补充要求")


class Exercise(BaseModel):
    """标准训练动作元数据"""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="动作唯一英文字符串 ID")
    name: str = Field(description="动作中文名")
    equipment: EquipmentType = Field(description="所需最低器械要求")
    muscle: str = Field(description="目标主要肌群")
    secondary_muscles: List[str] = Field(default_factory=list, description="辅助肌群")
    difficulty: ExperienceLevel = Field(default="初学者", description="推荐经验水平")
    category: str = Field(default="力量", description="动作类别：力量 / 有氧 / 核心 / 拉伸")
    description: str = Field(default="", description="动作要领说明")


class FoodItem(BaseModel):
    """食材标准宏量营养信息 (每 100g 可食部口径)"""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="食材唯一 ID")
    name: str = Field(description="食材中文名称")
    category: str = Field(default="其他", description="食品类别：主食/谷物、优质蛋白、蔬菜、水果、坚果/油脂")
    kcal: float = Field(ge=0, description="热量 (kcal / 100g)")
    protein: float = Field(ge=0, description="蛋白质 (g / 100g)")
    carb: float = Field(ge=0, description="碳水化合物 (g / 100g)")
    fat: float = Field(ge=0, description="脂肪 (g / 100g)")
    allergens: List[str] = Field(default_factory=list, description="所含过敏原列表")


class IngredientPortion(BaseModel):
    """餐食中的食材份量"""
    model_config = ConfigDict(extra="ignore")

    food_id: str = Field(description="食材 ID")
    grams: float = Field(gt=0, description="克重 (g)")


class NutritionSummary(BaseModel):
    """宏量营养素统计汇总"""
    model_config = ConfigDict(extra="ignore")

    kcal: float = Field(ge=0, description="总能量 (kcal)")
    protein: float = Field(ge=0, description="蛋白质 (g)")
    carb: float = Field(ge=0, description="碳水化合物 (g)")
    fat: float = Field(ge=0, description="脂肪 (g)")


class MealTemplate(BaseModel):
    """餐食组合模板数据"""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="餐食模板 ID")
    name: str = Field(description="餐食组合名称")
    slot: MealSlot = Field(description="适用时段 (早餐/午餐/晚餐/加餐)")
    ingredients: List[List[object]] = Field(description="食材列表，格式为 [[food_id, grams], ...]")


class MealDetail(BaseModel):
    """实际计算生成的单餐明细"""
    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="餐食模板 ID 或组合 ID")
    name: str = Field(description="餐食名称")
    ingredients: List[IngredientPortion] = Field(description="食材及重量列表")
    nutrition: NutritionSummary = Field(description="确定性计算的营养总和")


class DayMeals(BaseModel):
    """单日三餐结构（明确字段以严格兼容 OpenAI Structured Outputs）"""
    model_config = ConfigDict(extra="ignore")

    breakfast: MealDetail = Field(description="早餐安排")
    lunch: MealDetail = Field(description="午餐安排")
    dinner: MealDetail = Field(description="晚餐安排")

    @model_validator(mode="before")
    @classmethod
    def normalize_slot_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            mapped = {}
            if "breakfast" in data:
                mapped["breakfast"] = data["breakfast"]
            elif "早餐" in data:
                mapped["breakfast"] = data["早餐"]

            if "lunch" in data:
                mapped["lunch"] = data["lunch"]
            elif "午餐" in data:
                mapped["lunch"] = data["午餐"]

            if "dinner" in data:
                mapped["dinner"] = data["dinner"]
            elif "晚餐" in data:
                mapped["dinner"] = data["晚餐"]
            return mapped
        return data

    @classmethod
    def from_any_dict(cls, data: Any) -> DayMeals:
        if isinstance(data, DayMeals):
            return data
        if isinstance(data, dict):
            b = data.get("breakfast") or data.get("早餐")
            l = data.get("lunch") or data.get("午餐")
            d = data.get("dinner") or data.get("晚餐")
            return cls(
                breakfast=MealDetail.model_validate(b),
                lunch=MealDetail.model_validate(l),
                dinner=MealDetail.model_validate(d)
            )
        raise ValueError(f"无法将 {type(data)} 转换为 DayMeals")

    def __getitem__(self, slot: str) -> MealDetail:
        mapping = {
            "早餐": self.breakfast, "午餐": self.lunch, "晚餐": self.dinner,
            "breakfast": self.breakfast, "lunch": self.lunch, "dinner": self.dinner
        }
        return mapping[slot]

    def items(self):
        return [("早餐", self.breakfast), ("午餐", self.lunch), ("晚餐", self.dinner)]

    def get(self, slot: str, default=None):
        try:
            return self[slot]
        except KeyError:
            return default

    def to_slot_dict(self) -> Dict[str, Any]:
        return {
            "早餐": self.breakfast.model_dump(),
            "午餐": self.lunch.model_dump(),
            "晚餐": self.dinner.model_dump()
        }


class DailyWorkout(BaseModel):
    """单日训练规划"""
    model_config = ConfigDict(extra="ignore")

    exercise_ids: List[str] = Field(description="训练动作 ID 列表")
    minutes: int = Field(ge=10, le=180, description="计划训练时长 (分钟)")
    sets: int = Field(ge=1, le=10, description="每动作计划组数")
    notes: Optional[str] = Field(default=None, description="训练提示或热身建议")


class DailyPlan(BaseModel):
    """单日完整计划 (训练 + 餐饮)"""
    model_config = ConfigDict(extra="ignore")

    day: WeekDay = Field(description="星期 (周一至周日)")
    workout: Optional[DailyWorkout] = Field(default=None, description="当日训练安排，为 None 表示休息/恢复日")
    meals: DayMeals = Field(description="当日餐饮安排，包含早餐(breakfast)、午餐(lunch)、晚餐(dinner)")


class WeeklyPlan(BaseModel):
    """7 天完整周期计划"""
    model_config = ConfigDict(extra="ignore")

    mode: str = Field(default="fitmind_llm", description="生成机制标识 (local_rule_demo / fitmind_llm / baseline_direct_llm)")
    profile: UserProfile = Field(description="对应的用户画像基准")
    week: List[DailyPlan] = Field(min_length=7, max_length=7, description="连续7天(周一到周日)日程安排")
    explanation: str = Field(default="", description="计划制定科学性解释与执行建议")


class EditIntent(BaseModel):
    """自然语言修改意图结构化抽取结果"""
    model_config = ConfigDict(extra="ignore")

    source_day: Optional[WeekDay] = Field(default=None, description="训练移动的源日期")
    target_day: Optional[WeekDay] = Field(default=None, description="训练移动的目标日期")
    exclude_foods: List[str] = Field(default_factory=list, description="新增需要排除的食物 ID")
    replace_meal: Optional[Dict[str, str]] = Field(default=None, description="单餐替换指示，如 {'day': '周五', 'slot': '晚餐', 'target_meal_id': '...'}")
    explanation: str = Field(default="", description="调整意图解析说明")
    parsed_by: str = Field(default="llm", description="解析器类型 (llm / local_regex)")


class ValidationResult(BaseModel):
    """约束校验结果"""
    model_config = ConfigDict(extra="ignore")

    is_valid: bool = Field(description="是否完全符合所有硬约束")
    issues: List[str] = Field(default_factory=list, description="发现的违规问题清单")

