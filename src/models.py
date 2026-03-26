from pydantic import BaseModel, Field
from typing import Dict, Literal, List

class Observation(BaseModel):
    time_step: int
    reservoir_level: float = Field(description="Current fresh water stored (Megaliters)")
    water_salinity: float = Field(description="PPM of salt in the water. >500 is unsafe.")
    energy_price: float = Field(description="Current grid energy price ($/MWh)")
    membrane_fouling: float = Field(description="0.0 is clean, 1.0 is totally blocked")
    city_demand: float = Field(description="Water demand for this step (Megaliters)")
    weather_condition: Literal["Normal", "Heatwave", "Storm"] = Field(description="Current weather event affecting parameters")
    maintenance_cooldown: int = Field(description="Steps until a cleaning crew is available again")

class Action(BaseModel):
    production_rate: float = Field(description="Desired water output (ML/step), 0.0 to 50.0")
    run_cleaning: bool = Field(description="If True, halts production to chemically wash membranes (requires crew)")

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict

class TaskConfig(BaseModel):
    task_id: str
    max_steps: int
    reservoir_capacity: float
    base_demand: float
    price_volatility: float
    weather_pattern: List[str]
