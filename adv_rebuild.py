import os

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

models_py = """
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
"""

env_py = """
import math
import random
from src.models import Observation, Action, StepResult, TaskConfig

class DesalEnv:
    def __init__(self):
        self.state = None
        self.config = None
        self.total_reward = 0.0

    def reset(self, config: TaskConfig) -> Observation:
        self.config = config
        self.total_reward = 0.0
        
        initial_weather = config.weather_pattern[0] if config.weather_pattern else "Normal"
        
        self.state = Observation(
            time_step=0,
            reservoir_level=config.reservoir_capacity * 0.5,
            water_salinity=300.0,  # 300 PPM is superb drinking water
            energy_price=50.0,
            membrane_fouling=0.0,
            city_demand=config.base_demand,
            weather_condition=initial_weather,
            maintenance_cooldown=0
        )
        return self.state

    def step(self, action: Action) -> StepResult:
        if self.state is None:
            raise ValueError("Must reset prior to step")
        
        reward = 0.0
        info = {}

        # 0. Apply Maintenance Cooldown
        if self.state.maintenance_cooldown > 0:
            self.state.maintenance_cooldown -= 1

        # 1. Processing Action: Cleaning or Pumping
        actual_production = 0.0
        energy_used = 0.0
        
        if action.run_cleaning:
            if self.state.maintenance_cooldown == 0:
                # Successful Clean
                self.state.membrane_fouling = max(0.0, self.state.membrane_fouling - 0.6)
                reward -= 1000.0  # High cost of washing chemicals & crew dispatch
                energy_used = 5.0 # Baseline power for flushing
                self.state.maintenance_cooldown = 5 # Takes 5 steps to organize the next crew
                info["action_taken"] = "cleaned"
            else:
                # Failed clean! The crew wasn't ready, plant stayed idle wasting a step.
                info["action_taken"] = "failed_clean_idle"
                reward -= 100.0 # Penalty for mismanagement
        else:
            actual_production = min(max(0.0, action.production_rate), 50.0)
            info["action_taken"] = f"produced_{actual_production:.1f}"
            
            # Physics Engine: Energy required scales exponentially as the membrane clogs
            energy_used = actual_production * (1.5 + (self.state.membrane_fouling * 8.0))
            
            # Sub-scale Fouling Physics: pushing water increments fouling parameter
            self.state.membrane_fouling = min(1.0, self.state.membrane_fouling + (actual_production * 0.002))
        
        # 2. Water Quality (Salinity) Tracking
        # Baseline is 300PPM. Pushing hard on a fouled membrane allows micro-tears leading to salt leak.
        self.state.water_salinity = 300.0 + (actual_production * self.state.membrane_fouling * 15.0)
        
        health_penalty = 0.0
        if self.state.water_salinity > 500.0:
            # Massive fine per unit of violation
            health_penalty = (self.state.water_salinity - 500.0) * 100.0 
            
        # 3. Economy & City Demands
        water_revenue = actual_production * 25.0
        self.state.reservoir_level = min(self.config.reservoir_capacity, self.state.reservoir_level + actual_production)
        
        # The city draws water
        shortfall = max(0.0, self.state.city_demand - self.state.reservoir_level)
        self.state.reservoir_level = max(0.0, self.state.reservoir_level - self.state.city_demand)
        
        # 4. Calculate Immediate Reward
        energy_cost = energy_used * (self.state.energy_price / 100.0)
        sla_penalty = shortfall * 1500.0 # Catastrophic penalty for empty lines (No water in pipes)
        
        step_reward = water_revenue - energy_cost - sla_penalty - health_penalty
        self.total_reward += step_reward
        
        info.update({
            "energy_cost": energy_cost, 
            "sla_penalty": sla_penalty,
            "health_penalty": health_penalty,
            "revenue": water_revenue
        })
        
        # 5. Advance time and Environment changes
        self.state.time_step += 1
        
        # Environmental Stochasticity: Weather Updates
        # Weather phases change every 10 steps
        weather_idx = (self.state.time_step // 10) % len(self.config.weather_pattern)
        self.state.weather_condition = self.config.weather_pattern[weather_idx]
        
        demand_multiplier = 1.0
        price_multiplier = 1.0
        
        if self.state.weather_condition == "Heatwave":
            demand_multiplier = 1.5 # Massive water usage
            price_multiplier = 1.8  # AC units are running, grid is stressed
        elif self.state.weather_condition == "Storm":
            demand_multiplier = 0.8
            price_multiplier = 0.4 + random.random() # Erratic energy prices
            
        # Modulate environment bounds
        self.state.energy_price = (50.0 * price_multiplier) + (math.sin(self.state.time_step / 4.0) * self.config.price_volatility) + random.uniform(-10, 10)
        self.state.energy_price = max(10.0, self.state.energy_price)
        
        self.state.city_demand = (self.config.base_demand * demand_multiplier) + (math.sin(self.state.time_step / 6.0) * (self.config.base_demand * 0.2)) + random.uniform(-2, 2)
        self.state.city_demand = max(5.0, self.state.city_demand)
        
        done = self.state.time_step >= self.config.max_steps
        
        return StepResult(observation=self.state, reward=step_reward, done=done, info=info)
"""

tasks_py = """
from src.models import TaskConfig

TASKS = {
    "easy_spring": TaskConfig(
        task_id="easy_spring", max_steps=50, reservoir_capacity=200.0, 
        base_demand=15.0, price_volatility=10.0, weather_pattern=["Normal"]
    ),
    "summer_crisis": TaskConfig(
        task_id="summer_crisis", max_steps=100, reservoir_capacity=150.0, 
        base_demand=25.0, price_volatility=40.0, weather_pattern=["Normal", "Heatwave", "Heatwave", "Normal"]
    ),
    "hurricane_season": TaskConfig(
        task_id="hurricane_season", max_steps=150, reservoir_capacity=100.0, 
        base_demand=20.0, price_volatility=80.0, weather_pattern=["Normal", "Storm", "Normal", "Storm", "Storm"]
    ),
}
"""

main_py = """
from fastapi import FastAPI, HTTPException
from src.models import Action, TaskConfig
from src.env import DesalEnv
from src.tasks import TASKS
import subprocess

app = FastAPI(title="Advanced Municipal Desalination Plant Env")
env = DesalEnv()

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Advanced DesalEnv is running", "features": ["weather", "salinity", "mechanics"]}

@app.post("/reset")
def reset_env(task_id: str = "easy_spring"):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    obs = env.reset(TASKS[task_id])
    return {"observation": obs.dict()}

@app.post("/step")
def step_env(action: Action):
    try:
        result = env.step(action)
        return result.dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    if env.state is None:
        raise HTTPException(status_code=400, detail="Environment not initialized")
    return {"observation": env.state.dict()}

@app.get("/tasks")
def list_tasks():
    return {"tasks": list(TASKS.keys()), "action_schema": Action.schema()}

@app.get("/grader")
def grader():
    if env.state is None:
        return {"score": 0.0}
    # Grade relative to typical maximum and minimum returns to generate a 0.0-1.0 range
    baseline_offset = env.config.max_steps * 1000.0 # Compensate for penalties
    scale_factor = env.config.max_steps * 1500.0 
    score = max(0.0, min(1.0, (env.total_reward + baseline_offset) / scale_factor))
    return {"score": score}

@app.post("/baseline")
def run_baseline():
    result = subprocess.run(["python", "src/baseline.py"], capture_output=True, text=True)
    return {"output": result.stdout}
"""

baseline_py = """
import requests

BASE_URL = "http://localhost:7860"

def evaluate_baseline(task_id):
    requests.post(f"{BASE_URL}/reset?task_id={task_id}")
    done = False
    
    while not done:
        state = requests.get(f"{BASE_URL}/state").json()["observation"]
        
        # Advanced Heuristic logic
        # If deeply fouled and crew is ready, we clean! 
        # Don't try to clean if cooldown is > 0
        needs_cleaning = state["membrane_fouling"] > 0.65 and state["maintenance_cooldown"] == 0
        
        if needs_cleaning:
            action = {"production_rate": 0.0, "run_cleaning": True}
        else:
            # Weather and Salinity check
            # If weather is Heatwave, demand is high, pump up.
            # But if Salinity is getting dangerous (>450), throttle!
            base_prod = state["city_demand"] * 1.2 # Attempt slight overproduce
            
            if state["water_salinity"] > 450.0:
                base_prod *= 0.5 # Drop production sharply to avoid fines
            
            # Energy heuristic: if expensive, only meet immediate demand.
            if state["energy_price"] > 70.0:
                base_prod = min(base_prod, state["city_demand"] * 0.9)
                
            action = {"production_rate": max(0.0, min(base_prod, 50.0)), "run_cleaning": False}
        
        step_res = requests.post(f"{BASE_URL}/step", json=action).json()
        done = step_res["done"]
        
    score = requests.get(f"{BASE_URL}/grader").json()["score"]
    print(f"Task: {task_id} | Final Score: {score:.3f}")

if __name__ == "__main__":
    for task in ["easy_spring", "summer_crisis", "hurricane_season"]:
        evaluate_baseline(task)
"""

readme_md = """
---
title: Desalination RL Protocol
emoji: 🌊
colorFrom: cyan
colorTo: blue
sdk: docker
pinned: false
---

# Advanced Municipal Desalination Plant (DesalEnv)

An incredibly unique, real-world RL environment that bridges continuous control, resource arbitrage, dynamic system physics, and environmental noise.

The agent operates an industrial reverse-osmosis water desalination plant providing drinking water to a municipality. It must balance massive trade-offs under high pressure. This goes **far** above basic control loops, presenting specific non-linear phenomena.

### Key Mechanics ⚙️
1. **Weather Shifts:** The environment continuously cycles through weather patterns (`Normal`, `Heatwave`, `Storm`) which violently alter both the Grid Energy Price and the sheer amount of water the city demands. 
2. **Maintenance Logistics:** Pushing water fouls the RO membranes, dragging up energy costs. You can trigger a `run_cleaning` action, however, crews are not instantly available! Doing so locks a `maintenance_cooldown`. Trying to clean while on cooldown results in idle time and fines.
3. **Biological Safety Limits:** Overworking a fouled membrane causes micro-tears resulting in salt leakage. The agent tracks `water_salinity`. Processing high water yields while fouled raises PPM levels. Tipping above 500PPM induces strict city health department fines. 

## 🧠 Environment Structure

### Observation Space

| Feature | Description | Type |
| :--- | :--- | :--- |
| `reservoir_level` | Fresh water stored (Megaliters). | `float` |
| `water_salinity` | PPM of salt in the water. >500 triggers penalties. | `float` |
| `energy_price` | Fluctuating grid energy price ($/MWh). | `float` |
| `membrane_fouling` | Hardware Degradation index (0.0=clean, 1.0=blocked). | `float` |
| `city_demand` | Fluctuating water consumption for the current step. | `float` |
| `weather_condition` | String literal tracking macro-events (`Heatwave`, etc.) | `string` |
| `maintenance_cooldown` | Steps until a cleaning crew is available again. | `int` |

### Action Space (Continuous & Discrete Hybrid)

| Feature | Description | Type |
| :--- | :--- | :--- |
| `production_rate` | Target water extraction flow rate (0.0 to 50.0). | `float` |
| `run_cleaning` | Set True to halt production and wash membranes (checks cooldown). | `bool` |

## Tasks

Provides 3 heavily distinct curriculums:
- `easy_spring`: Generous reservoir, standard weather patterns.
- `summer_crisis`: Frequent extreme Heatwaves driving massive demand + peak electricity pricing.
- `hurricane_season`: Wild grid-volatility, lower demand, but requires extreme energy arbitrage. 
"""

files = {
    "d:/KYC/src/models.py": models_py,
    "d:/KYC/src/env.py": env_py,
    "d:/KYC/src/tasks.py": tasks_py,
    "d:/KYC/src/main.py": main_py,
    "d:/KYC/src/baseline.py": baseline_py,
    "d:/KYC/README.md": readme_md
}

for path, content in files.items():
    write_file(path, content)
    print(f"Updated advanced mechanics in {path}")
