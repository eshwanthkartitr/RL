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
