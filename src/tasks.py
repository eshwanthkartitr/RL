from src.models import TaskConfig

TASKS = {
    # -------------------------------------------------------------
    # TIER 1: Standard Evaluation (Learning the Basics)
    # -------------------------------------------------------------
    "easy_spring": TaskConfig(
        task_id="easy_spring", max_steps=50, reservoir_capacity=200.0, 
        base_demand=10.0, price_volatility=10.0, weather_pattern=["Normal"]
    ),
    
    # -------------------------------------------------------------
    # TIER 2: Volatile Environmental Shifts (Learning Constraints)
    # -------------------------------------------------------------
    "summer_crisis": TaskConfig(
        task_id="summer_crisis", max_steps=100, reservoir_capacity=150.0, 
        base_demand=25.0, price_volatility=40.0, weather_pattern=["Normal", "Heatwave", "Heatwave", "Normal"]
    ),
    "hurricane_season": TaskConfig(
        task_id="hurricane_season", max_steps=150, reservoir_capacity=100.0, 
        base_demand=20.0, price_volatility=80.0, weather_pattern=["Normal", "Storm", "Normal", "Storm", "Storm"]
    ),

    # -------------------------------------------------------------
    # TIER 3: Asymmetrical Shock Scenarios (Testing Robustness)
    # -------------------------------------------------------------
    "black_swan_drought": TaskConfig(
        # Brutal: Demand stays critically high, reservoir doesn't hold much, and energy volatility is high. 
        # Tests the agent's ability to perfectly time maintenance cooldowns. If they miss one cleaning window, the city drys out.
        task_id="black_swan_drought", max_steps=200, reservoir_capacity=120.0, 
        base_demand=35.0, price_volatility=50.0, weather_pattern=["Heatwave", "Heatwave", "Heatwave", "Heatwave"]
    ),
    "grid_failure": TaskConfig(
        # The ultimate energy arbitrage test. Standard demand, but grid energy pricing fluctuates by massive magnitudes.
        # Producing water at the wrong step bankrupts the enterprise instantly.
        task_id="grid_failure", max_steps=200, reservoir_capacity=250.0, 
        base_demand=15.0, price_volatility=250.0, weather_pattern=["Normal", "Storm", "Storm", "Normal"]
    ),
    "marathon_endurance": TaskConfig(
        # 500 Steps: The agent must manage micro-degradation perfectly over a very long time horizon.
        # Short-term greedy strategies (running fouled, taking salinity hits) will eventually snowball into total failure.
        task_id="marathon_endurance", max_steps=500, reservoir_capacity=200.0, 
        base_demand=20.0, price_volatility=30.0, weather_pattern=["Normal", "Heatwave", "Storm", "Normal"]
    ),
}
