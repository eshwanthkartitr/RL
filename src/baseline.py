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
    tasks_to_test = [
        "easy_spring", 
        "summer_crisis", 
        "hurricane_season", 
        "black_swan_drought", 
        "grid_failure", 
        "marathon_endurance"
    ]
    for task in tasks_to_test:
        evaluate_baseline(task)
