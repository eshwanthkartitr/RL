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
