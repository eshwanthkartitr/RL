import os
try:
    from fastapi import FastAPI
    # This might require openenv if used strictly. We mock a simple wrapper if not found.
    # Since the exact openenv module path can vary, here is the boilerplate template wrapper:
    
    app = FastAPI(title="ReleaseOps Arena Env")
    
    @app.post("/reset")
    def reset(params: dict):
        from releaseops_arena.tool_env import ReleaseOpsToolEnv
        env = ReleaseOpsToolEnv()
        obs = env.reset(**params)
        return {"observation": obs}
        
    @app.post("/step")
    def step(action: dict):
        # Mapped to tool methods directly in training
        pass

except ImportError:
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("releaseops_arena.server:app", host="0.0.0.0", port=8000)
