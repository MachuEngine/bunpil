from fastapi import FastAPI

app = FastAPI(title="분필 API", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}
