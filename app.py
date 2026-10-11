from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from router import predict_record, load_model

app=FastAPI(title="Kestrel Service Request Router", version="1.0.0",
 description="Local, explainable baseline for routing Kestrel service requests.")
class ServiceRequest(BaseModel):
    request_id: str | None = None
    channel: str = Field(default="chat", description="ivr, chat, whatsapp, or email")
    product_family: str = ""
    warranty_status: str = ""
    request_text: str = Field(min_length=1)
    source: str = "crm"
@app.get("/health")
def health():
    return {"status":"ok", "model_loaded":load_model() is not None}
@app.post("/predict")
def predict(req: ServiceRequest):
    result=predict_record(req.model_dump())
    if "error" in result: raise HTTPException(status_code=503 if result.get("code")=="MODEL_NOT_FOUND" else 422, detail=result)
    return result
