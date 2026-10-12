import base64, logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sam_service import generate_masks

app = FastAPI()
logger = logging.getLogger(__name__)

class SegmentRequest(BaseModel):
  image_base64: str 
  x: float = Field(ge=0, le=1, allow_inf_nan=False)
  y: float = Field(ge=0, le=1, allow_inf_nan=False)

@app.post("/segment")
def segment(request: SegmentRequest):
  try:
    image_bytes = base64.b64decode(request.image_base64, validate=True)
    polygons = generate_masks(image_bytes, request.x, request.y)
    return {"polygons": polygons}
  
  except Exception as error:
    logger.exception("SAM failed")
    raise HTTPException(status_code=500, detail="SAM inference failed") from error