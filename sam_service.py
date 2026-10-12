import cv2, hashlib, os, torch
import numpy as np
from pathlib import Path
from segment_anything import SamPredictor, sam_model_registry
from threading import Lock
from dotenv import load_dotenv
ai_directory = Path(__file__).resolve().parent
load_dotenv(ai_directory / ".env", override=True)

sam_checkpoint = str(ai_directory / "models" / os.environ["SAM_CHECKPOINT"])
model_type = "vit_h"
device = os.environ.get("SAM_DEVICE", "mps")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()

predictor = SamPredictor(sam)
predictor_lock = Lock()

# store Image Embedding
cached_image_hash = None

# (x,y) is the Point to click to detect Segment outside (NORMALIZED coordinates)
def generate_masks(image_bytes, x, y):
  global cached_image_hash
  image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
  if image is None:
    raise ValueError("Unable to decode image")
  
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  h, w = image.shape[:2]
  # Convert from NORMALIZED to PIXELS (which SAM needs); shape [1,2]
  input_point = np.array([
    [min(x*w, w-1), min(y*h, h-1)],
  ], dtype=np.float32)
  input_label = np.array([1], dtype=np.int32)
  # Create Image Embedding
  image_hash = hashlib.sha256(image_bytes).hexdigest()
  # Protect from concurrent Requests
  with predictor_lock:
    with torch.inference_mode():
      if cached_image_hash != image_hash:
        cached_image_hash = None
        predictor.set_image(image)
        cached_image_hash = image_hash
      masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
      mask = masks[int(np.argmax(scores))] # take the mask with max score only
  
  contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  polygons = []
  for contour in contours:
    # to simplify Contour having less vertices
    perimeter = cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, 0.002*perimeter, True)
    if len(simplified) < 3:
      continue
    points = simplified.reshape(-1, 2) # [n_points,1,2] => [n_points,2]
    polygons.append({
      "points": [
        {"x": float(x)/w, "y": float(y)/h}
        for x, y in points
      ]
    })
  return polygons