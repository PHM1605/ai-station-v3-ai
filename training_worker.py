import json, logging, math, os, pika, random, shutil, time, yaml
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from dotenv import load_dotenv
from functools import partial # library to create a Callback
from pathlib import Path
from pymongo import MongoClient, ReturnDocument
from ultralytics import YOLO
from urllib.request import Request, urlopen

AI_DIRECTORY = Path(__file__).resolve().parent
# SERVER_DIRECTORY = AI_DIRECTORY.parent / "server"
TRAINING_DIRECTORY = AI_DIRECTORY / "training_runs"
QUEUE_NAME = "ai_station.training"

load_dotenv(AI_DIRECTORY / ".env")
BACKEND_API_URL = os.environ["BACKEND_API_URL"].rstrip("/")
WORKER_API_TOKEN = os.environ["WORKER_API_TOKEN"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

mongo_client = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=5000)
database = mongo_client[os.getenv("DATABASE_NAME", "ai-station")]
jobs = database["training_jobs"]
snapshots = database["training_snapshots"]

def utc_now():
  return datetime.now(timezone.utc)

# polygon: {"annotation_id", "class_id", "points"} with "points" is [{x,y}]
def polygon_to_label(polygon):
  if polygon["class_id"] != 0:
    raise ValueError("Only class_id=0, fish, is supported")
  
  points = polygon["points"]
  if len(points) < 3:
    raise ValueError("A polygon requires at least 3 points")
  
  coordinates = [] # ["0.2", "0.4", "0.5", "0.1" ...]
  for point in points:
    x = float(point["x"])
    y = float(point["y"])
    if (not math.isfinite(x) or not math.isfinite(y) or not 0<=x<=1 or not 0<=y<=1):
      raise ValueError("Polygon points must be normalized")
    coordinates.extend([f"{x:.3f}", f"{y:.3f}"]) # ["0.2", "0.4"]
  
  # 1 line per polygon: "0 0.2 0.4 0.5 0.1 0.7 0.5...."
  return "0 " + " ".join(coordinates)
  
def split_samples(samples):
  random_generator = random.Random(42)
  video_ids = sorted({sample["video_id"] for sample in samples})
  if len(video_ids) >= 2:
    # Keep frames from same video in the same split
    random_generator.shuffle(video_ids)
    validation_count = max(1, round(len(video_ids)*0.2))
    validation_videos = set(video_ids[:validation_count])
    train_samples = [
      sample for sample in samples
      if sample["video_id"] not in validation_videos
    ]
    validation_samples = [
      sample for sample in samples
      if sample["video_id"] in validation_videos
    ]
  else:
    logging.warning("Only 1 video: using a frame split. Similar frames can make a validation score optimistic.")
    shuffled = list(samples)
    random_generator.shuffle(shuffled)
    
    validation_count = max(1, round(len(shuffled) * 0.2))
    validation_samples = shuffled[:validation_count]
    train_samples = shuffled[validation_count:]
  return train_samples, validation_samples

# Download from Go Server to GPU Server
def download_snapshot(job_id, frame_id, destination):
  url = f"{BACKEND_API_URL}/internal/training-jobs/{job_id}/frames/{frame_id}/image"
  request = Request(url, headers={"X-Worker-Token": WORKER_API_TOKEN})
  # Get Snapshot Image from Go Server
  with urlopen(request, timeout=120) as response:
    with destination.open("wb") as image_file:
      shutil.copyfileobj(response, image_file)

# Upload trained model from GPU Server to Go Server
def upload_model(job_id, best_model):
  url = f"{BACKEND_API_URL}/internal/training-jobs/{job_id}/model" # Go URL
  request = Request(
    url,
    data=best_model.read_bytes(),
    headers={
      "X-Worker-Token": WORKER_API_TOKEN,
      "Content-Type": "application/octet-stream",
    },
    method="PUT"
  )
  with urlopen(request, timeout=300) as response:
    return json.load(response)["model_path"] # Go Server path of Model (stored in MongoDB)

def prepare_dataset(job):
  # struct of "job": in training_controller
  samples = list(snapshots.find({
    "job_id": job["job_id"],
    "owner_id": job["owner_id"],
    "annotations.0": {"$exists": True}, # snapshot with annotations list length > 0
  }))
  if len(samples) < 2:
    raise ValueError("At least 2 annotated frames are required")
  
  job_directory = TRAINING_DIRECTORY / job["job_id"] # AI/training_runs/<job_id>/
  job_directory.mkdir(parents=True, exist_ok=False) # False means "overwrite if this Folder exists"
  
  dataset_directory = job_directory / "dataset" # AI/training_runs/<job_id>/dataset
  train_samples, validation_samples = split_samples(samples)
  # uploads_directory = (SERVER_DIRECTORY / "uploads").resolve() # server/uploads
  
  for split, selected_samples in [("train", train_samples),("val", validation_samples)]:
    images_directory = dataset_directory / "images" / split # AI/training_runs/<job_id>/dataset/images/train (or /val)
    labels_directory = dataset_directory / "labels" / split # AI/training_runs/<job_id>/dataset/labels/train (or /val)
    
    images_directory.mkdir(parents=True)
    labels_directory.mkdir(parents=True)
    
    # sample: 1 frame and its polygons
    # prepare "images/train (or val)/*.jpg" and "labels/train (or val)/*.txt"
    for sample in selected_samples:
      # For LOCAL AI server
      # source_image = (SERVER_DIRECTORY / sample["snapshot_path"]).resolve() # server/uploads/training/<jobId>/images/xxx.jpg
      # if not source_image.is_relative_to(uploads_directory):
      #   raise ValueError("Snapshot image is outside uploads")
      # if not source_image.is_file():
      #   raise FileNotFoundError(f"Missing snapshot for frame {sample['frame_id']}")
      # frame_id = sample["frame_id"]
      # image_name = frame_id + source_image.suffix.lower()
      # # copy from Snapshot directory to "train"/"val" directory
      # shutil.copy2(source_image, images_directory / image_name)
      
      # For REMOTE GPU AI Server
      frame_id = sample["frame_id"]
      image_name = f"{frame_id}.jpg"
      download_snapshot(
        job["job_id"],
        frame_id,
        images_directory / image_name
      )
      
      # create lines of polygons => write to a <frame_id>.txt file
      label_lines = [
        polygon_to_label(polygon)
        for polygon in sample["annotations"]
      ]
      # write to AI/training_runs/<job_id>/dataset/labels/train/<frame_id>.txt (or /val)
      (labels_directory / f"{frame_id}.txt").write_text(
        "\n".join(label_lines) + "\n",
        encoding="utf-8"
      )
    
  # prepare AI/training_runs/<job_id>/dataset/data.yaml
  dataset_yaml = dataset_directory / "data.yaml"
  dataset_yaml.write_text(
    yaml.safe_dump({
      "path": str(dataset_directory.resolve()), # AI/training_runs/<job_id>/dataset
      "train": "images/train",
      "val": "images/val",
      "names": {0: "fish"}
    }),
    encoding="utf-8"
  )
  # job_directory: AI/training_runs/<job_id>
  # dataset_yaml: AI/training_runs/<job_id>/dataset/data.yaml
  return job_directory, dataset_yaml
    
def train_job(job_id):
  # Claim the Job so that duplicated Messages don't start it twice
  # see "training_controller" to understand
  job = jobs.find_one_and_update(
    {
      "job_id": job_id,
      "snapshot_ready": True, 
      "status": {"$in": ["queued", "dispatching"]}
    }, 
    {
      "$set": {
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "error": ""
      },
    },
    return_document=ReturnDocument.AFTER, # from pymongo  
  )
  
  if job is None:
    existing_job = jobs.find_one({"job_id": job_id}, {"status": 1})
    if existing_job is None:
      logging.warning("Skipping deleted job %s", job_id)
      return 
    if existing_job["status"] in ["completed", "failed", "cancelled", "deleting"]:
      logging.info("Skipping finished job %s", job_id)
      return # => ACK in "finish_delivery"
    # if job not exist => error (to be catched in "finish_delivery") => NACK
    raise RuntimeError(f"Job {job_id} cannot be claimed; status={existing_job['status']}")
    
  # Update MongoDB from "cancel_requested" to "cancelled" Job
  # will be call once BEFORE and once AFTER "model.train()" to check status
  def finish_cancelled():
    jobs.update_one(
      {"job_id": job_id, "status": "cancel_requested"},
      {"$set": {
        "status": "cancelled",
        "completed_at": utc_now(),
        "updated_at": utc_now(),
      }}
    )  
  
  try: 
    if job["base_model"] != "yolo":
      raise ValueError("Unsupported base model")
    if job["gpu_type"] != "any":
      raise ValueError("GPU type routing is not implemented")
    if job["epochs"] < 1 or job["batch_size"] < 1:
      raise ValueError("Invalid training settings")
    
    # job_directory: AI/training_runs/<job_id>
    # dataset_yaml: AI/training_runs/<job_id>/dataset/data.yaml
    job_directory, dataset_yaml = prepare_dataset(job)
    model = YOLO("yolo26n-seg.pt")
    
    # call after each epoch => update MongoDB for status
    def update_progress(trainer):
      jobs.update_one(
        {"job_id": job_id, "status": "running"},
        {"$set": {
          "current_epoch": trainer.epoch + 1,
          "updated_at": utc_now(),
        }}
      )
    model.add_callback("on_train_epoch_end", update_progress)
    
          
    if jobs.find_one({"job_id": job_id}, {"status": 1})["status"] == "cancel_requested":
      finish_cancelled()
      return 
        
    # for Canceling Job catch EVERY 2 SECONDS
    last_cancel_check = 0.0
    def check_cancellation(trainer):
      nonlocal last_cancel_check
      now = time.monotonic()
      # don't check cancellation if it's sooner than 2s gap
      if now - last_cancel_check < 2:
        return 
      last_cancel_check = now
      
      current_job = jobs.find_one(
        {"job_id": job_id},
        {"status": 1} # means "get the "status" field out" (together with "_id")
      ) 
      # cancel Job if it's status is "cancel_requested" in DB
      if current_job and current_job["status"] == "cancel_requested":
        trainer.stop = True
      
    model.add_callback("on_train_batch_end", check_cancellation)
    
    model.train(
      data=str(dataset_yaml),
      epochs=job["epochs"],
      batch=job["batch_size"],
      imgsz=640,
      device=os.getenv("TRAIN_DEVICE", "cpu"),
      workers=0,
      project=str(job_directory),
      name="model",
      exist_ok=False, # overwrite "best.pt"
    )
    
    # Check cancellation of Job after train
    current = jobs.find_one({"job_id": job_id}, {"status": 1})
    if current and current["status"] == "cancel_requested":
      finish_cancelled()
      return 
    
    best_model = Path(model.trainer.best).resolve()
    if not best_model.is_file():
      raise RuntimeError("Training did not produce best.pt")
    
    # model_path = str(best_model) # for local AI server
    model_path = upload_model(job_id, best_model) # for separate GPU AI Server
    # update job status in DB
    result = jobs.update_one(
      {"job_id": job_id, "status": "running"},
      {"$set": {
        "status": "completed",
        "model_path": model_path,
        "completed_at": utc_now(),
        "updated_at": utc_now(),
      }}
    )
    # when "running" Job receives "cancel_requested" status from Frontend in the meantime
    if result.matched_count == 0:
      finish_cancelled()
      return
    logging.info("Completed job %s: %s", job_id, best_model)
  
  except Exception as error:
    # Error happens but Job has been cancelled => mark as "cancelled"
    current_status = jobs.find_one({"job_id": job_id}, {"status": 1})
    if current_status and current_status["status"] == "cancel_requested":
      finish_cancelled()
      return 
    
    logging.exception("Training failed for job %s", job_id)
    # update "failed" status for job in MongoDB
    jobs.update_one(
      {"job_id": job_id, "status": "running"},
      {
        "$set": {
          "status": "failed",
          "error": str(error),
          "completed_at": utc_now(),
          "updated_at": utc_now(),
        }
      }
    )
  finally:
    try:
      final_job = jobs.find_one({"job_id": job_id}, {"status": 1})
      if final_job and final_job["status"] in ("completed", "failed", "cancelled"):
        shutil.rmtree(TRAINING_DIRECTORY / job_id, ignore_errors=True)
    except Exception:
      logging.exception("Could not clean remote files for job %s", job_id)
    
# future is "Task Outcome"
def finish_delivery(channel, delivery_tag, future):
  if not channel.is_open:
    return 
  
  try:
    future.result()
  except Exception:
    logging.exception("Worker could not finish processing the message")
    # Returns that to the Queue
    channel.basic_nack(delivery_tag=delivery_tag, requeue=True)
  # this will run if "except" doesn't happen
  else:
    # Training completed OR its failure is recorded in MongoDB
    channel.basic_ack(delivery_tag=delivery_tag)

def main():
  mongo_client.admin.command("ping")
  
  # Setup RabbitMQ
  parameters = pika.URLParameters(os.environ["RABBITMQ_URL"])
  parameters.heartbeat = 60
  parameters.blocked_connection_timeout = 30
  
  connection = pika.BlockingConnection(parameters)
  channel = connection.channel()
  channel.queue_declare(queue=QUEUE_NAME, durable=True)
  
  # Send this Worker 1 job at a time, don't send another until it ACK
  channel.basic_qos(prefetch_count=1)
  # Create 1 background Python Thread for training => main Thread is responsive to RabbitMQ
  executor = ThreadPoolExecutor(max_workers=1) # later "future = executor.submit(train_job, job_id)" creates a new Thread
  
  def receive_job(channel, method, properties, body):
    try:
      message = json.loads(body)
      job_id = message["job_id"]
      if not isinstance(job_id, str) or len(job_id) != 24 or any(character not in "0123456789abcdef" for character in job_id):
        raise ValueError("Invalid job ID")
    except (ValueError, KeyError, TypeError):
      logging.exception("Invalid training message")
      channel.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
      return 
    
    logging.info("Received training job %s", job_id)
    # Create a Thread to execute training
    future = executor.submit(train_job, job_id) # "train_job" is a Callback
    
    # Callback when Training Thread finishes
    # completed_future: holds the Task's Outcome
    def training_finished(completed_future):
      # create a Callback from a Function and it's Arguments for later usage
      callback = partial(finish_delivery, channel, method.delivery_tag, completed_future)
      # schedule it in the Main Thread
      try:
        connection.add_callback_threadsafe(callback)
      except pika.exceptions.ConnectionWrongStateError:
        logging.warning("Connection closed before job acknowledgement")
    
    future.add_done_callback(training_finished)
  
  channel.basic_consume(queue=QUEUE_NAME, on_message_callback=receive_job, auto_ack=False)
  logging.info("Waiting for training jobs")
  
  try: 
    channel.start_consuming()
  except KeyboardInterrupt:
    channel.stop_consuming()
  finally:
    # Waiting for Thread to finish his Job
    executor.shutdown(wait=True)
    if connection.is_open:
      connection.close()
    mongo_client.close()
  
if __name__ == "__main__":
  main()