from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import os
from datetime import datetime
from fashion_clip.fashion_clip import FashionCLIP
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import base64
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http import models
import shutil

# Initialize FastAPI app
app = FastAPI(
    title="Human Detection API",
    description="API to detect humans in images using YOLOv8 Nano",
    version="1.0.0"
)

# Load YOLOv8 nano model (smallest weight version)
model = YOLO('yolov8n.pt')
fclip = FashionCLIP('fashion-clip')

client = QdrantClient(host="localhost", port=6333)

IMAGE_COLLECTION = "cctv_images"
# Define class names we're interested in (person class is typically 0 in COCO dataset)
HUMAN_CLASS_ID = 0
def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def compute_histogram(image):
    """Compute color histogram for an image"""
    image = np.array(image)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def image_to_base64(image_path):
    """
    Convert image to base64 encoding
    """
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")  # You can change the format to PNG if needed
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.post("/detect-humans-video/")
async def detect_humans_video(file: UploadFile = File(...)):
    """
    Endpoint to detect humans in a video file at 1 FPS, removing duplicates across frames
    Saves cropped images and processed video with timestamp, ensures each person is detected only once
    """
    try:
        
        # Create output directories
        output_dir = "cropped_humans_video"
        video_dir = "processed_videos"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        # Read video file
        contents = await file.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"temp_video_{timestamp}.mp4"
        with open(temp_filename, 'wb') as temp_file:
            temp_file.write(contents)

        # Open video
        cap = cv2.VideoCapture(temp_filename)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps)  # Process 1 frame per second
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer
        output_video_path = f"{video_dir}/processed_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Process results
        all_detections = []
        human_count = 0
        frame_count = 0
        tracked_persons = []  # List to store tracked persons (box + histogram)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Write original frame to output video
            current_frame = frame.copy()

            # Process only every 'frame_interval' frame (1 FPS)
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                results = model(np.array(image))

                frame_detections = []
                current_frame_boxes = []  # To track duplicates within the same frame

                for result in results:
                    boxes = result.boxes
                    for i, box in enumerate(boxes):
                        if int(box.cls) == HUMAN_CLASS_ID:
                            x_min = max(0, int(box.xyxy[0][0]))
                            y_min = max(0, int(box.xyxy[0][1]))
                            x_max = min(frame.shape[1], int(box.xyxy[0][2]))
                            y_max = min(frame.shape[0], int(box.xyxy[0][3]))
                            current_box = (x_min, y_min, x_max, y_max)

                            # Check for duplicates within the same frame using IoU
                            is_duplicate_in_frame = False
                            for seen_box in current_frame_boxes:
                                if calculate_iou(current_box, seen_box) > 0.5:
                                    is_duplicate_in_frame = True
                                    break

                            if is_duplicate_in_frame:
                                continue

                            current_frame_boxes.append(current_box)

                            # Crop the person for similarity comparison
                            cropped_image = image.crop((x_min, y_min, x_max, y_max))
                            cropped_image = cropped_image.convert('RGB')
                            current_hist = compute_histogram(cropped_image)

                            # Check if this person has been detected before
                            is_duplicate_person = False
                            for person in tracked_persons:
                                prev_box, prev_hist, prev_output_filename = person
                                iou_score = calculate_iou(current_box, prev_box)
                                hist_similarity = cosine_similarity([current_hist], [prev_hist])[0][0]

                                # Consider it the same person if IoU > 0.3 or histogram similarity > 0.9
                                if iou_score > 0.3 or hist_similarity > 0.9:
                                    is_duplicate_person = True
                                    # Delete the new cropped image if it exists
                                    output_filename = f"{output_dir}/human_{timestamp}_frame{frame_count}_{human_count+1}.jpg"
                                    if os.path.exists(output_filename):
                                        os.remove(output_filename)
                                    break

                            if not is_duplicate_person:
                                human_count += 1
                                output_filename = f"{output_dir}/human_{timestamp}_frame{frame_count}_{human_count}.jpg"
                                cropped_image.save(output_filename, "JPEG")
                                tracked_persons.append((current_box, current_hist, output_filename))

                                # Draw bounding box on frame
                                cv2.rectangle(current_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                                cv2.putText(current_frame, f"Human {human_count}", (x_min, y_min-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                # Prepare detection data
                                detection = {
                                    "frame_number": frame_count,
                                    "confidence": float(box.conf),
                                    "bbox": {"x_min": float(x_min), "y_min": float(y_min), "x_max": float(x_max), "y_max": float(y_max)},
                                    "cropped_image_path": output_filename
                                }
                                frame_detections.append(detection)

                if frame_detections:
                    all_detections.append({"frame": frame_count, "detections": frame_detections})

            # Write processed frame to video
            out.write(current_frame)
            frame_count += 1

        # Clean up
        cap.release()
        out.release()
        os.remove(temp_filename)

        image_files = [f for f in os.listdir('cropped_humans_video') if f.endswith(('.jpg', '.jpeg', '.png'))]
        images = [os.path.join('cropped_humans_video', image_file) for image_file in image_files]
        image_embeddings = fclip.encode_images(images, batch_size=len(image_files))
        print(image_embeddings[0].shape)
        base64_images = [image_to_base64(image) for image in images]

        if IMAGE_COLLECTION not in [c.name for c in client.get_collections().collections]:
            client.create_collection(
                collection_name=IMAGE_COLLECTION,
                vectors_config=models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created collection: {IMAGE_COLLECTION}")

        # Prepare points for Qdrant
        points = []
        for idx, (embedding, base64_image, image_path) in enumerate(zip(image_embeddings, base64_images, images)):
            points.append(
                models.PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={
                        "base64_image": base64_image,
                        "image_path": image_path,
                        "timestamp": timestamp
                    }
                )
            )

        # Upload points to Qdrant
        client.upsert(
            collection_name=IMAGE_COLLECTION,
            points=points
        )
        print(f"Uploaded {len(points)} points to Qdrant collection {IMAGE_COLLECTION}")

        # Prepare response
        response = {
            "status": "success",
            "total_human_count": human_count,
            "processed_frames": len(all_detections),
            "total_frames": total_frames,
            "detections_per_frame": all_detections,
            "processed_video_path": output_video_path,
            "message": f"Found {human_count} unique human(s) across {len(all_detections)} processed frames at 1 FPS. Video saved to {output_video_path}"
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.remove(temp_filename)
        return JSONResponse(status_code=500, content={"status": "error", "message": f"An error occurred: {str(e)}"})


@app.post("/fetch-by-text/")
async def fetch(text_query: str = Form(...)):  # Use Form to accept form data
    """
    Fetch images from Qdrant based on a text query by comparing text embedding with image embeddings
    """
    try:
        # Validate text_query
        if not text_query:
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": "Missing text_query parameter"
            })

        if not isinstance(text_query, str):
            return JSONResponse(status_code=400, content={
                "status": "error",
                "message": "text_query must be a string"
            })
        
        # Check if the Qdrant collection exists
        if IMAGE_COLLECTION not in [c.name for c in client.get_collections().collections]:
            return JSONResponse(status_code=404, content={
                "status": "error",
                "message": f"Collection {IMAGE_COLLECTION} does not exist. Please upload images first."
            })

        # Generate text embedding using FashionCLIP
        text_embedding = fclip.encode_text([text_query], batch_size=32)[0]
        print(f"Text embedding shape: {text_embedding.shape}")

        # Search for similar image embeddings in Qdrant
        search_results = client.search(
            collection_name=IMAGE_COLLECTION,
            query_vector=text_embedding.tolist(),
            limit=10,
            with_payload=True,
            with_vectors=True
        )

        # Process search results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "vector": result.vector,
                "payload": result.payload
            })

        return JSONResponse(content={
            "status": "success",
            "query": text_query,
            "results": results,
            "message": f"Found {len(results)} similar images for query '{text_query}'"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        })
    
@app.post("/fetch-by-image/")
async def fetch_by_image(file: UploadFile = File(...)):
    """
    Fetch similar images from Qdrant based on an uploaded image by comparing image embeddings
    """
    try:
        # Check if the Qdrant collection exists
        if IMAGE_COLLECTION not in [c.name for c in client.get_collections().collections]:
            return JSONResponse(status_code=404, content={
                "status": "error",
                "message": f"Collection {IMAGE_COLLECTION} does not exist. Please upload images first."
            })

        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Generate image embedding using FashionCLIP
        image_embedding = fclip.encode_images([image], batch_size=1)[0]
        print(f"Image embedding shape: {image_embedding.shape}")

        # Search for similar image embeddings in Qdrant
        search_results = client.search(
            collection_name=IMAGE_COLLECTION,
            query_vector=image_embedding.tolist(),
            limit=10,  # Retrieve top 10 most similar images
            with_payload=True,  # Include payload (base64_image, image_path, timestamp)
            with_vectors=True  # Include the vectors for reference
        )

        # Process search results
        results = []
        for result in search_results:
            results.append({
                "id": result.id,
                "score": result.score,
                "vector": result.vector,
                "payload": result.payload
            })

        return JSONResponse(content={
            "status": "success",
            "query_image": file.filename,
            "results": results,
            "message": f"Found {len(results)} similar images for the uploaded image '{file.filename}'"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        })

@app.get("/delete-collection/")
async def delete_collection():
    """
    Delete the image collection from Qdrant
    """
    try:
        dir_path = '/home/STUDENTS/pb0626/Documents/Fashion_CLIP/cropped_humans_video'
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Directory '{dir_path}' and its contents deleted successfully.")
            except OSError as e:
                print(f"Error deleting directory '{dir_path}': {e}")
        else:
            print(f"Directory '{dir_path}' does not exist.")
        # Check if the collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if IMAGE_COLLECTION not in collection_names:
            return JSONResponse(status_code=404, content={
                "status": "error",
                "message": f"Collection '{IMAGE_COLLECTION}' does not exist."
            })
        
        # Delete the collection
        client.delete_collection(collection_name=IMAGE_COLLECTION)
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Collection '{IMAGE_COLLECTION}' has been deleted successfully."
        })
    
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        })

# @app.post("/fetch/")
# async def fetch(request: Request):  # Changed to accept a string directly
#     """
#     Fetch images from Qdrant based on a text query by comparing text embedding with image embeddings
#     """
#     try:
#         # Parse JSON body
#         data = await request.json()
#         text_query = data.get("text_query")
        
#         if not text_query:
#             return JSONResponse(status_code=400, content={
#                 "status": "error",
#                 "message": "Missing text_query parameter"
#             })
        
#         # Generate text embedding using FashionCLIP
#         text_embedding = fclip.encode_text([text_query], batch_size=32)[0]
#         print(f"Text embedding shape: {text_embedding.shape}")

#         # Search for similar image embeddings in Qdrant
#         search_results = client.search(
#             collection_name=IMAGE_COLLECTION,
#             query_vector=text_embedding.tolist(),
#             limit=10,  # Retrieve top 10 most similar images
#             with_payload=True,  # Include payload (base64_image, image_path, timestamp)
#             with_vectors=True  # Include the vectors for reference
#         )

#         # Process search results
#         results = []
#         for result in search_results:
#             results.append({
#                 "id": result.id,
#                 "score": result.score,
#                 "vector": result.vector,
#                 "payload": result.payload
#             })

#         return JSONResponse(content={
#             "status": "success",
#             "query": text_query,
#             "results": results,
#             "message": f"Found {len(results)} similar images for query '{text_query}'"
#         })

#     except Exception as e:
#         return JSONResponse(status_code=500, content={
#             "status": "error",
#             "message": f"An error occurred: {str(e)}"
#         })

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "Welcome to Human Detection API",
        "model": "YOLOv8 Nano",
        "endpoints": {
            "/detect-humans/": "POST - Upload an image to detect humans"
        }
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)