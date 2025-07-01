import os, shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from vision.detect_from_video import process_frame_stream

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("temp_videos", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

latest_video_path = None

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global latest_video_path
    latest_video_path = os.path.join("temp_videos", file.filename)
    with open(latest_video_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    print(f"[UPLOAD] saved raw video: {latest_video_path}")
    return {"message": "Uploaded", "stream_url": "/video_feed"}

@app.get("/video_feed")
def stream():
    if not latest_video_path or not os.path.exists(latest_video_path):
        return {"error": "No video uploaded"}
    return StreamingResponse(process_frame_stream(latest_video_path),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/dashboard", response_class=HTMLResponse)
def dash():
    return """
    <!DOCTYPE html><html><head>
    <title>Live Stream</title>
    <style>body{background:#121212;color:#00ff90;font-family:monospace;text-align:center}
    img{border:3px solid #00ff90;border-radius:10px;margin-top:20px}
    </style></head><body>
      <h1>🚘 Real-Time Vehicle Tracking</h1>
      <img src="/video_feed" width="800"/>
    </body></html>
    """
