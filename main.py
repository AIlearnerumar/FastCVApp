# main.py
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import os
from processor import apply_filter

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    print("requesting page")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process", response_class=HTMLResponse)
async def process_image(request: Request, file: UploadFile = File(...), operation: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    result = apply_filter(image, operation)
    output_path = "static/result.jpg"
    cv2.imwrite(output_path, result)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "image_url": "/" + output_path,
        "operation": operation
    })
