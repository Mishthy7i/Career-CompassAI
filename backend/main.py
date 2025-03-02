from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException,File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from PIL import Image
import pytesseract
from io import BytesIO
import fitz  # PyMuPDF

app = FastAPI()

# CORS middleware,
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

client = OpenAI(api_key="")

# Store conversation history
conversation_history: List[Dict] = []


class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None
    is_context_setting: bool = False
    chat_mode: bool = False

# # test home route.
# @app.get("/")
# def read_root():
#     return {"status": "API is running"}

# frontend directory
frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")
app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")

# Serve all HTML files
@app.get("/{path:path}")
def serve_static_files(path: str):
    file_path = os.path.join(frontend_path, path)
    
    allowed_files = ['index.html', 'user-mode.html']
    
    if path in allowed_files or path.endswith('.html'):
        if os.path.exists(file_path):
            return FileResponse(file_path)
    
    return FileResponse(os.path.join(frontend_path, 'index.html'))

@app.post("/api/chat")
def chat(request: ChatRequest) -> Dict:
    try:
        global conversation_history

        if request.is_context_setting and request.context:
            conversation_history.append({
                "role": "system",
                "content": f"Context: {request.context}"
            })
            conversation_history.append({
                "role": "system",
                "content": f"Instructions about context: {request.message}"
            })
            if request.chat_mode:
                messages = conversation_history.copy()
                messages.append({"role": "user", "content": request.message})

                completion = client.chat.completions.create(messages=messages,
                                                            model="gpt-4o",
                                                            stream=False)

                response = completion.choices[0].message.content
                return {"response": response}

            return {"status": "Context set successfully"}

        messages = conversation_history.copy()
        messages.append({"role": "user", "content": request.message})

        completion = client.chat.completions.create(messages=messages,
                                                    model="gpt-4o",
                                                    stream=False)

        response = completion.choices[0].message.content

        conversation_history.append({"role": "assistant", "content": response})

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
def reset_conversation():
    global conversation_history
    conversation_history = []
    return {"status": "Conversation reset successfully"}

# {
#     "message": "info about user",
#     "context": "i like fruits",
#     "is_context_setting": true,
#     "chat_mode": false
# }
# {
#     "message": "tell me about myself",
#     "context": "you are his girlfriend, tell him with love",
#     "is_context_setting": true,
#     "chat_mode": true
# }

if not os.path.exists("uploaded-images"):
    os.makedirs("uploaded-images")

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" #tessaract OCR path 

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...)):
    try:
        file_data = await file.read()

        if file.content_type == 'application/pdf':
            pdf_document = fitz.open(stream=file_data, filetype="pdf")
            text = ""
            for page in pdf_document:
                text += page.get_text()

            pdf_filename = f"uploaded-images/{file.filename}"
            with open(pdf_filename, "wb") as f:
                f.write(file_data)

            return JSONResponse(content={"extracted_text": text, "saved_file_path": pdf_filename})

        image = Image.open(BytesIO(file_data))

        image_filename = f"uploaded-images/{file.filename}"
        image.save(image_filename)

        text = pytesseract.image_to_string(image)

        return JSONResponse(content={"extracted_text": text, "saved_image_path": image_filename})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)