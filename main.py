# main.py

import os
import whisper
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
from moviepy import VideoFileClip

# --- Load API Key ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Initialize FastAPI App ---
app = FastAPI()

# --- Set Up Directories ---
UPLOAD_DIR = "uploads"
TEXT_DIR = "processed_text"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# --- Load Models ---
print("🔄 Loading AI models...")
whisper_model = whisper.load_model("base")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)
text_data = []
print("✅ Models loaded.")

# --- Root Check ---
@app.get("/")
def read_root():
    return {"message": "✅ FastAPI is running"}

# --- Upload Files ---
@app.post("/upload/")
async def upload_files(files: list[UploadFile]):
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        print(f"📂 Uploaded: {file.filename}")

        if file.filename.endswith((".mp3", ".wav")):
            process_audio(file_path)
        elif file.filename.endswith(".pdf"):
            process_pdf(file_path)
        elif file.filename.endswith((".mp4", ".mov")):
            process_video(file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")

    return {"message": "✅ Files uploaded and processed"}

# --- Process Audio ---
def process_audio(file_path):
    print(f"🎙️ Transcribing: {file_path}")
    result = whisper_model.transcribe(file_path)
    text = result["text"]
    text_filename = os.path.join(TEXT_DIR, os.path.basename(file_path) + ".txt")
    with open(text_filename, "w") as f:
        f.write(text)
    store_text_embedding(text, text_filename)

# --- Process PDF ---
def process_pdf(file_path):
    print(f"📄 Extracting from PDF: {file_path}")
    with open(file_path, "rb") as f:
        reader = PdfReader(f)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    text_filename = os.path.join(TEXT_DIR, os.path.basename(file_path) + ".txt")
    with open(text_filename, "w") as f:
        f.write(text)
    store_text_embedding(text, text_filename)

# --- Process Video ---
def process_video(file_path):
    print(f"🎥 Processing video: {file_path}")
    clip = VideoFileClip(file_path)
    audio_path = file_path.rsplit(".", 1)[0] + ".wav"
    clip.audio.write_audiofile(audio_path)
    clip.close()
    process_audio(audio_path)

# --- Store Embeddings ---
def store_text_embedding(text, source):
    global text_data
    embedding = embedding_model.encode(text).astype(np.float32)
    index.add(np.array([embedding]))
    text_data.append({"text": text, "source": source})

# --- Ask Gemini ---
class AskRequest(BaseModel):
    question: str
    top_k: int = 3

@app.post("/ask/")
async def ask_question(request: AskRequest):
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No embeddings found.")

    print(f"💡 Asking: {request.question}")
    query_embedding = embedding_model.encode(request.question).astype(np.float32)
    distances, indices = index.search(np.array([query_embedding]), request.top_k)
    retrieved = [text_data[i]["text"] for i in indices[0]]
    context = "

".join(retrieved)

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(f"Context:
{context}

Question: {request.question}")
    return {
        "question": request.question,
        "retrieved_context": retrieved,
        "answer": response.text
    }

# --- Run the app if this script is called directly (for local testing only) ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
