# 🎓 AI-Powered Q&A System with FastAPI + Whisper + Gemini

This project is a **Retrieval-Augmented Generation (RAG)** system that allows users to upload documents (PDFs, audio, video) and ask questions about their content. It uses:

- ✅ FastAPI for backend API
- ✅ Whisper for transcription
- ✅ SentenceTransformers + FAISS for retrieval
- ✅ Gemini 2.0 Flash for AI-powered answers

---

## 🚀 Features

- Upload and process:
    - PDF documents
    - Audio files (`.mp3`, `.wav`)
    - Video files (`.mp4`, `.mov`)
- Transcribes audio/video using Whisper
- Extracts text from PDFs
- Generates embeddings using SentenceTransformers
- Stores embeddings in FAISS
- Answers questions using Gemini 2.0 Flash

---

## 🛠️ How to Run Locally

1. Install dependencies:
    pip install -r requirements.txt

2. Create a `.env` file with your Gemini key:
    GEMINI_API_KEY=your-gemini-api-key-here

3. Run the FastAPI server:
    uvicorn main:app --host=0.0.0.0 --port=8000 --reload

Then open:
    http://localhost:8000/docs

---

## 🔒 Environment Variables

GEMINI_API_KEY — required to use Gemini 2.0 Flash

---

## 📁 Project Structure

    .
    ├── main.py             # FastAPI app
    ├── requirements.txt    # Dependencies
    ├── .env.example        # Template for environment variables
    ├── README.md           # This file
    ├── uploads/            # Uploaded files
    ├── processed_text/     # Extracted or transcribed text

---

## 📄 License

MIT License

---

## ✨ Author

**Shubham Agarwal**  
[GitHub Profile](https://github.com/shubham119413)
