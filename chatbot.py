import os
import requests
import json
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# --- CẤU HÌNH HỆ THỐNG ---
CHROMA_PATH = "./eyespy_memory_db"
OLLAMA_API = "http://localhost:11434/api/generate"

# 1. Khởi tạo bộ nhớ dài hạn (Tự học)
# Sử dụng model nhúng miễn phí chạy trên CPU/GPU của bạn
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

app = Flask(__name__)

# 2. Hàm gọi bộ não Llama 3 (Chạy hoàn toàn trên máy bạn)
def call_local_ai(prompt, context):
    full_prompt = f"Kiến thức đã học: {context}\n\nCâu hỏi: {prompt}\nTrả lời như một chuyên gia EyeSpyhub:"
    
    payload = {
        "model": "llama3",
        "prompt": full_prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_API, json=payload)
    return response.json().get("response", "Lỗi kết nối bộ não.")

# 3. API xử lý yêu cầu và Tự học
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_query = data.get("query", "")

    # TRUY XUẤT: Lấy kiến thức cũ liên quan từ bộ nhớ
    docs = vector_db.similarity_search(user_query, k=3)
    context = "\n".join([d.page_content for d in docs])

    # SUY LUẬN: Gọi bộ não AI local
    answer = call_local_ai(user_query, context)

    # TỰ HỌC: Lưu câu hỏi và câu trả lời mới vào bộ nhớ vĩnh viễn
    new_knowledge = f"Q: {user_query} | A: {answer}"
    vector_db.add_documents([Document(page_content=new_knowledge)])
    vector_db.persist()

    return jsonify({
        "status": "success",
        "answer": answer,
        "memory_updated": True
    })

if __name__ == "__main__":
    print("🚀 EyeSpyhub AI Server is running locally on port 5000...")
    app.run(port=5000)
