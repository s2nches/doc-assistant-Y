import os
import uuid
import requests
import pandas as pd
import torch
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

# Инициализация приложения (как в методичке)
app = FastAPI(title="Doc Assistant")

# Глобальные переменные для хранения базы знаний
model = None
tokenizer = None
index = None
all_chunks = []

# Модель данных для запроса (Pydantic)
class Query(BaseModel):
    question: str
    k: int = 3

# Загрузка базы знаний при старте сервера (согласно методичке)
@app.on_event("startup")
def load():
    global model, tokenizer, index, all_chunks
    
    print("Загрузка данных и модели...")
    # 1. Читаем CSV
    df = pd.read_csv("dataset_clean.csv")
    all_texts = df['text'].dropna().tolist()
    
    # 2. Чанки
    chunk_size, overlap = 500, 50
    step = chunk_size - overlap
    for text in all_texts:
        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 50:
                all_chunks.append(chunk)
                
    # 3. Модель
    model_name = "intfloat/multilingual-e5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # 4. Векторизация
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    def get_embedding(texts):
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    embeddings_list = []
    batch_size = 32
    for i in range(0, len(all_chunks), batch_size):
        batch = ["passage: " + t for t in all_chunks[i:i+batch_size]]
        embeddings_list.append(get_embedding(batch))
    
    embeddings_array = np.vstack(embeddings_list).astype('float32')
    
    # 5. FAISS
    index = faiss.IndexFlatIP(embeddings_array.shape[1])
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)
    print("✅ База знаний загружена!")

# Эндпоинт проверки здоровья (как в методичке)
@app.get("/health")
def health():
    return {"status": "ok"}

# Главный эндпоинт для вопросов (как в методичке)
@app.post("/ask")
def ask(q: Query):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос")
        
    # Векторизация вопроса
    inputs = tokenizer(["query: " + q.question], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    q_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype('float32')
    faiss.normalize_L2(q_emb)
    
    # Поиск в FAISS
    distances, indices = index.search(q_emb, q.k)
    found_chunks = [all_chunks[idx] for idx in indices[0]]
    context = "\n\n".join(found_chunks)
    
    # Запрос к GigaChat
    auth_key = os.environ.get("GIGACHAT_AUTH_KEY")
    scope = os.environ.get("GIGACHAT_SCOPE")
    
    # Получаем токен
    token_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'RqUID': str(uuid.uuid4()),
        'Authorization': f'Bearer {auth_key}'
    }
    resp = requests.post(token_url, headers=headers, data={'scope': scope}, verify=False)
    token = resp.json()['access_token']
    
    # Генерируем ответ
    prompt = f"Ответь на вопрос ТОЛЬКО по приведённым фрагментам.\nЕсли ответа в них нет — скажи 'в документах не найдено'.\n\nФрагменты:\n{context}\n\nВопрос: {q.question}"
    
    chat_url = "https://api.giga.chat/v1/chat/completions"
    chat_headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'}
    data = {"model": "GigaChat-3-Ultra", "messages": [{"role": "user", "content": prompt}]}
    resp = requests.post(chat_url, headers=chat_headers, json=data, verify=False)
    answer = resp.json()['choices'][0]['message']['content']
    
    # Возвращаем ответ и источники
    return {"answer": answer, "sources": found_chunks}
