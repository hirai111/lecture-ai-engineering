import os
import torch
from transformers import pipeline
import time
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import nest_asyncio
from pyngrok import ngrok

# --- 設定 ---
MODEL_NAME = "google/gemma-2-2b-jpn-it"
print(f"モデル名を設定: {MODEL_NAME}")

class Config:
    def __init__(self, model_name=MODEL_NAME):
        self.MODEL_NAME = model_name

config = Config(MODEL_NAME)

# --- FastAPI アプリ定義 ---
app = FastAPI(
    title="ローカルLLM APIサービス",
    description="transformersモデルを使用したテキスト生成のためのAPI",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- データモデル ---
class Message(BaseModel):
    role: str
    content: str

class SimpleGenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 512
    do_sample: Optional[bool] = True
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class GenerationResponse(BaseModel):
    generated_text: str
    response_time: float

# Batch エンドポイント用モデル
class BatchGenerationRequest(BaseModel):
    prompts: List[str]
    max_new_tokens: Optional[int] = 512
    do_sample: Optional[bool] = True
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class BatchGenerationResponse(BaseModel):
    generated_texts: List[str]
    response_times: List[float]

# --- モデル読み込み関数 ---
model = None

def load_model():
    global model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用デバイス: {device}")
        pipe = pipeline(
            "text-generation",
            model=config.MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=0 if device == "cuda" else -1
        )
        print(f"モデル '{config.MODEL_NAME}' の読み込みに成功しました")
        model = pipe
        return pipe
    except Exception as e:
        print(f"モデル '{config.MODEL_NAME}' の読み込みに失敗: {e}")
        traceback.print_exc()
        return None


def extract_assistant_response(outputs, user_prompt):
    assistant_response = ""
    try:
        if outputs and isinstance(outputs, list) and outputs[0].get("generated_text"):
            full_text = outputs[0]["generated_text"]
            if user_prompt and full_text.startswith(user_prompt):
                assistant_response = full_text[len(user_prompt):].strip()
            else:
                assistant_response = full_text.strip()
    except Exception as e:
        print(f"応答の抽出中にエラー: {e}")
        traceback.print_exc()
        assistant_response = "応答を生成できませんでした。"
    return assistant_response or "応答を生成できませんでした。"

# --- イベント ---
@app.on_event("startup")
async def startup_event():
    print("load_model_task: モデルの読み込みを開始...")
    load_model()
    if model:
        print("起動時にモデルの初期化が完了しました。")
    else:
        print("警告: 起動時にモデルの初期化に失敗しました")

# --- エンドポイント ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "Local LLM API is running"}

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "error", "message": "No model loaded"}
    return {"status": "ok", "model": config.MODEL_NAME}

@app.post("/generate", response_model=GenerationResponse)
async def generate_simple(request: SimpleGenerationRequest):
    global model
    if model is None:
        load_model()
        if model is None:
            raise HTTPException(status_code=503, detail="モデルが利用できません。")
    start_time = time.time()
    outputs = model(
        request.prompt,
        max_new_tokens=request.max_new_tokens,
        do_sample=request.do_sample,
        temperature=request.temperature,
        top_p=request.top_p
    )
    result = extract_assistant_response(outputs, request.prompt)
    elapsed = time.time() - start_time
    return GenerationResponse(generated_text=result, response_time=elapsed)

# --- Batch エンドポイント追加 ---
@app.post("/batch_generate", response_model=BatchGenerationResponse)
async def generate_batch(request: BatchGenerationRequest):
    global model
    if model is None:
        load_model()
        if model is None:
            raise HTTPException(status_code=503, detail="モデルが利用できません。")
    texts, times = [], []
    for prompt in request.prompts:
        start = time.time()
        outputs = model(
            prompt,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p
        )
        texts.append(extract_assistant_response(outputs, prompt))
        times.append(time.time() - start)
    return BatchGenerationResponse(generated_texts=texts, response_times=times)

# --- ngrok 起動関数 ---
def run_with_ngrok(port=8501):
    nest_asyncio.apply()
    ngrok_token = os.environ.get("NGROK_TOKEN")
    if not ngrok_token:
        try:
            ngrok_token = input("Ngrok認証トークンを入力してください: ")
        except EOFError:
            print("Ngrokトークンが設定されていません")
            return
    ngrok.set_auth_token(ngrok_token)
    print("既存トンネルを切断中...")
    for t in ngrok.get_tunnels() or []:
        ngrok.disconnect(t.public_url)
    print(f"ポート{port}にトンネルを開きます...")
    tunnel = ngrok.connect(port)
    print("公開URL:", tunnel.public_url)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    run_with_ngrok(port=8501)
    print("サーバープロセスが終了しました。")
