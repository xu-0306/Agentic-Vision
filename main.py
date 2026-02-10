import os
import io
import uuid
import warnings
import requests
import uvicorn
from PIL import Image
from threading import Thread
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage, ImageSendMessage
from linebot import LineBotSdkDeprecatedIn30
from google import genai
from google.genai import types
from pyngrok import ngrok

# 1. 忽略過時警告與環境清理
warnings.filterwarnings("ignore", category=LineBotSdkDeprecatedIn30)
!fuser -k 8000/tcp  # 強制關閉佔用 8000 port 的舊程序

# 2. 設定區 (已填入你提供的資訊)
GEMINI_KEY = "AIzaSyAVJ1e0v5ramDHm2MpQqBaYGGlB-EjQBjg"
LINE_SECRET = "1444f2a0008343b59de050ac52037ef4"
LINE_TOKEN = "vpoqYbSS4cDYLIXaMyBNUaFX+NnJIebG4MyXX660CwzVRjcfOw81BwtGQcc+XRFVwdkBa5NWOm6N0TJrLCEm4635bpQplR7PEFu3xEUNSVVz8uGucTjIpsQpgN5CoXERfXkQ9pen/odZK0yzH72emAdB04t89/1O/w1cDnyilFU="
NGROK_TOKEN = "39R0KGOYN9YAiu1X5cuggRDtjiS_6GeLfPjDQf6KTqiULGWMf"

# 3. 初始化客戶端
app = FastAPI()
image_store = {}
line_bot_api = LineBotApi(LINE_TOKEN)
handler = WebhookHandler(LINE_SECRET)
gemini_client = genai.Client(api_key=GEMINI_KEY)

# --- 圖片服務介面 ---
@app.get("/images/{image_id}")
def serve_image(image_id: str):
    if image_id in image_store:
        return Response(content=image_store[image_id], media_type="image/jpeg")
    raise HTTPException(status_code=404)

# --- LINE Webhook 入口 ---
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return "OK"

# --- 處理圖片與 Gemini 標註 ---
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # 下載圖片
    message_content = line_bot_api.get_message_content(event.message.id)
    input_image_bytes = io.BytesIO(message_content.content).read()
    
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="📸 收到！AI 調查員正在分析圖片..."))

    try:
        # Agentic Vision 調用
        # 修改這部分
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=input_image_bytes, mime_type="image/jpeg"),
                "請找出圖中的重要物件，並『務必執行程式碼』來繪製紅框標註圖片。 "
                "【回傳規則】：不要在文字回覆中顯示任何 JSON 座標或數字代碼。 "
                "請直接給予繁體中文的物件總結描述即可，並附上標註好的紅框圖片。"
            ],
            config=types.GenerateContentConfig(
                tools=[types.Tool(code_execution=types.ToolCodeExecution())],
                thinking_config=types.ThinkingConfig(thinking_budget=2048),
                max_output_tokens=4096
            )
        )

        text_result = ""
        annotated_image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.text: text_result += part.text
            if part.as_image(): annotated_image_bytes = part.as_image().image_bytes

        reply_messages = [TextSendMessage(text=text_result or "標註完成")]
        
        if annotated_image_bytes:
            img_id = str(uuid.uuid4())
            image_store[img_id] = annotated_image_bytes
            img_url = f"{public_url}/images/{img_id}"
            reply_messages.append(ImageSendMessage(original_content_url=img_url, preview_image_url=img_url))

        line_bot_api.push_message(event.source.user_id, reply_messages)

    except Exception as e:
        line_bot_api.push_message(event.source.user_id, TextSendMessage(text=f"AI 標註失敗: {str(e)}"))

# --- 自動更新 LINE Webhook 函式 ---
def update_line_webhook(new_url):
    endpoint = "https://api.line.me/v2/bot/channel/webhook/endpoint"
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {"endpoint": f"{new_url}/callback"}
    r = requests.put(endpoint, headers=headers, json=payload)
    if r.status_code == 200:
        print(f"✅ LINE 後台 Webhook 已自動更新為: {new_url}/callback")
    else:
        print(f"❌ 自動更新失敗: {r.text}")

# --- 執行啟動流程 ---
try:
    ngrok.kill()
    ngrok.set_auth_token(NGROK_TOKEN)
    tunnel = ngrok.connect(8000)
    public_url = tunnel.public_url
    
    print(f"\n✨ 伺服器啟動成功 ✨")
    print(f"🔗 臨時網址: {public_url}")
    
    # 執行自動更新網址
    update_line_webhook(public_url)
    
    def start_server():
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
    
    Thread(target=start_server).start()
    print("\n🚀 機器人已就緒！現在你可以直接傳圖片測試了。")
except Exception as e:
    print(f"❌ 啟動失敗：{e}")
