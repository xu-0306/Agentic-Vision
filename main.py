import os
import io
import uuid
import warnings
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextSendMessage, ImageSendMessage
from google import genai
from google.genai import types

# 1. 初始化與設定
# 這裡改從 Render 的 Environment Variables 讀取
GEMINI_KEY = os.getenv("GEMINI_KEY")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
APP_DOMAIN = os.getenv("APP_DOMAIN")  # 例如：xxx.onrender.com

app = FastAPI()
image_store = {}
line_bot_api = LineBotApi(LINE_TOKEN)
handler = WebhookHandler(LINE_SECRET)
gemini_client = genai.Client(api_key=GEMINI_KEY)

# 2. 圖片服務介面 (讓 LINE 抓取標註後的圖)
@app.get("/images/{image_id}")
def serve_image(image_id: str):
    if image_id in image_store:
        return Response(content=image_store[image_id], media_type="image/jpeg")
    raise HTTPException(status_code=404)

# 3. LINE Webhook 入口
@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return "OK"

# 4. 核心邏輯：處理圖片與 Agentic Vision
# 新增一個字典來紀錄每個使用者最後傳送的圖片內容
user_last_image = {}

# --- 處理圖片：純接收並儲存 ---
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    user_id = event.source.user_id
    message_content = line_bot_api.get_message_content(event.message.id)
    image_bytes = io.BytesIO(message_content.content).read()
    
    # 將圖片存入記憶（以 user_id 為鍵）
    user_last_image[user_id] = image_bytes
    
    line_bot_api.reply_message(
        event.reply_token, 
        TextSendMessage(text="📸 照片已就緒！\n現在請輸入指令，例如：\n「找出圖中所有貓咪」\n「這張發票的總金額是多少？」")
    )

# --- 處理文字：作為指令來觸發 Agentic Vision ---
from linebot.models import TextMessage
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_id = event.source.user_id
    user_prompt = event.message.text
    
    # 檢查該使用者是否先傳過圖片
    if user_id not in user_last_image:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請先傳送一張圖片，我才能幫你分析喔！"))
        return

    # 取得暫存的圖片
    input_image_bytes = user_last_image[user_id]
    
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"🤖 正在根據指令「{user_prompt}」思考並執行中..."))

    try:
        # 呼叫 Gemini 3
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=input_image_bytes, mime_type="image/jpeg"),
                f"使用者指令：{user_prompt}。請根據指令思考並執行 code_execution 來標註圖片。不要輸出 JSON，直接以繁體中文回覆結果。"
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
            if part.text and "box_2d" not in part.text:
                text_result += part.text
            if part.as_image():
                annotated_image_bytes = part.as_image().image_bytes

        reply_messages = [TextSendMessage(text=text_result or "執行完畢")]
        
        if annotated_image_bytes:
            img_id = str(uuid.uuid4())
            image_store[img_id] = annotated_image_bytes
            base_url = APP_DOMAIN if APP_DOMAIN.startswith("http") else f"https://{APP_DOMAIN}"
            img_url = f"{base_url}/images/{img_id}"
            reply_messages.append(ImageSendMessage(original_content_url=img_url, preview_image_url=img_url))

        line_bot_api.push_message(user_id, reply_messages)

    except Exception as e:
        line_bot_api.push_message(user_id, TextSendMessage(text=f"執行出錯: {str(e)}"))
