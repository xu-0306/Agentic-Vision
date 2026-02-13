import os
import io
import uuid
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, ImageMessage, TextMessage, TextSendMessage, ImageSendMessage
from google import genai
from google.genai import types

# 1. 初始化與環境變數
GEMINI_KEY = os.getenv("GEMINI_KEY")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
APP_DOMAIN = os.getenv("APP_DOMAIN")

app = FastAPI()
image_store = {}        # 存放 AI 標註後的結果圖 (供 LINE 抓取)
user_last_image = {}    # 存放使用者剛傳送的原圖 (供指令分析用)

line_bot_api = LineBotApi(LINE_TOKEN)
handler = WebhookHandler(LINE_SECRET)
gemini_client = genai.Client(api_key=GEMINI_KEY)

@app.get("/")
def health_check():
    return {"status": "running"}

# 2. 圖片服務介面 (讓 LINE 抓取畫好的紅框圖)
@app.get("/images/{image_id}")
def serve_image(image_id: str):
    if image_id in image_store:
        return Response(content=image_store[image_id], media_type="image/jpeg")
    raise HTTPException(status_code=404)

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return "OK"

# 3. 處理圖片：接收並執行壓縮優化
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    user_id = event.source.user_id
    message_content = line_bot_api.get_message_content(event.message.id)
    
    # --- 圖片壓縮優化 (減輕 Zeabur 記憶體負擔) ---
    img = Image.open(io.BytesIO(message_content.content))
    if img.width > 1024:
        img.thumbnail((1024, 1024))
    
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    image_bytes = output.getvalue()
    
    # 將壓縮後的圖片存入暫存記憶
    user_last_image[user_id] = image_bytes
    
    line_bot_api.reply_message(
        event.reply_token, 
        TextSendMessage(text="📸 照片已就緒！\n請輸入指令讓 AI 執行，例如：\n「找出圖中所有貓咪」\n「這張發票的總金額是多少？」")
    )

# 4. 處理文字：作為指令觸發 Agentic Vision
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_id = event.source.user_id
    user_prompt = event.message.text
    
    # 檢查該使用者是否先傳過圖片
    if user_id not in user_last_image:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="請先傳送一張圖片，我才能幫你分析喔！"))
        return

    # 取得之前暫存的圖片
    input_image_bytes = user_last_image[user_id]
    
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"🤖 正在根據指令「{user_prompt}」思考並執行中..."))

    try:
        # 呼叫 Gemini 3 Agentic Vision
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=input_image_bytes, mime_type="image/jpeg"),
                f"使用者指令：{user_prompt}。請根據指令思考並執行 code_execution 來標註圖片。不要輸出 JSON，直接以繁體中文回覆結果。"
            ],
            config=types.GenerateContentConfig(
                tools=[types.Tool(code_execution=types.ToolCodeExecution())],
                #thinking_config=types.ThinkingConfig(thinking_budget=2048),
                max_output_tokens=4096
            )
        )

        text_result = ""
        annotated_image_bytes = None
        
        # 解析 Gemini 回傳內容
        for part in response.candidates[0].content.parts:
            if part.text:
                # 過濾掉可能噴出的 JSON 座標字串
                if "box_2d" not in part.text:
                    text_result += part.text
            if part.as_image():
                annotated_image_bytes = part.as_image().image_bytes

        reply_messages = [TextSendMessage(text=text_result or "分析完成")]
        
        # 處理並發送標註後的紅框圖
        if annotated_image_bytes:
            img_id = str(uuid.uuid4())
            image_store[img_id] = annotated_image_bytes
            
            # 確保網址格式正確
            domain = APP_DOMAIN.strip().replace("https://", "").replace("/", "")
            img_url = f"https://{domain}/images/{img_id}"
            
            reply_messages.append(ImageSendMessage(original_content_url=img_url, preview_image_url=img_url))

        line_bot_api.push_message(user_id, reply_messages)

    except Exception as e:
        line_bot_api.push_message(user_id, TextSendMessage(text=f"AI 執行出錯: {str(e)}"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
