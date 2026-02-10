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
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # 下載 LINE 圖片
    message_content = line_bot_api.get_message_content(event.message.id)
    input_image_bytes = io.BytesIO(message_content.content).read()
    
    # 立即初步回覆
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text="📸 AI 正在分析圖片並標註中，請稍候..."))

    try:
        # 呼叫 Gemini 3 Agentic Vision
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[
                types.Part.from_bytes(data=input_image_bytes, mime_type="image/jpeg"),
                "請找出圖中的重要物件，並『務必執行程式碼』來繪製紅框標註圖片。 "
                "【回傳規則】：不要在文字回覆中顯示任何 JSON 座標或數字代碼。 "
                "請直接給予繁體中文的物件總結描述即可，並附上標註好的紅框圖片。"
                "如果判斷為醫療相關的影像，可以詳細說明他可能是甚麼疾病或是任何你知道的知識"
            ],
            config=types.GenerateContentConfig(
                tools=[types.Tool(code_execution=types.ToolCodeExecution())],
                thinking_config=types.ThinkingConfig(thinking_budget=2048),
                max_output_tokens=4096
            )
        )

        text_result = ""
        annotated_image_bytes = None
        
        # 解析 AI 回傳的文字與圖片
        for part in response.candidates[0].content.parts:
            if part.text:
                # 額外過濾可能殘留的 JSON 內容
                if "box_2d" not in part.text:
                    text_result += part.text
            if part.as_image():
                annotated_image_bytes = part.as_image().image_bytes

        reply_messages = [TextSendMessage(text=text_result or "標註完成！")]
        
        # 如果 AI 有產生圖片，處理圖片網址
        if annotated_image_bytes:
            img_id = str(uuid.uuid4())
            image_store[img_id] = annotated_image_bytes
            # 確保 APP_DOMAIN 網址正確
            base_url = APP_DOMAIN if APP_DOMAIN.startswith("http") else f"https://{APP_DOMAIN}"
            img_url = f"{base_url}/images/{img_id}"
            reply_messages.append(ImageSendMessage(original_content_url=img_url, preview_image_url=img_url))

        line_bot_api.push_message(event.source.user_id, reply_messages)

    except Exception as e:
        line_bot_api.push_message(event.source.user_id, TextSendMessage(text=f"AI 分析失敗: {str(e)}"))

# 5. 啟動伺服器 (Render 會調用 uvicorn，但加上這個方便本地測試)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
