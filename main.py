import os
import io
import uuid
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response

# LINE Bot SDK v3
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi, MessagingApiBlob,
    ReplyMessageRequest, PushMessageRequest,
    TextMessage, ImageMessage as LineImageMessage,
)
from linebot.v3.webhooks import MessageEvent, ImageMessageContent, TextMessageContent

from google import genai
from google.genai import types

# 1. 初始化與環境變數
GEMINI_KEY  = os.getenv("GEMINI_KEY")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_TOKEN  = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
APP_DOMAIN  = os.getenv("APP_DOMAIN")

app = FastAPI()
image_store     = {}
user_last_image = {}

configuration  = Configuration(access_token=LINE_TOKEN)
handler        = WebhookHandler(LINE_SECRET)
gemini_client  = genai.Client(api_key=GEMINI_KEY)

@app.get("/")
def health_check():
    return {"status": "running"}

# 2. 圖片服務介面
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

# 3. 處理圖片
@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image(event):
    user_id = event.source.user_id

    with ApiClient(configuration) as api_client:
        blob_api = MessagingApiBlob(api_client)
        image_bytes_raw = blob_api.get_message_content(event.message.id)

    img = Image.open(io.BytesIO(image_bytes_raw))
    if img.width > 1024:
        img.thumbnail((1024, 1024))

    output = io.BytesIO()
    img.save(output, format="JPEG", quality=85)
    user_last_image[user_id] = output.getvalue()

    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text="📸 照片已就緒！\n請輸入指令讓 AI 執行，例如：\n「找出圖中所有貓咪」\n「這張發票的總金額是多少？」")]
            )
        )

# 4. 處理文字指令
@handler.add(MessageEvent, message=TextMessageContent)
def handle_text(event):
    user_id     = event.source.user_id
    user_prompt = event.message.text

    with ApiClient(configuration) as api_client:
        messaging_api = MessagingApi(api_client)

        if user_id not in user_last_image:
            messaging_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text="請先傳送一張圖片，我才能幫你分析喔！")]
                )
            )
            return

        messaging_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=f"🤖 正在根據指令「{user_prompt}」思考並執行中...")]
            )
        )

    input_image_bytes = user_last_image[user_id]

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[
                types.Part.from_bytes(data=input_image_bytes, mime_type="image/jpeg"),
                f"""你是一個具備視覺標註能力的 AI 助手。
使用者指令：{user_prompt}

執行規範：
1. 你『必須』使用 code_execution (Python) 來處理圖片並繪製標註框。
2. 如果指令要求找東西，請務必在圖上畫出紅色方框。
3. 嚴禁只輸出數字座標或 JSON，請將結果轉化為易懂的繁體中文描述。
4. 如果你沒有畫圖，請解釋原因。"""
            ],
            config=types.GenerateContentConfig(
                tools=[types.Tool(code_execution=types.ToolCodeExecution())],
                thinking_config=types.ThinkingConfig(thinking_budget=2048),
                max_output_tokens=4096
            )
        )

        text_result          = ""
        annotated_image_bytes = None

        for part in response.candidates[0].content.parts:
            if part.text and "box_2d" not in part.text:
                text_result += part.text
            if part.as_image():
                annotated_image_bytes = part.as_image().image_bytes

        reply_messages = [TextMessage(text=text_result or "分析完成")]

        if annotated_image_bytes:
            img_id = str(uuid.uuid4())
            image_store[img_id] = annotated_image_bytes
            domain  = APP_DOMAIN.strip().replace("https://", "").replace("/", "")
            img_url = f"https://{domain}/images/{img_id}"
            reply_messages.append(
                LineImageMessage(original_content_url=img_url, preview_image_url=img_url)
            )

        with ApiClient(configuration) as api_client:
            MessagingApi(api_client).push_message(
                PushMessageRequest(to=user_id, messages=reply_messages)
            )

    except Exception as e:
        with ApiClient(configuration) as api_client:
            MessagingApi(api_client).push_message(
                PushMessageRequest(to=user_id, messages=[TextMessage(text=f"AI 執行出錯: {str(e)}")])
            )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
