import asyncio
import collections
import json
import logging
import os
import ssl
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import BackgroundTasks
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from linebot.v3 import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import ApiClient
from linebot.v3.messaging import Configuration
from linebot.v3.messaging import MessagingApi
from linebot.v3.messaging import PushMessageRequest
from linebot.v3.messaging import TextMessage
from linebot.v3.webhooks import GroupSource
from linebot.v3.webhooks import JoinEvent
from linebot.v3.webhooks import LeaveEvent
from linebot.v3.webhooks import MemberJoinedEvent
from linebot.v3.webhooks import MessageEvent
from linebot.v3.webhooks import RoomSource
from linebot.v3.webhooks import Source
from linebot.v3.webhooks import TextMessageContent
from linebot.v3.webhooks import UserSource

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Load Environment Variables ---
# 請確保您為每個 Bot 使用了正確的 .env 檔案或環境變數
load_dotenv(".env/.env_sherry_bot_v2")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BOT_USER_ID = os.getenv("BOT_USER_ID")

# --- Debug: Log loaded BOT_USER_ID ---
logging.info(f"載入的 BOT_USER_ID 環境變數值為: '{BOT_USER_ID}'")
if not BOT_USER_ID:
    logging.critical("錯誤：BOT_USER_ID 環境變數未設定或載入失敗。")


# --- Check Environment Variables ---
if not all(
    [LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET, GEMINI_API_KEY, BOT_USER_ID]
):
    logging.critical(
        "錯誤：請設定 LINE_CHANNEL_ACCESS_TOKEN, LINE_CHANNEL_SECRET, GEMINI_API_KEY 和 BOT_USER_ID 環境變數。"
    )
    exit()

# --- Constants and Configuration ---
HISTORY_DIR = Path("history")
HISTORY_DIR.mkdir(exist_ok=True)
HISTORY_FILE = HISTORY_DIR / "conversation_history_sherry_bot.json"
HISTORY_LENGTH = 20  # Store 10 rounds (10 user + 10 model)

# --- Define Base Prompt ---
BASE_PROMPT = "你是一個小助手"
logging.info(f"基本 Prompt 已設定為常數: '{BASE_PROMPT}'")


# --- LINE Bot SDK Configuration ---
line_configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
line_parser = WebhookParser(LINE_CHANNEL_SECRET)

# --- Gemini API Configuration ---
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", system_instruction=BASE_PROMPT
)


# --- Conversation History Functions ---
def load_history_from_file(filepath: Path) -> dict[str, collections.deque]:
    """Loads conversation history from a JSON file."""
    if not filepath.exists():
        logging.warning(f"歷史紀錄檔案 {filepath} 不存在，將創建新的歷史紀錄。")
        return {}
    try:
        with filepath.open("r", encoding="utf-8") as f:
            data_with_lists = json.load(f)
            history = {}
            for user_id, messages in data_with_lists.items():
                if isinstance(messages, list):
                    history[user_id] = collections.deque(
                        messages, maxlen=HISTORY_LENGTH
                    )
                else:
                    logging.warning(
                        f"載入歷史紀錄時發現使用者 {user_id} 的資料格式不正確(非列表)，已跳過。"
                    )
            logging.info(f"成功從 {filepath} 載入 {len(history)} 位使用者的歷史紀錄。")
            return history
    except json.JSONDecodeError:
        logging.error(
            f"歷史紀錄檔案 {filepath} 格式錯誤或已損壞，將創建新的歷史紀錄。",
            exc_info=True,
        )
        return {}
    except Exception as e:
        logging.error(f"載入歷史紀錄檔案 {filepath} 時發生未預期錯誤。", exc_info=True)
        return {}


def save_history_to_file(filepath: Path, history_data: dict[str, collections.deque]):
    """Saves conversation history to a JSON file atomically."""
    data_with_lists = {
        user_id: list(messages) for user_id, messages in history_data.items()
    }

    temp_filepath = filepath.with_suffix(".tmp")
    try:
        with temp_filepath.open("w", encoding="utf-8") as f:
            json.dump(data_with_lists, f, ensure_ascii=False, indent=4)

        os.replace(temp_filepath, filepath)
        logging.debug(f"歷史紀錄已成功儲存至 {filepath}")

    except Exception as e:
        logging.error(f"儲存歷史紀錄至 {filepath} 時發生錯誤。", exc_info=True)
        if temp_filepath.exists():
            try:
                temp_filepath.unlink()
            except OSError:
                logging.error(f"刪除臨時歷史紀錄檔 {temp_filepath} 失敗。")


# --- Load Initial History ---
conversation_histories = load_history_from_file(HISTORY_FILE)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Gemini LINE Bot (Persistent, No Chunking, Mention Reply Debug)",
    description="LINE Bot with Gemini, persistent history, persona, and mention reply in groups (with debug logs).",
)


# --- Background Task Function ---
async def process_and_send_message(source: Source, user_message: str):
    """Gets Gemini response, sends it as a single message to the correct source, updates history, and saves."""
    # Determine the target ID based on the source type
    if isinstance(source, UserSource):
        target_id = source.user_id
        logging.info(f"訊息來源：個人聊天 (使用者 ID: {target_id})")
    elif isinstance(source, GroupSource):
        target_id = source.group_id
        logging.info(f"訊息來源：群組 (群組 ID: {target_id})")
    elif isinstance(source, RoomSource):
        target_id = source.room_id
        logging.info(f"訊息來源：多人聊天室 (聊天室 ID: {target_id})")
    else:
        logging.warning(f"收到未知來源類型的訊息: {type(source)}. 無法回覆。")
        return  # Cannot reply to unknown source types

    # Use the target_id for history tracking.
    # In group/room chats, history is tracked per group/room ID.
    # In 1:1 chat, history is tracked per user ID.
    history_key = target_id

    logging.info(f"背景任務開始：處理來源 {target_id} 的訊息: '{user_message}'")

    try:
        # --- Get or Initialize History ---
        global conversation_histories
        if history_key not in conversation_histories:
            # Initialize history for this source (user, group, or room)
            conversation_histories[history_key] = collections.deque(
                maxlen=HISTORY_LENGTH
            )
            logging.info(f"為新來源 {history_key} 初始化對話歷史。")

        # Get the current history for this source
        history_deque = conversation_histories[history_key]
        # Convert deque to list for Gemini API format
        current_history_list = list(history_deque)

        # --- Prepare messages for Gemini API ---
        # The history sent to Gemini includes the last HISTORY_LENGTH/2 turns (user+model)
        # that the bot has processed for this specific source (user, group, or room).
        # This is NOT the full chat history of the LINE group/room.
        logging.info(f"準備傳送給 Gemini 的歷史紀錄長度: {len(current_history_list)}")
        messages_for_gemini = current_history_list + [
            {"role": "user", "parts": [user_message]}
        ]

        # --- Call Gemini API (in thread) ---
        logging.info(f"正在呼叫 Gemini API for source {history_key}...")
        response = await asyncio.to_thread(
            gemini_model.generate_content, messages_for_gemini
        )
        logging.info(f"Gemini API 呼叫完成 for source {history_key}.")

        # --- Process Gemini Response ---
        gemini_full_reply = ""
        if response.parts:
            gemini_full_reply = response.text
        else:
            gemini_full_reply = "嗯...我好像不知道該說什麼了。"
            try:
                logging.warning(
                    f"Gemini 回應 for source {history_key} 沒有 parts. PF: {response.prompt_feedback}"
                )
            except Exception:
                logging.warning(f"Gemini 回應 for source {history_key} 沒有 parts.")

        logging.info(
            f"Gemini 完整回應 for source {history_key}: {gemini_full_reply[:100]}..."
        )

        # --- Update History (In Memory First) ---
        # Append the current user message and the bot's reply to the history deque for this source.
        history_deque.append({"role": "user", "parts": [user_message]})
        history_deque.append({"role": "model", "parts": [gemini_full_reply]})
        logging.info(
            f"來源 {history_key} 的記憶體歷史紀錄已更新，目前長度: {len(history_deque)}"
        )

        # --- Send Full Message ---
        if not gemini_full_reply or gemini_full_reply.isspace():
            logging.warning(
                f"Gemini 回應為空或只有空白字元，不傳送訊息給來源 {target_id}。"
            )
        else:
            logging.info(f"傳送完整回應給 {target_id}: {gemini_full_reply[:50]}...")
            try:
                with ApiClient(line_configuration) as api_client:
                    line_bot_api = MessagingApi(api_client)
                    push_request = PushMessageRequest(
                        to=target_id,
                        messages=[
                            TextMessage(text=gemini_full_reply)
                        ],  # Use target_id here
                    )
                    await asyncio.to_thread(line_bot_api.push_message, push_request)
                logging.info(f"完整訊息已傳送完畢給來源 {target_id}")
            except Exception as push_err:
                logging.error(f"推送完整訊息給 {target_id} 時失敗。", exc_info=True)

        # --- Save Updated History to File ---
        await asyncio.to_thread(
            save_history_to_file, HISTORY_FILE, conversation_histories
        )

    except Exception as e:
        logging.error(
            f"背景任務處理來源 {target_id} 訊息時發生嚴重錯誤。", exc_info=True
        )
        # Try to send a final error message back to the source
        try:
            with ApiClient(line_configuration) as api_client:
                line_bot_api = MessagingApi(api_client)
                await asyncio.to_thread(
                    line_bot_api.push_message,
                    PushMessageRequest(
                        to=target_id,  # Send error message to the source
                        messages=[
                            TextMessage(text="抱歉，處理您的訊息時發生了一些內部問題。")
                        ],
                    ),
                )
        except Exception as final_err:
            logging.error(f"連回覆背景任務錯誤訊息都失敗了。", exc_info=True)


# --- Webhook Endpoint ---
@app.post("/callback")
async def callback(request: Request, background_tasks: BackgroundTasks):
    """Handles incoming LINE webhooks, validates, parses, and schedules tasks."""
    signature = request.headers.get("X-Line-Signature")
    if not signature:
        logging.warning("接收到請求但缺少 X-Line-Signature 標頭")
        raise HTTPException(status_code=400, detail="Missing X-Line-Signature header")

    body_bytes = await request.body()
    body_text = body_bytes.decode("utf-8")
    logging.info(f"Webhook Request body: {body_text}")

    try:
        events = line_parser.parse(body_text, signature)
    except InvalidSignatureError:
        logging.error("Webhook 簽名驗證失敗")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        logging.error(f"處理 Webhook 事件解析時發生錯誤。", exc_info=True)
        raise HTTPException(status_code=400, detail="Error parsing webhook request")

    for event in events:
        try:  # Add try-except around individual event processing
            # Handle MessageEvent with TextMessageContent
            if isinstance(event, MessageEvent) and isinstance(
                event.message, TextMessageContent
            ):
                source = event.source
                user_message = event.message.text
                logging.info(
                    f"Webhook 事件觸發：來源類型 {type(source)}, 訊息: '{user_message}'"
                )

                should_reply = False
                # In personal chat, always reply
                if isinstance(source, UserSource):
                    should_reply = True
                    logging.info("個人聊天，將回覆。")
                # In group or room chat, only reply if bot is mentioned
                elif isinstance(source, (GroupSource, RoomSource)):
                    logging.info(f"收到群組/多人聊天室訊息。檢查提及...")
                    # Debug: Log mention object if present
                    if event.message.mention:
                        logging.info(f"訊息包含提及: {event.message.mention}")
                        if event.message.mention.mentionees:
                            logging.info(
                                f"提及列表: {event.message.mention.mentionees}"
                            )
                            # Iterate through mentioned users/bots
                            for mentionee in event.message.mention.mentionees:
                                logging.info(
                                    f"檢查提及對象 User ID: {mentionee.user_id}"
                                )
                                # Check if the mentioned user ID is the bot's own user ID
                                if BOT_USER_ID and mentionee.user_id == BOT_USER_ID:
                                    should_reply = True
                                    logging.info("在群組/多人聊天室中被提及，將回覆。")
                                    break  # Found the bot mention, no need to check others
                        else:
                            logging.info("訊息包含提及，但提及列表為空。")
                    else:
                        logging.info("訊息不包含提及。")

                if should_reply:
                    background_tasks.add_task(
                        process_and_send_message, source, user_message
                    )
                    logging.info(f"已將來源 {source.type} 的訊息處理加入背景任務。")
                else:
                    logging.info(
                        f"來源 {source.type} 的文字訊息未觸發回覆 (非個人聊天且未被提及)。"
                    )

            # Explicitly handle JoinEvent
            elif isinstance(event, JoinEvent):
                logging.info(
                    f"Webhook 事件觸發：Bot 加入事件。來源類型: {type(event.source)}"
                )
                logging.info(
                    f"成功處理 JoinEvent for source {type(event.source)}. Preparing to return 200 OK."
                )

            # Explicitly handle MemberJoinedEvent (when members join a group/room the bot is in)
            elif isinstance(event, MemberJoinedEvent):
                logging.info(
                    f"Webhook 事件觸發：成員加入事件。來源類型: {type(event.source)}"
                )
                logging.info(
                    f"成功處理 MemberJoinedEvent for source {type(event.source)}. Preparing to return 200 OK."
                )

            # Explicitly handle LeaveEvent (for debugging purposes)
            elif isinstance(event, LeaveEvent):
                logging.info(
                    f"Webhook 事件觸發：Bot 離開事件。來源類型: {type(event.source)}"
                )
                logging.info(
                    f"成功處理 LeaveEvent for source {type(event.source)}. Preparing to return 200 OK."
                )

            # Handle other unexpected event types
            else:
                logging.info(f"收到非預期事件類型: {type(event)}")

        except Exception as event_process_err:
            logging.error(
                f"處理 Webhook 事件 {type(event)} 時發生錯誤。", exc_info=True
            )
            # Continue processing other events if possible, but log the error

    logging.info("Webhook 事件處理迴圈完畢，已排定背景任務。立即回傳 OK。")
    return (
        "OK"  # Ensure 200 OK is returned even if individual event processing had errors
    )


# --- ASGI Server Startup (using uvicorn) ---
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    use_reload = os.environ.get("UVICORN_RELOAD", "true").lower() == "true"

    logging.info(f"啟動伺服器於 http://{host}:{port} (Reload: {use_reload})")
    # 確保這裡執行的模組名稱是正確的，例如如果檔案名是 main.py，這裡應該是 "main:app"
    # 如果您將檔案命名為 angry_bot.py，則 "angry_bot:app" 是正確的
    ssl_context = ssl.SS
    uvicorn.run("bot:app", host=host, port=port, reload=use_reload, ssl=)
