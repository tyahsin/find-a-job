from config import _require
import httpx

telegram_bot_token = _require("BOT_TOKEN")
telegram_chat_id = _require("CHAT_ID")
telegram_bot_api = f"https://api.telegram.org/bot{telegram_bot_token}"

def notify(message: str, parse_mode: str = "HTML") -> None:
    url = f"{telegram_bot_api}/sendMessage"
    httpx.post(url, json={"chat_id": telegram_chat_id, "text": message, "parse_mode": parse_mode}, timeout=10)
