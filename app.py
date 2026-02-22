import os
import time
import csv
import numpy as np
import tensorflow as tf

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ===================== CONFIG =====================
MODEL_PATH = "solar_model.h5"
IMG_SIZE = (224, 224)

CLASS_NAMES = ['Bird_Droppings', 'Clean', 'Dusty', 'Physical_Damage', 'Snow']

SAVE_DIR = "tg_incoming"
LOG_FILE = "alerts_log.csv"

CONF_THRESHOLD = 0.60

# IMPORTANT: get from Render environment
BOT_TOKEN = os.getenv("8491303460:AAH_O-Sd1m6hyCG2HMX5hIG9QHiiEzTZjTc")

ALLOWED_CHAT_ID = os.getenv("7428489851", None)

# ==================================================

ADVICE = {
    "Clean": {
        "urgency": "Normal",
        "remedy": "No action needed. Continue normal monitoring.",
        "risks": "No immediate risk.",
        "prevention": "Regular inspection schedule; keep records."
    },
    "Dusty": {
        "urgency": "Soon",
        "remedy": "Clean with soft brush/microfiber + clean water.",
        "risks": "Energy loss; hotspot risk.",
        "prevention": "Regular cleaning schedule."
    },
    "Bird_Droppings": {
        "urgency": "Soon",
        "remedy": "Soak and wipe gently.",
        "risks": "Hotspot risk.",
        "prevention": "Use bird deterrents."
    },
    "Snow": {
        "urgency": "Soon",
        "remedy": "Remove snow safely.",
        "risks": "Energy loss.",
        "prevention": "Seasonal checks."
    },
    "Physical_Damage": {
        "urgency": "Urgent",
        "remedy": "Inspect immediately and replace panel.",
        "risks": "Fire/shock risk.",
        "prevention": "Avoid impacts."
    }
}


def ensure_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)


def init_csv():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "filename", "class", "confidence", "urgency"])


def log_to_csv(filename, pred_class, conf, urgency):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts, filename, pred_class, f"{conf:.4f}", urgency])


def is_allowed(update: Update):
    if ALLOWED_CHAT_ID is None:
        return True
    return str(update.effective_chat.id) == str(ALLOWED_CHAT_ID)


# Load model once
MODEL = tf.keras.models.load_model(MODEL_PATH)


def predict_one(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    probs = MODEL.predict(x, verbose=0)[0]

    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])


def build_message(filename, pred_class, conf):
    if conf < CONF_THRESHOLD:
        return f"⚠️ Uncertain result\nConfidence: {conf*100:.2f}%", "Check"

    info = ADVICE[pred_class]
    urgency = info["urgency"]

    msg = (
        f"⚠️ Solar Panel Fault Report\n"
        f"File: {filename}\n"
        f"Fault: {pred_class}\n"
        f"Confidence: {conf*100:.2f}%\n"
        f"Urgency: {urgency}\n\n"
        f"Remedy:\n- {info['remedy']}\n\n"
        f"Risk if ignored:\n- {info['risks']}\n\n"
        f"Prevention:\n- {info['prevention']}"
    )

    return msg, urgency


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send solar panel image 📷")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update):
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    photo = update.message.photo[-1]
    filename = f"{photo.file_unique_id}.jpg"
    path = os.path.join(SAVE_DIR, filename)

    file = await context.bot.get_file(photo.file_id)
    await file.download_to_drive(path)

    pred_class, conf = predict_one(path)
    report, urgency = build_message(filename, pred_class, conf)

    with open(path, "rb") as f:
        await update.message.reply_photo(photo=f, caption=report)

    log_to_csv(filename, pred_class, conf, urgency)


def main():
    ensure_dirs()
    init_csv()

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # ===== WEBHOOK PART =====
    port = int(os.environ.get("PORT", 10000))
    PUBLIC_URL = os.getenv("PUBLIC_URL")

    app.run_webhook(
        listen="0.0.0.0",
        port=port,
        url_path=BOT_TOKEN,
        webhook_url=f"{PUBLIC_URL}/{BOT_TOKEN}"
    )


if __name__ == "__main__":
    main()