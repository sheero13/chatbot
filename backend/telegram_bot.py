from telegram import Update

from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

from deep_translator import GoogleTranslator

from langdetect import detect

from graph import app_graph


# =========================================
# TELEGRAM BOT TOKEN
# =========================================

BOT_TOKEN = "8957560769:AAHDHPgJ290Ht1RW_OMepf2sHgrsA9I75vY"


# =========================================
# START COMMAND
# =========================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "Hello! I am your SSN College AI Assistant."
    )


# =========================================
# HANDLE USER MESSAGE
# =========================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    # User message
    user_message = update.message.text

    print("\n========================")
    print("Original Message:")
    print(user_message)

    # =====================================
    # LANGUAGE DETECTION
    # =====================================

    try:

        detected_lang = detect(user_message)

    except:

        detected_lang = "en"

    print("Detected Language:", detected_lang)

    # =====================================
    # TRANSLATE TO ENGLISH
    # =====================================

    if detected_lang != "en":

        translated_question = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(user_message)

    else:

        translated_question = user_message

    print("Translated Question:")
    print(translated_question)

    # =====================================
    # RAG CHATBOT
    # =====================================

    result = app_graph.invoke({
        "question": translated_question
    })

    bot_reply = result["answer"]

    print("Bot Reply:")
    print(bot_reply)

    # =====================================
    # TRANSLATE BACK
    # =====================================

    if detected_lang != "en":

        final_reply = GoogleTranslator(
            source="en",
            target=detected_lang
        ).translate(bot_reply)

    else:

        final_reply = bot_reply

    print("Final Reply:")
    print(final_reply)

    # =====================================
    # SEND MESSAGE
    # =====================================

    await update.message.reply_text(final_reply)


# =========================================
# MAIN
# =========================================

def main():

    app = ApplicationBuilder().token(
        BOT_TOKEN
    ).build()

    # Commands
    app.add_handler(
        CommandHandler("start", start)
    )

    # Messages
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    print("Telegram bot running...")

    app.run_polling()


# =========================================
# RUN
# =========================================

if __name__ == "__main__":

    main()