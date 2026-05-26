from telegram.ext import Updater, MessageHandler, Filters

from graph import app_graph

# =====================================
# BOT TOKEN
# =====================================

TOKEN = "8957560769:AAHDHPgJ290Ht1RW_OMepf2sHgrsA9I75vY"

# =====================================
# MESSAGE HANDLER
# =====================================

def reply(update, context):

    user_message = update.message.text

    # Get chatbot response
    result = app_graph.invoke({
        "question": user_message
    })

    bot_reply = result["answer"]

    # Send reply
    update.message.reply_text(bot_reply)

# =====================================
# START BOT
# =====================================

updater = Updater(TOKEN, use_context=True)

dispatcher = updater.dispatcher

dispatcher.add_handler(
    MessageHandler(Filters.text, reply)
)

print("Telegram bot running...")

updater.start_polling()

updater.idle()