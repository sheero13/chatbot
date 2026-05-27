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
# NEW IMPORTS FOR FILE UPLOAD
# =========================================

import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from pypdf import PdfReader

from docx import Document as DocxDocument


# =========================================
# TELEGRAM BOT TOKEN
# =========================================

BOT_TOKEN = "8957560769:AAHDHPgJ290Ht1RW_OMepf2sHgrsA9I75vY"


# =========================================
# PATH SETUP
# =========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

VECTOR_DB_DIR = os.path.join(
    BASE_DIR,
    "vector_db"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================================
# EMBEDDINGS + VECTOR DB
# =========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)


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
# HANDLE TEXT MESSAGE
# =========================================

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

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
    # CHATBOT
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

    await update.message.reply_text(final_reply)


# =========================================
# HANDLE DOCUMENT UPLOAD
# =========================================

async def handle_document(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    try:

        telegram_file = update.message.document

        filename = telegram_file.file_name

        print(f"\nReceiving File: {filename}")

        # =================================
        # DOWNLOAD FILE
        # =================================

        file = await context.bot.get_file(
            telegram_file.file_id
        )

        save_path = os.path.join(
            UPLOAD_DIR,
            filename
        )

        await file.download_to_drive(save_path)

        print(f"Saved: {save_path}")

        # =================================
        # EXTRACT TEXT
        # =================================

        text = ""

        # TXT
        if filename.endswith(".txt"):

            with open(
                save_path,
                "r",
                encoding="utf-8"
            ) as f:

                text = f.read()

        # PDF
        elif filename.endswith(".pdf"):

            pdf = PdfReader(save_path)

            for page in pdf.pages:

                extracted = page.extract_text()

                if extracted:

                    text += extracted + "\n"

        # DOCX
        elif filename.endswith(".docx"):

            doc = DocxDocument(save_path)

            for para in doc.paragraphs:

                text += para.text + "\n"

        else:

            await update.message.reply_text(
                "Unsupported file type."
            )

            return

        # =================================
        # VALIDATE
        # =================================

        if len(text.strip()) < 20:

            await update.message.reply_text(
                "Document is empty."
            )

            return

        # =================================
        # FAQ CHUNKING
        # =================================

        chunks = text.split("\n\n")

        documents = [

            Document(page_content=chunk.strip())

            for chunk in chunks

            if chunk.strip()
        ]

        # =================================
        # ADD TO VECTOR DB
        # =================================

        db.add_documents(documents)

        print("Documents added to vector DB")

        await update.message.reply_text(
            f"{filename} uploaded successfully.\n"
            f"Chunks added: {len(documents)}"
        )

    except Exception as e:

        print("Upload Error:", e)

        await update.message.reply_text(
            "File upload failed."
        )


# =========================================
# MAIN
# =========================================

def main():

    app = ApplicationBuilder().token(
        BOT_TOKEN
    ).build()

    # Start command
    app.add_handler(
        CommandHandler("start", start)
    )

    # Text messages
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message
        )
    )

    # =====================================
    # DOCUMENT HANDLER
    # =====================================

    app.add_handler(
        MessageHandler(
            filters.Document.ALL,
            handle_document
        )
    )

    print("Telegram bot running...")

    app.run_polling()


# =========================================
# RUN
# =========================================

if __name__ == "__main__":

    main()