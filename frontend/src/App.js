import React, { useState } from "react";
import "./App.css";

function App() {

  // =====================================
  // STATES
  // =====================================

  const [question, setQuestion] = useState("");

  const [messages, setMessages] = useState([]);

  const [loading, setLoading] = useState(false);

  const [file, setFile] = useState(null);

  const [uploadMessage, setUploadMessage] = useState("");

  // =====================================
  // SEND MESSAGE
  // =====================================

  const sendMessage = async () => {

  if (!question.trim()) return;

  const userQuestion = question;

  setQuestion("");

  setLoading(true);

  try {

    // =================================
    // ADD USER MESSAGE
    // =================================

    setMessages((prev) => [
      ...prev,
      {
        type: "user",
        content: userQuestion
      }
    ]);

    // =================================
    // API CALL
    // =================================

    const response = await fetch(
      "http://127.0.0.1:8000/chat",
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json"
        },

        body: JSON.stringify({
          message: userQuestion
        })
      }
    );

    const data = await response.json();

    // =================================
    // ADD BOTH RESPONSES
    // =================================

    setMessages((prev) => [
      ...prev,

        {
          type: "rag",
          content: data.rag_response
        },

        {
          type: "ft",
          content: data.finetuned_response
        }
      ]);

    } catch (error) {

      console.log(error);

      setMessages((prev) => [
        ...prev,
        {
          type: "error",
          content: "Server error"
        }
      ]);
    }

    setLoading(false);
  };

  // =====================================
  // ENTER KEY
  // =====================================

  const handleKeyDown = (e) => {

    if (e.key === "Enter") {

      sendMessage();
    }
  };

  const uploadFile = async () => {

  if (!file) return;

  const formData = new FormData();

  formData.append("file", file);

  try {

    const response = await fetch(
      "http://127.0.0.1:8000/upload",
      {
        method: "POST",
        body: formData
      }
    );

    const data = await response.json();

    setUploadMessage(data.message);

  } catch (error) {

    console.log(error);

    setUploadMessage("Upload failed");
  }
  };
  // =====================================
  // UI
  // =====================================

  return (

    <div className="app">

      {/* ============================ */}
      {/* HEADER */}
      {/* ============================ */}

      <div className="header">

        <h1>SSN AI Assistant</h1>

        <p>
          Compare RAG vs Fine-Tuned TinyLlama
        </p>

      </div>
      {/* ============================ */}
      {/* FILE UPLOAD */}
      {/* ============================ */}

      <div className="upload-section">

        <h2>Upload Knowledge Base</h2>

        <input
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
        />

        <button onClick={uploadFile}>
          Upload
        </button>

        {
          uploadMessage && (
            <p>{uploadMessage}</p>
          )
        }

      </div>
      {/* ============================ */}
      {/* CHAT AREA */}
      {/* ============================ */}

      <div className="chat-container">

        {
          messages.map((msg, index) => (

            <div key={index}>

              {/* USER MESSAGE */}

              {
                msg.type === "user" && (

                  <div className="user-box">

                    <div className="label">
                      YOU
                    </div>

                    <div className="message">
                      {msg.content}
                    </div>

                  </div>
                )
              }

              {/* RAG RESPONSE */}

              {
                msg.type === "rag" && (

                  <div className="rag-box">

                    <div className="label">
                      RAG MODEL
                    </div>

                    <div className="message">
                      {msg.content}
                    </div>

                  </div>
                )
              }

              {/* FINETUNED MODEL */}

              {
                msg.type === "ft" && (

                  <div className="ft-box">

                    <div className="label">
                      FINETUNED TINYLLAMA
                    </div>

                    <div className="message">
                      {msg.content}
                    </div>

                  </div>
                )
              }

              {/* ERROR */}

              {
                msg.type === "error" && (

                  <div className="error-box">

                    {msg.content}

                  </div>
                )
              }

            </div>
          ))
        }

        {/* LOADING */}

        {
          loading && (

            <div className="loading">

              Thinking...

            </div>
          )
        }

      </div>

      {/* ============================ */}
      {/* INPUT AREA */}
      {/* ============================ */}

      <div className="input-container">

        <input
          type="text"
          placeholder="Ask something about SSN College..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
        />

        <button onClick={sendMessage}>

          Send

        </button>

      </div>

    </div>
  );
}

export default App;