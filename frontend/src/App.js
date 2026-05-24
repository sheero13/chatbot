import React, { useState } from "react";
import axios from "axios";

import "./App.css";

function App() {

  // Chat states
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([]);
  const [loading, setLoading] = useState(false);

  // File upload state
  const [selectedFile, setSelectedFile] = useState(null);

  // =========================
  // SEND MESSAGE
  // =========================

  const sendMessage = async () => {

    if (!message.trim()) return;

    const userMessage = {
      sender: "user",
      text: message
    };

    setChat((prev) => [...prev, userMessage]);

    const currentMessage = message;

    setMessage("");
    setLoading(true);

    try {

      const response = await axios.post(
        "http://127.0.0.1:8000/chat",
        {
          message: currentMessage
        }
      );

      const botMessage = {
        sender: "bot",
        text: response.data.response
      };

      setChat((prev) => [...prev, botMessage]);

    } catch (error) {

      const errorMessage = {
        sender: "bot",
        text: "Error connecting to backend."
      };

      setChat((prev) => [...prev, errorMessage]);
    }

    setLoading(false);
  };

  // =========================
  // ENTER KEY SUPPORT
  // =========================

  const handleKeyPress = (e) => {

    if (e.key === "Enter") {
      sendMessage();
    }
  };

  // =========================
  // FILE UPLOAD
  // =========================

  const handleUpload = async () => {

    if (!selectedFile) {

      alert("Please select a file first");
      return;
    }

    const formData = new FormData();

    formData.append("file", selectedFile);

    try {

      const response = await axios.post(
        "http://127.0.0.1:8000/upload",
        formData
      );

      alert(response.data.message);

      setSelectedFile(null);

    } catch (error) {

      alert("Upload failed");
    }
  };

  // =========================
  // UI
  // =========================

  return (

    <div className="app">

      {/* HEADER */}
      <div className="header">
        <div className="header-title">
          SSN College AI Assistant
        </div>

        <div className="header-subtitle">
          Intelligent RAG-based Student Support System
        </div>
      </div>

      {/* CHAT AREA */}
      <div className="chat-container">

        {chat.map((msg, index) => (

          <div
            key={index}
            className={
              msg.sender === "user"
                ? "message-row user-row"
                : "message-row bot-row"
            }
          >

            <div
              className={
                msg.sender === "user"
                  ? "message user-message"
                  : "message bot-message"
              }
            >

              {msg.text}

            </div>

          </div>

        ))}

        {/* LOADING */}
        {loading && (

          <div className="message-row bot-row">

            <div className="message bot-message typing">

              Thinking...

            </div>

          </div>

        )}

      </div>

      {/* INPUT AREA */}
      <div className="input-container">

        <input
          type="text"
          placeholder="Ask something about SSN College..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyPress}
        />

        <button
          className="send-button"
          onClick={sendMessage}
        >
          Send
        </button>

      </div>

      {/* FLOATING UPLOAD BUTTON */}
      <div className="floating-upload">

        <label className="upload-circle">

          +

          <input
            type="file"
            hidden
            onChange={(e) =>
              setSelectedFile(e.target.files[0])
            }
          />

        </label>

        {selectedFile && (

          <div className="upload-popup">

            <div className="file-name">
              {selectedFile.name}
            </div>

            <button onClick={handleUpload}>
              Upload
            </button>

          </div>

        )}

      </div>

    </div>
  );
}

export default App;