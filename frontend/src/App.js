// src/App.js
import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import VideoFeed from "./components/VideoFeed";
import "./assets/index.css";

export default function App() {
  const [streamUrl, setStreamUrl] = useState("");

  const handleNewUpload = () => {
    setStreamUrl("");
  };

  return (
    <div className="container">
      <h1>Vehicle Speed Tracker</h1>

      {!streamUrl ? (
        <UploadForm
          onUploaded={(url) => {
            // Ensure absolute URL (adjust host/port if needed)
            const fullUrl =
              url.startsWith("http") ? url : `http://localhost:8000${url}`;
            setStreamUrl(fullUrl);
          }}
        />
      ) : (
        <>
          <button
            className="new-upload-btn"
            onClick={handleNewUpload}
            style={{
              marginBottom: "16px",
              background: "#00ff90",
              border: "none",
              padding: "8px 16px",
              cursor: "pointer",
              borderRadius: "4px",
              fontFamily: "monospace",
            }}
          >
            🔄 Upload New Video
          </button>
          <VideoFeed streamUrl={streamUrl} />
        </>
      )}
    </div>
);
}
