// src/components/VideoFeed.js
import React from "react";

export default function VideoFeed({ streamUrl }) {
  return (
    <div className="video-feed">
      <h2>🚘 Live Detection Stream</h2>
      <img
        src={streamUrl}
        alt="Live Vehicle Feed"
        style={{
          width: "100%",
          maxWidth: "800px",
          border: "3px solid #00ff90",
          borderRadius: "8px",
        }}
      />
    </div>
  );
}
