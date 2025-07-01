import React, { useState } from "react";
import axios from "axios";

export default function UploadForm({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await axios.post(
        "http://localhost:8000/upload",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      // Backend returns { stream_url: "/video_feed" }
      const streamUrl = `http://localhost:8000${res.data.stream_url}`;
      onUploaded(streamUrl);
    } catch (err) {
      console.error("Upload failed:", err);
      alert("Upload failed. Check console for details.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-form">
      <h2>📤 Upload Video</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files[0])}
          disabled={uploading}
        />
        <button type="submit" disabled={uploading || !file}>
          {uploading ? "Uploading…" : "Upload & Stream"}
        </button>
      </form>
    </div>
  );
}
