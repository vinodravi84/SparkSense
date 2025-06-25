// components/UploadForm.js
import React, { useState } from "react";
import axios from "axios";

export default function UploadForm({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [results, setResults] = useState({});

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await axios.post("http://localhost:8000/upload", formData);
      setVideoUrl("http://localhost:8000" + res.data.video_url);
      setResults(res.data.results);
      onUploaded();
    } catch (err) {
      alert("Upload failed");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-form">
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button type="submit" disabled={uploading}>
          {uploading ? "Uploading..." : "Upload & Process"}
        </button>
      </form>

      {videoUrl && (
        <div style={{ marginTop: "20px" }}>
          <video src={videoUrl} controls width="100%" />
        </div>
      )}

      {Object.keys(results).length > 0 && (
        <div style={{ marginTop: "20px" }}>
          <h3>Max Speeds</h3>
          <ul>
            {Object.entries(results).map(([id, speed]) => (
              <li key={id}>
                {id.toUpperCase()} → {speed} km/h
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
