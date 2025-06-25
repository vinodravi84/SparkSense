import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import VideoFeed from "./components/VideoFeed";

export default function App() {
  const [uploaded, setUploaded] = useState(false);

  return (
    <div className="container">
      <h1>Vehicle Speed Tracker</h1>
      {!uploaded && <UploadForm onUploaded={() => setUploaded(true)} />}
      {uploaded && <VideoFeed />}
    </div>
  );
}
