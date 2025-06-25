// components/VideoFeed.js
import React, { useRef, useEffect } from "react";

export default function VideoFeed() {
  const canvasRef = useRef();

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // open WS to our new /ws/stream endpoint
    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/stream`);
    ws.binaryType = "arraybuffer";

    ws.onmessage = async (evt) => {
      // each message is a JPEG frame
      const blob = new Blob([evt.data], { type: "image/jpeg" });
      const bitmap = await createImageBitmap(blob);
      // draw full frame scaled to canvas
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    };

    ws.onerror = (e) => console.error("WebSocket error", e);
    ws.onclose = () => console.log("Stream closed");

    return () => {
      ws.close();
    };
  }, []);

  return (
    <div style={{ background: "#000", padding: 10, textAlign: "center" }}>
      <h2 style={{ color: "#00ff90" }}>Live Detection Feed</h2>
      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        style={{
          width: "90%",
          border: "2px solid #00ff90",
          borderRadius: 10,
        }}
      />
    </div>
  );
}
