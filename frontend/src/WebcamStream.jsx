import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import { io } from "socket.io-client";

const socket = io("ws://127.0.0.1:8000", { 
  transports: ["websocket", "polling"]
});


const WebcamStream = () => {
  const webcamRef = useRef(null);
  const [capturing, setCapturing] = useState(false);

  useEffect(() => {
    if (capturing) {
      const interval = setInterval(() => {
        if (webcamRef.current) {
          const imageSrc = webcamRef.current.getScreenshot();
          if (imageSrc) {
            console.log("Sending frame to backend:", imageSrc);
            socket.emit("video_frame", imageSrc);  // Send frame to backend
          }
        }
      }, 100); // Send frame every 100ms

      return () => clearInterval(interval);
    }
  }, [capturing]);

  return (
    <div>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width={640}
        height={480}
      />
      <button onClick={() => setCapturing(!capturing)}>
        {capturing ? "Stop Streaming" : "Start Streaming"}
      </button>
    </div>
  );
};

export default WebcamStream;