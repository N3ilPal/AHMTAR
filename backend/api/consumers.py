import cv2
import numpy as np
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
from .processor import process_frame  # Import your function

class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        print("WebSocket Connected")

    async def disconnect(self, close_code):
        print("WebSocket Disconnected")

    async def receive(self, text_data):
        try:
            # Decode base64 image
            image_data = base64.b64decode(text_data)
            np_arr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # ✅ Apply Mediapipe processing
            processed_frame = self.process_frame(frame)

            # Encode the processed frame
            _, buffer = cv2.imencode('.jpg', processed_frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')

            # Send processed frame back to frontend
            await self.send(text_data=encoded_frame)

        except Exception as e:
            print(f"Error processing frame: {e}")