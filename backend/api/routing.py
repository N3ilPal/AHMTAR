from django.urls import re_path
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from api.consumers import VideoConsumer  # ✅ Make sure this is correctly imported

websocket_urlpatterns = [
    re_path(r"ws/video_stream/$", VideoConsumer.as_asgi()),  # ✅ Correct WebSocket route
]

application = ProtocolTypeRouter({
    "websocket": AuthMiddlewareStack(
        URLRouter(websocket_urlpatterns)
    ),
})

from django.urls import re_path
