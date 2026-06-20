from io import BytesIO

from locust import HttpUser, between, task
from PIL import Image


def sample_image() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (800, 600), color=(240, 240, 235)).save(buffer, format="JPEG")
    return buffer.getvalue()


class AlbumUploadUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self) -> None:
        response = self.client.post("/api/v1/albums", json={"name": "Load Test", "description": None})
        response.raise_for_status()
        self.album_id = response.json()["id"]

    @task
    def upload_batch(self) -> None:
        payload = sample_image()
        files = [("files", ("page.jpg", payload, "image/jpeg")) for _ in range(5)]
        self.client.post(f"/api/v1/albums/{self.album_id}/pages/upload", files=files)

