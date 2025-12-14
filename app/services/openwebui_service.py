import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenWebUIService:
    """
    Service for uploading images to Open WebUI's Files API.

    When configured, images are uploaded to Open WebUI's storage,
    making them accessible regardless of network topology.
    """

    def __init__(self):
        if not settings.OPENWEBUI_API_URL:
            raise ValueError("OPENWEBUI_API_URL not configured")
        if not settings.OPENWEBUI_API_KEY:
            raise ValueError("OPENWEBUI_API_KEY not configured")

        self.base_url = settings.OPENWEBUI_API_URL.rstrip("/")
        self.api_key = settings.OPENWEBUI_API_KEY

    async def upload_image(self, image_data: bytes, filename: str) -> str:
        """
        Upload image to Open WebUI Files API.

        Args:
            image_data: Image bytes
            filename: Filename with extension (e.g., "image.png")

        Returns:
            URL to access the uploaded file
        """
        url = f"{self.base_url}/api/v1/files/"

        # Determine content type from filename
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
        content_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "webp": "image/webp",
            "gif": "image/gif",
        }
        content_type = content_types.get(ext, "image/png")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

        # Multipart file upload
        files = {
            "file": (filename, image_data, content_type),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, files=files)
            response.raise_for_status()

            data = response.json()
            file_id = data.get("id")

            if not file_id:
                raise ValueError(f"No file ID in response: {data}")

            # Return URL to access the file
            file_url = f"{self.base_url}/api/v1/files/{file_id}/content"
            logger.info(f"Uploaded image to Open WebUI: {file_url}")

            return file_url


# Singleton instance (lazy initialization)
_openwebui_service: OpenWebUIService | None = None


def get_openwebui_service() -> OpenWebUIService:
    """Get Open WebUI service instance."""
    global _openwebui_service
    if _openwebui_service is None:
        _openwebui_service = OpenWebUIService()
    return _openwebui_service
