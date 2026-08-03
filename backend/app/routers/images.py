import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

router = APIRouter(prefix="/uploads", tags=["images"])


@router.get("/{filename}")
def serve_upload(filename: str):
    """
    Serve an uploaded image or saved mask by filename.
    Used by the Streamlit front-end to display original images and
    segmentation masks stored in the uploads directory.
    """
    # Prevent path traversal attacks
    safe_name = os.path.basename(filename)
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File '{safe_name}' not found.")

    return FileResponse(file_path)
