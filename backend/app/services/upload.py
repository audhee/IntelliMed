import os
import shutil
import uuid
import cloudinary
import cloudinary.uploader
from fastapi import UploadFile
from app.config import settings

# Initialize Cloudinary if credentials are set
has_cloudinary = all([
    settings.CLOUDINARY_CLOUD_NAME,
    settings.CLOUDINARY_API_KEY,
    settings.CLOUDINARY_API_SECRET
])

if has_cloudinary:
    cloudinary.config(
        cloud_name=settings.CLOUDINARY_CLOUD_NAME,
        api_key=settings.CLOUDINARY_API_KEY,
        api_secret=settings.CLOUDINARY_API_SECRET,
        secure=True
    )

def upload_file_to_storage(file: UploadFile) -> str:
    """Uploads file to Cloudinary. Falls back to local directory if credentials are empty."""
    if has_cloudinary:
        try:
            # Read file data to upload
            result = cloudinary.uploader.upload(
                file.file,
                resource_type="auto",
                folder="intellimed_reports"
            )
            return result.get("secure_url")
        except Exception as e:
            print(f"Cloudinary upload failed, falling back to local: {e}")
    
    # Local Storage Fallback
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'uploads'))
    os.makedirs(local_dir, exist_ok=True)
    
    # Generate unique filename to avoid collision
    unique_id = str(uuid.uuid4())
    file_ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    safe_filename = f"{unique_id}{file_ext}"
    
    dest_path = os.path.join(local_dir, safe_filename)
    
    # Reset file pointer and save locally
    file.file.seek(0)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Return local static server URL
    return f"/static/uploads/{safe_filename}"
