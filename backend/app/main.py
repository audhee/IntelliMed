import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import engine, SessionLocal
from app.routes import auth, reports, analytics
from app.services.normalization import seed_master_biomarkers

app = FastAPI(
    title=settings.APP_NAME,
    description="A production-grade, highly scalable asynchronous API providing clinical OCR normalization and safe longitudinal intelligence.",
    version="1.0.0"
)

# ----------------------------------------------------
# CORS Middleware Setup (React Native Expo support)
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits all local Expo emulator ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Static Local Files Fallback Directory Routing
# ----------------------------------------------------
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))
os.makedirs(os.path.join(static_dir, 'uploads'), exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ----------------------------------------------------
# Startup Event: Seed Master Biomarkers
# ----------------------------------------------------
@app.on_event("startup")
def on_startup():
    db = SessionLocal()
    try:
        print("Database startup: Seeding canonical master biomarkers...")
        seed_master_biomarkers(db)
        print("Canonical master biomarkers successfully seeded!")
    except Exception as e:
        print(f"Error seeding master biomarkers on startup: {e}")
    finally:
        db.close()

# ----------------------------------------------------
# Endpoint Router Registrations
# ----------------------------------------------------
app.include_router(auth.router)
app.include_router(reports.router)
app.include_router(analytics.router)

# Basic health status endpoint
@app.get("/", tags=["Health Index"])
def health_index():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "worker_broker": "connected"
    }
