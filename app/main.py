"""Main FastAPI application."""

import os
import sys
from pathlib import Path

print("\n" + "="*70)
print("MAIN.PY STARTUP - LOADING .ENV FILE")
print("="*70)

# MUST come BEFORE any app imports
from dotenv import load_dotenv

# Load environment variables from .env file - explicit path
env_path = Path(__file__).parent.parent / ".env"
print(f"🔍 Step 1: Looking for .env at: {env_path}")
print(f"   File exists: {env_path.exists()}")

if env_path.exists():
    # DEBUG: Read the file to see what's actually in it
    print(f"\n🔍 Step 2: Reading .env file to verify content")
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Look for OPENAI_API_KEY line
    for line in content.split('\n'):
        if line.startswith('OPENAI_API_KEY'):
            print(f"   Found line: {line[:80]}...")
            if len(line) > 80:
                print(f"   (total length: {len(line)})")
    
    print(f"\n🔍 Step 3: Calling load_dotenv() with override=True")
    result = load_dotenv(dotenv_path=env_path, override=True)
    print(f"   load_dotenv() returned: {result}")
    
    print(f"\n🔍 Step 4: Reading OPENAI_API_KEY from environment after load_dotenv")
    api_key_from_file = os.getenv("OPENAI_API_KEY", "")
    print(f"   Length: {len(api_key_from_file)}")
    print(f"   Value (first 50 chars): {api_key_from_file[:50]}")
    
    if api_key_from_file and len(api_key_from_file) > 50:
        print(f"✓ REAL API KEY DETECTED! Length: {len(api_key_from_file)}")
        os.environ["OPENAI_API_KEY"] = api_key_from_file
        print(f"✓ Set os.environ['OPENAI_API_KEY']")
    else:
        print(f"❌ PLACEHOLDER OR SHORT KEY FOUND: '{api_key_from_file}'")
else:
    print(f"❌ .env file NOT found at {env_path}")

print("\n🔍 Step 5: Verifying os.environ has the key")
check_key = os.environ.get("OPENAI_API_KEY", "NOT SET")
print(f"   os.environ['OPENAI_API_KEY'] length: {len(check_key)}")
print(f"   First 50 chars: {check_key[:50]}")

print("\n" + "="*70)
print("NOW IMPORTING FastAPI AND APP MODULES")
print("="*70 + "\n")

# NOW import FastAPI and other app modules
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import logger
from app.api.routes import router

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    debug=settings.debug,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Verify OpenAI configuration
    from app.core.ai_config import verify_openai_config
    if verify_openai_config():
        logger.info("✓ OpenAI configuration verified")
    else:
        logger.warning("⚠️  OpenAI not properly configured - practice features may not work")
# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
