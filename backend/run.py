import uvicorn
import os
from api.api import app

if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Run the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=port)