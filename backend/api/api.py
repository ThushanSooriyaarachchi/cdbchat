from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# Import the get_final_result function from your existing code
from ai_agents.multi_agent import get_final_result  # Replace with the actual module name

# Create FastAPI app
app = FastAPI(
    title="Query Processing API",
    description="API for processing queries using an intelligent routing system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for query
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

# Response model
class QueryResponse(BaseModel):
    query: str
    result: str
    status: str

# Endpoint for query processing
@app.post("/process-query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Process the query using the existing function
        result = get_final_result(request.query)
        
        return QueryResponse(
            query=request.query,
            result=result,
            status="success"
        )
    except Exception as e:
        # Handle any errors during processing
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Optional: Add a simple documentation route
@app.get("/")
async def root():
    return {
        "message": "Welcome to Query Processing API",
        "endpoints": {
            "/process-query": "POST endpoint for processing queries",
            "/health": "Health check endpoint"
        }
    }