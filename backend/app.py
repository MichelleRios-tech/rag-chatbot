import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os

from config import config
from rag_system import RAGSystem

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None
    model_id: Optional[str] = None  # NEW: For dynamic model selection

class SourceCitation(BaseModel):
    """Model for a single source citation with optional link"""
    display_text: str
    lesson_link: Optional[str] = None
    course_title: str
    lesson_number: Optional[int] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]

class HealthStatus(BaseModel):
    """Response model for health check"""
    status: str
    llm_provider: str
    vector_store: str
    message: str

# NEW: Models endpoint response models
class ModelInfo(BaseModel):
    """Information about an available model"""
    model_id: str
    display_name: str
    supports_tools: bool
    context_window: int
    is_default: bool

class ProviderInfo(BaseModel):
    """Information about an available provider"""
    name: str
    display_name: str
    models: List[ModelInfo]
    is_available: bool

class ModelsResponse(BaseModel):
    """Response model for available models"""
    providers: List[ProviderInfo]
    default_model_id: str

# API Endpoints

@app.get("/api/models", response_model=ModelsResponse)
async def get_available_models():
    """Get all available models from all providers"""
    try:
        # Check if dynamic mode is enabled
        if not rag_system.provider_registry:
            # Static mode: return current provider's model only
            return ModelsResponse(
                providers=[
                    ProviderInfo(
                        name="current",
                        display_name=rag_system.default_provider.get_provider_name(),
                        models=[ModelInfo(
                            model_id="default",
                            display_name="Current Model",
                            supports_tools=True,
                            context_window=100000,
                            is_default=True
                        )],
                        is_available=True
                    )
                ],
                default_model_id="default"
            )

        # Dynamic mode: fetch from provider registry
        providers_info = rag_system.provider_registry.get_all_available_models()
        default_model = rag_system.provider_registry.get_default_model_id()

        # Convert to response format
        providers_list = []
        for provider_info in providers_info:
            models_list = [
                ModelInfo(
                    model_id=model.model_id,
                    display_name=model.display_name,
                    supports_tools=model.supports_tools,
                    context_window=model.context_window,
                    is_default=model.is_default
                )
                for model in provider_info.models
            ]

            providers_list.append(ProviderInfo(
                name=provider_info.name,
                display_name=provider_info.display_name,
                models=models_list,
                is_available=provider_info.is_available
            ))

        return ModelsResponse(
            providers=providers_list,
            default_model_id=default_model
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = rag_system.session_manager.create_session()

        # Validate model_id if provided (only in dynamic mode)
        if request.model_id and rag_system.provider_registry:
            # Check if model exists
            model_info = rag_system.provider_registry.find_model_info(request.model_id)
            if not model_info:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{request.model_id}' not found or not available"
                )

        # Process query using RAG system with optional model selection
        answer, sources = rag_system.query(
            query=request.query,
            session_id=session_id,
            model_id=request.model_id
        )

        return QueryResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health", response_model=HealthStatus)
async def health_check():
    """Check system health and component status"""
    try:
        # Check vector store
        vector_store_status = "ok" if rag_system.vector_store else "error"

        # Get provider name based on mode
        if rag_system.provider_registry:
            provider_name = "Dynamic (Multi-provider)"
        else:
            provider_name = rag_system.default_provider.get_provider_name()

        return HealthStatus(
            status="healthy",
            llm_provider=provider_name,
            vector_store=vector_store_status,
            message=f"System operational with {provider_name}"
        )
    except Exception as e:
        return HealthStatus(
            status="unhealthy",
            llm_provider="unknown",
            vector_store="unknown",
            message=f"System error: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup and verify system health"""
    print("\n" + "="*60)
    print("Starting Course Materials RAG System")
    print("="*60)

    # Display provider mode
    if rag_system.provider_registry:
        print(f"Mode: Dynamic model selection (runtime switching enabled)")
        available = rag_system.provider_registry.get_available_providers()
        providers = [name for name, status in available.items() if status]
        print(f"Available providers: {', '.join(providers)}")
    else:
        print(f"Mode: Static provider - {rag_system.default_provider.get_provider_name()}")

    # Load documents
    docs_path = "../docs"
    if os.path.exists(docs_path):
        print("\nLoading course documents...")
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            print(f"Loaded {courses} new courses with {chunks} chunks")
        except Exception as e:
            print(f"Warning: Error loading documents: {e}")

    print("\n" + "="*60)
    print("System ready!")
    print("="*60 + "\n")

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    
# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")