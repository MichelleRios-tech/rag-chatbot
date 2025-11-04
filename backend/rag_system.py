from typing import List, Tuple, Optional, Dict
import os
from document_processor import DocumentProcessor
from vector_store import VectorStore
from ai_generator import AIGenerator
from session_manager import SessionManager
from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool
from models import Course, Lesson, CourseChunk
from llm_provider import create_provider, LLMProvider
from provider_registry import ProviderRegistry

class RAGSystem:
    """Main orchestrator for the Retrieval-Augmented Generation system"""

    def __init__(self, config):
        self.config = config

        # Initialize core components
        self.document_processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.vector_store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
        self.session_manager = SessionManager(config.MAX_HISTORY)

        # Initialize search tools
        self.tool_manager = ToolManager()
        self.search_tool = CourseSearchTool(self.vector_store)
        self.tool_manager.register_tool(self.search_tool)
        self.outline_tool = CourseOutlineTool(self.vector_store)
        self.tool_manager.register_tool(self.outline_tool)

        # Check if dynamic model selection is enabled
        if config.MODEL_SELECTION_MODE == "dynamic":
            # Dynamic mode: Initialize provider registry for runtime switching
            self.provider_registry = ProviderRegistry(config)
            self.provider_cache: Dict[str, LLMProvider] = {}  # Cache providers by model_id
            self.default_provider = None  # Will be created on first query
            print(f"[RAG System] Dynamic model selection enabled")
        else:
            # Static mode: Single provider selected at startup
            provider = self._select_provider(config)
            self.ai_generator = AIGenerator(provider)
            self.provider_registry = None
            self.provider_cache = None
            self.default_provider = provider
            provider_name = provider.get_provider_name()
            print(f"[RAG System] Static mode - Using LLM provider: {provider_name}")

    def _select_provider(self, config):
        """
        Select and create an LLM provider based on configuration.

        Priority (auto mode): Anthropic → Gemini → LM Studio
        1. If LLM_PROVIDER is explicitly set, use that
        2. If "auto", use fallback chain based on API key availability
        3. Validate the selected provider can connect

        Returns:
            LLMProvider instance

        Raises:
            RuntimeError: If no provider is available or working
        """
        provider_type = config.LLM_PROVIDER.lower()

        # Auto-detect provider if set to "auto"
        if provider_type == "auto":
            # Priority: Anthropic → Gemini → LM Studio
            if config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY.strip():
                provider_type = "anthropic"
                print("[RAG System] Auto-detected: Using Anthropic (API key found)")
            elif config.GEMINI_API_KEY and config.GEMINI_API_KEY.strip():
                provider_type = "gemini"
                print("[RAG System] Auto-detected: Using Gemini (API key found)")
            else:
                provider_type = "lmstudio"
                print("[RAG System] Auto-detected: Using LM Studio (no cloud API keys)")

        # Create the provider
        try:
            if provider_type == "anthropic":
                if not config.ANTHROPIC_API_KEY or not config.ANTHROPIC_API_KEY.strip():
                    raise RuntimeError("Anthropic API key is required but not set")
                provider = create_provider(
                    provider_type="anthropic",
                    api_key=config.ANTHROPIC_API_KEY,
                    model=config.ANTHROPIC_MODEL
                )
            elif provider_type == "gemini":
                if not config.GEMINI_API_KEY or not config.GEMINI_API_KEY.strip():
                    raise RuntimeError("Gemini API key is required but not set")
                provider = create_provider(
                    provider_type="gemini",
                    api_key=config.GEMINI_API_KEY,
                    model=config.GEMINI_MODEL
                )
            elif provider_type == "lmstudio":
                provider = create_provider(
                    provider_type="lmstudio",
                    api_key=config.LMSTUDIO_API_KEY,
                    model=config.LMSTUDIO_MODEL,
                    base_url=config.LMSTUDIO_BASE_URL
                )
            else:
                raise ValueError(f"Unknown provider type: {provider_type}")

            # Test the provider connection
            print(f"[RAG System] Testing {provider.get_provider_name()} connection...")
            if not provider.test_connection():
                # If primary provider fails and we're in auto mode, try fallback chain
                if config.LLM_PROVIDER.lower() == "auto":
                    print(f"[RAG System] {provider.get_provider_name()} connection failed, trying fallback...")

                    # Try Gemini if Anthropic failed
                    if provider_type == "anthropic" and config.GEMINI_API_KEY and config.GEMINI_API_KEY.strip():
                        print("[RAG System] Falling back to Gemini...")
                        provider = create_provider(
                            provider_type="gemini",
                            api_key=config.GEMINI_API_KEY,
                            model=config.GEMINI_MODEL
                        )
                        if provider.test_connection():
                            print(f"[RAG System] Successfully connected to {provider.get_provider_name()}")
                            return provider

                    # Try LM Studio as final fallback
                    print("[RAG System] Falling back to LM Studio...")
                    provider = create_provider(
                        provider_type="lmstudio",
                        api_key=config.LMSTUDIO_API_KEY,
                        model=config.LMSTUDIO_MODEL,
                        base_url=config.LMSTUDIO_BASE_URL
                    )
                    if not provider.test_connection():
                        raise RuntimeError("All provider connections failed (Anthropic, Gemini, LM Studio)")
                else:
                    raise RuntimeError(f"{provider.get_provider_name()} connection test failed")

            print(f"[RAG System] Successfully connected to {provider.get_provider_name()}")
            return provider

        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM provider: {str(e)}")

    def _get_or_create_provider(self, model_id: Optional[str] = None) -> LLMProvider:
        """
        Get or create a provider for the specified model.

        Args:
            model_id: Optional model ID. If None, use default.

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If model_id is invalid or not available
        """
        # If not in dynamic mode, return the default provider
        if not self.provider_registry:
            return self.default_provider

        # If no model_id specified, use default
        if not model_id:
            if not self.default_provider:
                default_model_id = self.provider_registry.get_default_model_id()
                self.default_provider = self.provider_registry.create_provider_for_model(default_model_id)
            return self.default_provider

        # Check cache first
        if model_id in self.provider_cache:
            return self.provider_cache[model_id]

        # Create new provider for this model
        provider = self.provider_registry.create_provider_for_model(model_id)
        if not provider:
            raise ValueError(f"Model '{model_id}' is not available")

        # Cache it
        self.provider_cache[model_id] = provider
        print(f"[RAG System] Created provider for model: {model_id}")

        return provider

    def _detect_and_compress_on_model_switch(
        self,
        session_id: str,
        new_model_id: str
    ) -> bool:
        """
        Detect if model changed and compress conversation if needed.

        Args:
            session_id: Session ID
            new_model_id: The new model being used

        Returns:
            True if model changed and compression occurred, False otherwise
        """
        current_model = self.session_manager.get_current_model(session_id)

        if current_model and current_model != new_model_id:
            # Model changed! Compress the conversation
            print(f"[RAG System] Model switch detected: {current_model} → {new_model_id}")
            print(f"[RAG System] Compressing conversation history...")
            self.session_manager.compress_conversation(session_id, current_model)

            # Update to new model
            self.session_manager.set_current_model(session_id, new_model_id)
            return True

        # First time setting model for this session
        if not current_model:
            self.session_manager.set_current_model(session_id, new_model_id)

        return False

    def add_course_document(self, file_path: str) -> Tuple[Course, int]:
        """
        Add a single course document to the knowledge base.
        
        Args:
            file_path: Path to the course document
            
        Returns:
            Tuple of (Course object, number of chunks created)
        """
        try:
            # Process the document
            course, course_chunks = self.document_processor.process_course_document(file_path)
            
            # Add course metadata to vector store for semantic search
            self.vector_store.add_course_metadata(course)
            
            # Add course content chunks to vector store
            self.vector_store.add_course_content(course_chunks)
            
            return course, len(course_chunks)
        except Exception as e:
            print(f"Error processing course document {file_path}: {e}")
            return None, 0
    
    def add_course_folder(self, folder_path: str, clear_existing: bool = False) -> Tuple[int, int]:
        """
        Add all course documents from a folder.
        
        Args:
            folder_path: Path to folder containing course documents
            clear_existing: Whether to clear existing data first
            
        Returns:
            Tuple of (total courses added, total chunks created)
        """
        total_courses = 0
        total_chunks = 0
        
        # Clear existing data if requested
        if clear_existing:
            print("Clearing existing data for fresh rebuild...")
            self.vector_store.clear_all_data()
        
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist")
            return 0, 0
        
        # Get existing course titles to avoid re-processing
        existing_course_titles = set(self.vector_store.get_existing_course_titles())
        
        # Process each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(('.pdf', '.docx', '.txt')):
                try:
                    # Check if this course might already exist
                    # We'll process the document to get the course ID, but only add if new
                    course, course_chunks = self.document_processor.process_course_document(file_path)
                    
                    if course and course.title not in existing_course_titles:
                        # This is a new course - add it to the vector store
                        self.vector_store.add_course_metadata(course)
                        self.vector_store.add_course_content(course_chunks)
                        total_courses += 1
                        total_chunks += len(course_chunks)
                        print(f"Added new course: {course.title} ({len(course_chunks)} chunks)")
                        existing_course_titles.add(course.title)
                    elif course:
                        print(f"Course already exists: {course.title} - skipping")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
        
        return total_courses, total_chunks
    
    def query(
        self,
        query: str,
        session_id: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Process a user query using the RAG system with tool-based search.

        Args:
            query: User's question
            session_id: Optional session ID for conversation context
            model_id: Optional model ID for dynamic provider selection

        Returns:
            Tuple of (response, sources list - empty for tool-based approach)
        """
        # Get or create provider for the specified model
        try:
            provider = self._get_or_create_provider(model_id)
        except ValueError as e:
            return f"Error: {str(e)}", []

        # Determine the actual model ID being used
        actual_model_id = model_id or self.provider_registry.get_default_model_id() if self.provider_registry else "default"

        # Detect model switching and compress conversation if needed
        if session_id and self.provider_registry:
            self._detect_and_compress_on_model_switch(session_id, actual_model_id)

        # Create AI generator with the selected provider (dynamic mode)
        if self.provider_registry:
            ai_generator = AIGenerator(provider)
        else:
            # Static mode: use existing ai_generator
            ai_generator = self.ai_generator

        # Create prompt for the AI with clear instructions
        prompt = f"""Answer this question about course materials: {query}"""

        # Get conversation history if session exists
        history = None
        if session_id:
            history = self.session_manager.get_conversation_history(session_id)

        # Generate response using AI with tools
        response = ai_generator.generate_response(
            query=prompt,
            conversation_history=history,
            tools=self.tool_manager.get_tool_definitions(),
            tool_manager=self.tool_manager
        )

        # Get sources from the search tool
        sources = self.tool_manager.get_last_sources()

        # Reset sources after retrieving them
        self.tool_manager.reset_sources()

        # Update conversation history
        if session_id:
            self.session_manager.add_exchange(session_id, query, response)

        # Return response with sources from tool searches
        return response, sources
    
    def get_course_analytics(self) -> Dict:
        """Get analytics about the course catalog"""
        return {
            "total_courses": self.vector_store.get_course_count(),
            "course_titles": self.vector_store.get_existing_course_titles()
        }