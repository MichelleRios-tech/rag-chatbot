# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system for querying course materials. It's a full-stack application with:
- **Backend**: FastAPI server with ChromaDB vector storage
- **Frontend**: Vanilla JavaScript SPA
- **AI**: Anthropic Claude with tool-based search
- **Embeddings**: sentence-transformers for semantic search

## Common Commands

### Running the Application

**Quick start:**
```bash
chmod +x run.sh
./run.sh
```

**Manual start (from project root):**
```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

**Install dependencies:**
```bash
uv sync
```

**Access points:**
- Web UI: http://localhost:8000
- API docs: http://localhost:8000/docs

### Windows Users
Use Git Bash to run all commands.

## Architecture

### Core RAG Pipeline

The RAG system follows this flow:

1. **Document Processing** (`document_processor.py`)
   - Parses course documents with structured format (title, instructor, lessons)
   - Chunks text into overlapping segments (default: 800 chars, 100 overlap)
   - Expected format: Course metadata at top, then "Lesson N: Title" markers

2. **Vector Storage** (`vector_store.py`)
   - **Two ChromaDB collections**:
     - `course_catalog`: Course metadata for semantic course name resolution
     - `course_content`: Actual lesson content chunks
   - Search flow: Resolves fuzzy course names → Filters by course/lesson → Returns ranked results

3. **AI Generation** (`ai_generator.py`)
   - Uses Claude with tool calling (not traditional RAG context injection)
   - Single system prompt with strict response guidelines
   - Temperature: 0, Max tokens: 800

4. **Tool-Based Search** (`search_tools.py`)
   - Claude decides when to call `search_course_content` tool
   - Tool parameters: `query` (required), `course_name` (optional), `lesson_number` (optional)
   - Sources tracked for frontend display

5. **Session Management** (`session_manager.py`)
   - Maintains conversation history (default: last 2 exchanges)
   - History injected into Claude's system prompt for context

### Key Design Patterns

**Tool-Based RAG vs Traditional RAG**
- This system does NOT inject retrieved context directly into prompts
- Instead, Claude uses the `search_course_content` tool when needed
- The tool returns formatted results that Claude synthesizes

**Course Name Resolution**
- Uses semantic search on `course_catalog` collection
- Enables fuzzy matching (e.g., "MCP" matches "Introduction to MCP")
- Vector search happens in `_resolve_course_name()` before content search

**Document Structure**
- Course files in `docs/` follow strict format:
  ```
  Course Title: [name]
  Course Link: [url]
  Course Instructor: [name]

  Lesson 0: [title]
  Lesson Link: [url]
  [content]
  ```

### Component Interactions

```
User Query → FastAPI → RAGSystem.query()
                         ↓
              AI Generator (with tools)
                         ↓
              (Claude decides to search)
                         ↓
              CourseSearchTool.execute()
                         ↓
              VectorStore.search()
                         ↓
              AI Generator (synthesizes)
                         ↓
              Response + Sources
```

## Configuration

All settings in `backend/config.py`:
- `ANTHROPIC_API_KEY`: Required env var
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
- `CHUNK_SIZE`: 800
- `CHUNK_OVERLAP`: 100
- `MAX_RESULTS`: 5
- `MAX_HISTORY`: 2 conversation exchanges

## Data Models

**Course hierarchy** (`models.py`):
- `Course`: title (unique ID), course_link, instructor, lessons[]
- `Lesson`: lesson_number, title, lesson_link
- `CourseChunk`: content, course_title, lesson_number, chunk_index

## Important Implementation Details

1. **ChromaDB IDs**: Course title is used as unique identifier in `course_catalog`
2. **Chunk Context**: First chunk of each lesson gets "Lesson N content:" prefix
3. **JSON Serialization**: Lesson metadata stored as JSON string in `lessons_json` field
4. **Duplicate Prevention**: `add_course_folder()` checks existing titles before re-indexing
5. **Startup Loading**: FastAPI startup event auto-loads `docs/` folder
6. **Session IDs**: Auto-generated as "session_{counter}" if not provided
7. **CORS**: Configured for development with allow_origins=["*"]

## Testing Considerations

When adding tests:
- Mock ChromaDB client to avoid filesystem dependencies
- Test document parser with malformed course files
- Verify chunk overlap logic with edge cases
- Test course name resolution with partial matches
- Validate tool execution through ToolManager
- always use uv to run the server never use pip directly
- make sure to use uv to manage all dependencies
- use uv to run python files