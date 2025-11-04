"""
Test script to verify LLM provider functionality
"""

import os
from config import config
from llm_provider import create_provider

print("="*60)
print("LLM Provider Test Script")
print("="*60)

print("\nConfiguration:")
print(f"  LLM_PROVIDER: {config.LLM_PROVIDER}")
print(f"  ANTHROPIC_API_KEY: {'SET' if config.ANTHROPIC_API_KEY else 'NOT SET'}")
print(f"  LMSTUDIO_BASE_URL: {config.LMSTUDIO_BASE_URL}")
print(f"  LMSTUDIO_MODEL: {config.LMSTUDIO_MODEL}")

print("\n" + "="*60)
print("Test 1: LM Studio Provider")
print("="*60)

try:
    lm_provider = create_provider(
        provider_type="lmstudio",
        api_key=config.LMSTUDIO_API_KEY,
        model=config.LMSTUDIO_MODEL,
        base_url=config.LMSTUDIO_BASE_URL
    )
    print(f"Provider created: {lm_provider.get_provider_name()}")

    # Test connection
    print("Testing connection...")
    if lm_provider.test_connection():
        print("SUCCESS: LM Studio is accessible!")

        # Try a simple generation
        print("\nTesting simple generation...")
        try:
            response, tool_calls, _ = lm_provider.generate(
                system_prompt="You are a helpful assistant.",
                messages=[{"role": "user", "content": "Say hello in 5 words or less"}],
                temperature=0,
                max_tokens=50
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Generation test failed: {e}")
    else:
        print("WARNING: LM Studio connection test failed")
        print("Make sure LM Studio is running at " + config.LMSTUDIO_BASE_URL)
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "="*60)
print("Test 2: Anthropic Provider (if API key available)")
print("="*60)

if config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY.strip():
    try:
        anthropic_provider = create_provider(
            provider_type="anthropic",
            api_key=config.ANTHROPIC_API_KEY,
            model=config.ANTHROPIC_MODEL
        )
        print(f"Provider created: {anthropic_provider.get_provider_name()}")

        # Test connection
        print("Testing connection...")
        if anthropic_provider.test_connection():
            print("SUCCESS: Anthropic API is accessible!")

            # Try a simple generation
            print("\nTesting simple generation...")
            try:
                response, tool_calls, _ = anthropic_provider.generate(
                    system_prompt="You are a helpful assistant.",
                    messages=[{"role": "user", "content": "Say hello in 5 words or less"}],
                    temperature=0,
                    max_tokens=50
                )
                print(f"Response: {response}")
            except Exception as e:
                print(f"Generation test failed: {e}")
        else:
            print("WARNING: Anthropic API connection test failed")
    except Exception as e:
        print(f"ERROR: {e}")
else:
    print("SKIPPED: No Anthropic API key set")
    print("To test Anthropic, set ANTHROPIC_API_KEY in .env file")

print("\n" + "="*60)
print("Test 3: Auto Provider Selection (via RAG System)")
print("="*60)

try:
    from rag_system import RAGSystem
    rag = RAGSystem(config)
    print(f"SUCCESS: RAG System initialized with {rag.provider_name}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "="*60)
print("Test 4: Gemini Provider (if API key available)")
print("="*60)

if config.GEMINI_API_KEY and config.GEMINI_API_KEY.strip():
    try:
        gemini_provider = create_provider(
            provider_type="gemini",
            api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL
        )
        print(f"Provider created: {gemini_provider.get_provider_name()}")

        # Test connection
        print("Testing connection...")
        if gemini_provider.test_connection():
            print("SUCCESS: Gemini API is accessible!")

            # Try a simple generation without tools
            print("\nTest 4a: Simple generation (no tools)...")
            try:
                response, tool_calls, _ = gemini_provider.generate(
                    system_prompt="You are a helpful assistant.",
                    messages=[{"role": "user", "content": "Say hello in 5 words or less"}],
                    temperature=0,
                    max_tokens=50
                )
                print(f"Response: {response}")
                print("SUCCESS: Simple generation works!")
            except Exception as e:
                print(f"ERROR in simple generation: {e}")
                import traceback
                traceback.print_exc()

            # Try generation with tools
            print("\nTest 4b: Generation with tools...")
            try:
                test_tools = [{
                    "name": "test_tool",
                    "description": "A test tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "A query"}
                        },
                        "required": ["query"]
                    }
                }]

                response, tool_calls, _ = gemini_provider.generate(
                    system_prompt="You are a helpful assistant.",
                    messages=[{"role": "user", "content": "Say hello in 5 words"}],
                    tools=test_tools,
                    temperature=0,
                    max_tokens=50
                )
                print(f"Response: {response}")
                print(f"Tool calls: {tool_calls}")
                print("SUCCESS: Tool-based generation works!")
            except Exception as e:
                print(f"ERROR in tool-based generation: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("WARNING: Gemini API connection test failed")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
else:
    print("SKIPPED: No Gemini API key set")
    print("To test Gemini, set GEMINI_API_KEY in .env file")

print("\n" + "="*60)
print("All tests complete!")
print("="*60)
