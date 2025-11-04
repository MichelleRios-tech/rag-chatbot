"""
Quick test for Gemini provider only
"""

import os
from config import config
from llm_provider import create_provider

print("="*60)
print("Gemini Provider Quick Test")
print("="*60)

if not config.GEMINI_API_KEY or not config.GEMINI_API_KEY.strip():
    print("ERROR: GEMINI_API_KEY not set in .env file")
    print("Please add GEMINI_API_KEY=your-key-here to .env")
    exit(1)

print(f"\nUsing model: {config.GEMINI_MODEL}")
print(f"API Key: {'SET' if config.GEMINI_API_KEY else 'NOT SET'}")

try:
    print("\n1. Creating Gemini provider...")
    provider = create_provider(
        provider_type="gemini",
        api_key=config.GEMINI_API_KEY,
        model=config.GEMINI_MODEL
    )
    print(f"   [OK] Provider created: {provider.get_provider_name()}")

    print("\n2. Testing connection...")
    if provider.test_connection():
        print("   ✓ Connection successful!")
    else:
        print("   ✗ Connection failed!")
        exit(1)

    print("\n3. Testing simple generation (no tools)...")
    try:
        response, tool_calls, _ = provider.generate(
            system_prompt="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Say 'Hello from Gemini!' in exactly those words"}],
            temperature=0,
            max_tokens=50
        )
        print(f"   Response: {response}")
        print(f"   Tool calls: {tool_calls}")
        print("   ✓ Simple generation works!")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n4. Testing generation with tools...")
    try:
        test_tools = [{
            "name": "get_weather",
            "description": "Get the weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            }
        }]

        response, tool_calls, _ = provider.generate(
            system_prompt="You are a helpful assistant with access to weather data.",
            messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
            tools=test_tools,
            temperature=0,
            max_tokens=100
        )
        print(f"   Response: {response}")
        print(f"   Tool calls: {tool_calls}")

        if tool_calls and len(tool_calls) > 0:
            print("   ✓ Tool calling works! Model requested tool use.")
        else:
            print("   ⚠ No tool calls made (model may have answered directly)")

    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n" + "="*60)
    print("All Gemini tests passed! ✓")
    print("="*60)

except Exception as e:
    print(f"\n✗ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
