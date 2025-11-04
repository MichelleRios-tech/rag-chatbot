from typing import List, Optional, Dict, Any
from llm_provider import LLMProvider

class AIGenerator:
    """Handles interactions with LLM providers for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Search Tool Usage:
- **search_course_content**: Use for questions about specific course content or detailed educational materials
- **get_course_outline**: Use for questions about course structure, lesson lists, or course overview
  - Returns: Course title, course link, instructor, and complete list of lessons with numbers and titles
  - Use when users ask about "what lessons", "course structure", "outline", "table of contents", etc.
- **One search per query maximum** (choose the most appropriate tool)
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, provider: LLMProvider):
        """
        Initialize AI Generator with a provider.

        Args:
            provider: An LLMProvider instance (Anthropic or LM Studio)
        """
        self.provider = provider

        # Base generation parameters
        self.temperature = 0
        self.max_tokens = 800
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare messages
        messages = [{"role": "user", "content": query}]

        # Get initial response from provider
        response_text, tool_calls, updated_messages = self.provider.generate(
            system_prompt=system_content,
            messages=messages,
            tools=tools,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Handle tool execution if needed
        if tool_calls and tool_manager:
            return self._handle_tool_execution(
                tool_calls,
                updated_messages,
                system_content,
                tool_manager
            )

        # Return direct response
        return response_text
    
    def _handle_tool_execution(
        self,
        tool_calls: List[Dict[str, Any]],
        messages: List[Dict[str, str]],
        system_content: str,
        tool_manager
    ) -> str:
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            tool_calls: List of tool calls from the provider
            messages: Current message history
            system_content: System prompt content
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Execute all tool calls and collect results
        tool_results = []
        for tool_call in tool_calls:
            tool_result = tool_manager.execute_tool(
                tool_call["name"],
                **tool_call["input"]
            )

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call["id"],
                "content": tool_result
            })

        # Add assistant message with tool calls (provider-specific format will be handled)
        # For simplicity, we'll add tool results as a user message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Get final response without tools
        response_text, _, _ = self.provider.generate(
            system_prompt=system_content,
            messages=messages,
            tools=None,  # Don't allow tools in follow-up
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response_text