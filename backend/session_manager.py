from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Message:
    """Represents a single message in a conversation"""
    role: str     # "user" or "assistant"
    content: str  # The message content

@dataclass
class SessionInfo:
    """Information about a conversation session"""
    messages: List[Message]
    current_model: Optional[str] = None
    compression_summary: Optional[str] = None

class SessionManager:
    """Manages conversation sessions and message history"""

    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.sessions: Dict[str, SessionInfo] = {}
        self.session_counter = 0
    
    def create_session(self) -> str:
        """Create a new conversation session"""
        self.session_counter += 1
        session_id = f"session_{self.session_counter}"
        self.sessions[session_id] = SessionInfo(messages=[])
        return session_id

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to the conversation history"""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionInfo(messages=[])

        message = Message(role=role, content=content)
        self.sessions[session_id].messages.append(message)

        # Keep conversation history within limits
        if len(self.sessions[session_id].messages) > self.max_history * 2:
            self.sessions[session_id].messages = self.sessions[session_id].messages[-self.max_history * 2:]
    
    def add_exchange(self, session_id: str, user_message: str, assistant_message: str):
        """Add a complete question-answer exchange"""
        self.add_message(session_id, "user", user_message)
        self.add_message(session_id, "assistant", assistant_message)
    
    def get_conversation_history(self, session_id: Optional[str]) -> Optional[str]:
        """Get formatted conversation history for a session"""
        if not session_id or session_id not in self.sessions:
            return None

        session_info = self.sessions[session_id]
        messages = session_info.messages

        if not messages and not session_info.compression_summary:
            return None

        # Start with compression summary if it exists
        formatted_messages = []
        if session_info.compression_summary:
            formatted_messages.append(f"[Previous conversation summary: {session_info.compression_summary}]")

        # Add current messages
        for msg in messages:
            formatted_messages.append(f"{msg.role.title()}: {msg.content}")

        return "\n".join(formatted_messages)

    def get_current_model(self, session_id: str) -> Optional[str]:
        """Get the current model being used for a session"""
        if session_id not in self.sessions:
            return None
        return self.sessions[session_id].current_model

    def set_current_model(self, session_id: str, model_id: str):
        """Set the current model for a session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionInfo(messages=[])
        self.sessions[session_id].current_model = model_id

    def compress_conversation(self, session_id: str, previous_model: str) -> str:
        """
        Compress the conversation history into a summary for model switching.

        Args:
            session_id: The session to compress
            previous_model: The model that was previously being used

        Returns:
            Compressed summary string
        """
        if session_id not in self.sessions:
            return ""

        session_info = self.sessions[session_id]
        messages = session_info.messages

        if not messages:
            return ""

        # Create a concise summary of the conversation
        summary_parts = [f"Previous conversation with {previous_model}:"]

        # Group messages into exchanges (user + assistant pairs)
        exchanges = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                user_msg = messages[i].content if messages[i].role == "user" else ""
                assistant_msg = messages[i + 1].content if messages[i + 1].role == "assistant" else messages[i].content

                # Truncate long messages for summary
                user_msg = user_msg[:200] + "..." if len(user_msg) > 200 else user_msg
                assistant_msg = assistant_msg[:200] + "..." if len(assistant_msg) > 200 else assistant_msg

                exchanges.append(f"Q: {user_msg} | A: {assistant_msg}")

        # Take last few exchanges (max 3 for summary)
        recent_exchanges = exchanges[-3:] if len(exchanges) > 3 else exchanges
        summary_parts.extend(recent_exchanges)

        summary = " | ".join(summary_parts)

        # Store the compression summary
        session_info.compression_summary = summary

        # Clear the messages since we've compressed them
        session_info.messages = []

        return summary

    def clear_session(self, session_id: str):
        """Clear all messages from a session"""
        if session_id in self.sessions:
            self.sessions[session_id] = SessionInfo(messages=[])