# llms/openai.py

import os
import openai
from dotenv import load_dotenv
from llms.base import AbstractLLM
from typing import List, Dict, Optional

# Load environment variables from .env file
load_dotenv()

class OpenAI(AbstractLLM):
    
    _VALID_EFFORTS = {"off", "minimal", "low", "medium", "high", "xhigh"}

    def __init__(self, model: str = "gpt-5-thinking", temperature: float = 0.0, max_tokens: int = 8192, system_prompt: str = "", reasoning_effort: str = "off"):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        if reasoning_effort not in self._VALID_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {sorted(self._VALID_EFFORTS)}, got {reasoning_effort!r}")
        self.reasoning_effort = reasoning_effort
        
        # Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OpenAI API key must be set as OPENAI_API_KEY environment variable")
            
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, use_history: bool = False) -> str:
        """
        Generate a response to a prompt.
        
        Args:
            prompt: The user's message
            use_history: Whether to include conversation history in the request
        
        Returns:
            The assistant's response
        """
        try:
            # Add user message to history
            self.add_user_message(prompt)
            
            # Prepare messages for API call
            messages = self._prepare_messages(use_history)
            
            create_kwargs = dict(
                model=self.model,
                messages=messages,
                max_completion_tokens=self.max_tokens,
            )
            if self.reasoning_effort != "off":
                # GPT-5 reasoning models reject non-default temperature.
                create_kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                create_kwargs["temperature"] = self.temperature

            response = self.client.chat.completions.create(**create_kwargs)
            
            assistant_response = response.choices[0].message.content.strip()
            
            # Add assistant response to history, or drop this turn entirely in
            # stateless mode so history does not accumulate orphan user messages.
            if use_history:
                self.add_assistant_message(assistant_response)
            else:
                self.conversation_history.pop()

            return assistant_response
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _prepare_messages(self, use_history: bool = True) -> List[Dict[str, str]]:
        """
        Prepare messages for the API call, including system prompt and history.
        """
        messages = []
        
        # Add system prompt if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        if use_history:
            # Add conversation history
            messages.extend(self.conversation_history)
        else:
            # Only add the last user message (already added to history)
            if self.conversation_history:
                messages.append(self.conversation_history[-1])
        
        return messages
    
    def chat(self, message: str) -> str:
        """
        Convenience method for chatting with history enabled.
        """
        return self.generate(message, use_history=True)
    
    def single_prompt(self, prompt: str) -> str:
        """
        Generate a response without using conversation history.
        """
        return self.generate(prompt, use_history=False)
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history."""
        self.conversation_history.append({
            "role": "user", 
            "content": content
        })
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history."""
        self.conversation_history.append({
            "role": "assistant", 
            "content": content
        })
    
    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation history."""
        self.conversation_history.append({
            "role": "system",
            "content": content
        })
    
    def add_custom_message(self, role: str, content: str) -> None:
        """Add a custom message to the conversation history."""
        if role not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")
        
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()
    
    def set_history(self, history: List[Dict[str, str]]) -> None:
        """Set the conversation history."""
        # Validate history format
        for message in history:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dictionary")
            if "role" not in message or "content" not in message:
                raise ValueError("Each message must have 'role' and 'content' keys")
            if message["role"] not in ["user", "assistant", "system"]:
                raise ValueError("Role must be 'user', 'assistant', or 'system'")
        
        self.conversation_history = history.copy()
    
    def get_conversation_length(self) -> int:
        """Get the number of messages in the conversation history."""
        return len(self.conversation_history)
    
    def get_last_n_messages(self, n: int) -> List[Dict[str, str]]:
        """Get the last n messages from the conversation history."""
        return self.conversation_history[-n:] if n > 0 else []
    
    def trim_history(self, max_messages: int) -> None:
        """Keep only the last max_messages in the conversation history."""
        if max_messages > 0:
            self.conversation_history = self.conversation_history[-max_messages:]
    
    def update_system_prompt(self, new_system_prompt: str) -> None:
        """
        Update the system prompt. This will affect future API calls but won't
        modify existing conversation history.
        """
        self.system_prompt = new_system_prompt
    
    def insert_system_message_in_history(self, content: str, position: int = 0) -> None:
        """
        Insert a system message at a specific position in the conversation history.
        Useful for adding context mid-conversation.
        """
        system_message = {"role": "system", "content": content}
        self.conversation_history.insert(position, system_message)
    
    def export_conversation(self, format: str = "json") -> str:
        """
        Export the conversation in different formats.
        
        Args:
            format: "json", "text", or "markdown"
        """
        if format == "json":
            import json
            return json.dumps(self.conversation_history, indent=2)
        
        elif format == "text":
            result = []
            for message in self.conversation_history:
                role = message["role"].upper()
                content = message["content"]
                result.append(f"{role}: {content}")
            return "\n\n".join(result)
        
        elif format == "markdown":
            result = []
            for message in self.conversation_history:
                if message["role"] == "user":
                    result.append(f"**User:** {message['content']}")
                elif message["role"] == "assistant":
                    result.append(f"**Assistant:** {message['content']}")
                else:  # system
                    result.append(f"**System:** {message['content']}")
            return "\n\n".join(result)
        
        else:
            raise ValueError("Format must be 'json', 'text', or 'markdown'")
    
    def load_conversation_from_json(self, json_str: str) -> None:
        """Load conversation history from a JSON string."""
        import json
        try:
            history = json.loads(json_str)
            self.set_history(history)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
    
    def count_tokens_estimate(self) -> int:
        """
        Rough estimate of tokens in the conversation history and system prompt.
        This is a simple approximation (1 token ≈ 4 characters).
        For more accurate token counting, consider using tiktoken library.
        """
        total_chars = 0
        
        # Count system prompt
        if self.system_prompt:
            total_chars += len(self.system_prompt)
        
        # Count conversation history
        total_chars += sum(len(msg["content"]) for msg in self.conversation_history)
        
        return total_chars // 4
    
    def get_conversation_stats(self) -> Dict[str, int]:
        """
        Get statistics about the current conversation.
        """
        stats = {
            "total_messages": len(self.conversation_history),
            "user_messages": 0,
            "assistant_messages": 0,
            "system_messages": 0,
            "estimated_tokens": self.count_tokens_estimate()
        }
        
        for message in self.conversation_history:
            if message["role"] == "user":
                stats["user_messages"] += 1
            elif message["role"] == "assistant":
                stats["assistant_messages"] += 1
            elif message["role"] == "system":
                stats["system_messages"] += 1
        
        return stats
    
    def create_conversation_summary(self, max_length: int = 200) -> str:
        """
        Create a brief summary of the conversation for context preservation.
        Useful when trimming long conversations.
        """
        if not self.conversation_history:
            return "No conversation history."
        
        # Get the main topics discussed
        all_content = " ".join(msg["content"] for msg in self.conversation_history 
                              if msg["role"] in ["user", "assistant"])
        
        # Simple summary by taking first part of conversation
        if len(all_content) <= max_length:
            return all_content
        else:
            return all_content[:max_length] + "..."
    
    def reset_with_summary(self, summary_length: int = 200) -> str:
        """
        Reset conversation but preserve context with a summary.
        Returns the summary that was created.
        """
        summary = self.create_conversation_summary(summary_length)
        self.clear_history()
        
        if summary != "No conversation history.":
            self.add_system_message(f"Previous conversation summary: {summary}")
        
        return summary