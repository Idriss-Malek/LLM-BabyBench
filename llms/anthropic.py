# llms/anthropic.py

import os
import anthropic
from dotenv import load_dotenv
from llms.base import AbstractLLM
from typing import List, Dict, Optional

# Load environment variables from .env file
load_dotenv()

class Anthropic(AbstractLLM):

    _EFFORT_TO_BUDGET = {
        "off": 0,
        "low": 2048,
        "medium": 8192,
        "high": 24576,
    }

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.0, max_tokens: int = 8192, system_prompt: str = "", reasoning_effort: str = "off"):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        if reasoning_effort not in self._EFFORT_TO_BUDGET:
            raise ValueError(f"reasoning_effort must be one of {list(self._EFFORT_TO_BUDGET)}, got {reasoning_effort!r}")
        self.reasoning_effort = reasoning_effort
        self._thinking_budget = self._EFFORT_TO_BUDGET[reasoning_effort]
        if self._thinking_budget > 0 and self.max_tokens < self._thinking_budget + 4096:
            self.max_tokens = self._thinking_budget + 4096
        
        # Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError("Anthropic API key must be set as ANTHROPIC_API_KEY environment variable")
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
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
            if use_history:
                messages = self.conversation_history.copy()
            else:
                messages = [{"role": "user", "content": prompt}]
            
            create_kwargs = dict(
                model=self.model,
                messages=messages,
                system=self.system_prompt if self.system_prompt else "",
                max_tokens=self.max_tokens,
            )
            if self._thinking_budget > 0:
                create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}
                create_kwargs["temperature"] = 1
            else:
                create_kwargs["temperature"] = self.temperature

            # Anthropic SDK rejects non-streaming requests that may exceed 10
            # minutes; high-effort thinking (24k budget) on complex tasks can.
            # Low/medium budgets stay well under, so only stream for high.
            if self._thinking_budget > 16384:
                with self.client.messages.stream(**create_kwargs) as stream:
                    final_message = stream.get_final_message()
                content = final_message.content
            else:
                response = self.client.messages.create(**create_kwargs)
                content = response.content

            assistant_response = next(
                (block.text for block in content if getattr(block, "type", None) == "text"),
                ""
            ).strip()
            
            # Add assistant response to history, or drop this turn entirely in
            # stateless mode so history does not accumulate orphan user messages.
            if use_history:
                self.add_assistant_message(assistant_response)
            else:
                self.conversation_history.pop()

            return assistant_response
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def chat(self, message: str) -> str:
        """
        Convenience method for chatting with history enabled.
        """
        return self.generate(message, use_history=True)
    
    '''
    def single_prompt(self, prompt: str) -> str:
        """
        Generate a response without using conversation history.
        """
        return self.generate(prompt, use_history=False)
    '''
        
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
    
    def add_custom_message(self, role: str, content: str) -> None:
        """Add a custom message to the conversation history."""
        if role not in ["user", "assistant"]:
            raise ValueError("Role must be either 'user' or 'assistant'")
        
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
            if message["role"] not in ["user", "assistant"]:
                raise ValueError("Role must be either 'user' or 'assistant'")
        
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
                else:
                    result.append(f"**Assistant:** {message['content']}")
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
        Rough estimate of tokens in the conversation history.
        This is a simple approximation (1 token ≈ 4 characters).
        """
        total_chars = sum(len(msg["content"]) for msg in self.conversation_history)
        return total_chars // 4
