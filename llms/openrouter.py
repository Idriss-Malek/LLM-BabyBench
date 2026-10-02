# llms/openrouter.py
#
# OpenRouter provider (OpenAI-compatible endpoint). Lets us re-run configs that
# were originally written for the native `anthropic` / `openai` / `deepinfra`
# providers without renaming anything: the yaml keeps the original short model
# id (e.g. `claude-sonnet-4-6`) so results merge cleanly with existing runs, and
# MODEL_ALIASES maps it to the OpenRouter id actually requested. The OpenRouter
# id is exposed as `model_name`, which the pipeline records in
# llm_responses[].model_metadata.actual_model_used.

import os
from openai import OpenAI as _OpenAIClient
from dotenv import load_dotenv
from llms.base import AbstractLLM
from typing import List, Dict, Optional

load_dotenv()

# short/native id  ->  OpenRouter id
MODEL_ALIASES = {
    # Anthropic (native ids use dashes, OpenRouter uses dots)
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "claude-opus-4-6": "anthropic/claude-opus-4.6",
    # OpenAI
    "gpt-5.4": "openai/gpt-5.4",
    # DeepInfra ids used in create_yamls.py
    "moonshotai/Kimi-K2.5": "moonshotai/kimi-k2.5",
    "meta-llama/Llama-4-Scout-17B-16E-Instruct": "meta-llama/llama-4-scout",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8": "meta-llama/llama-4-maverick",
}


class OpenRouter(AbstractLLM):

    _VALID_EFFORTS = {"off", "minimal", "low", "medium", "high", "xhigh"}
    # Same thinking budgets the native Anthropic provider uses, so a non-"off"
    # effort on a Claude model through OpenRouter matches llms/anthropic.py.
    _ANTHROPIC_EFFORT_TO_BUDGET = {"low": 2048, "medium": 8192, "high": 24576}

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        system_prompt: str = "",
        reasoning_effort: str = "off",
        provider_order: Optional[List[str]] = None,
        allow_fallbacks: bool = True,
    ):
        self.model = model                                  # as written in the yaml
        self.model_name = MODEL_ALIASES.get(model, model)   # what OpenRouter is asked for
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        if reasoning_effort not in self._VALID_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {sorted(self._VALID_EFFORTS)}, got {reasoning_effort!r}")
        self.reasoning_effort = reasoning_effort

        # Optional provider routing (e.g. ["DeepInfra"] to reproduce DeepInfra runs).
        self.provider_order = provider_order
        self.allow_fallbacks = allow_fallbacks

        self.conversation_history: List[Dict[str, str]] = []

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OpenRouter API key must be set as OPENROUTER_API_KEY environment variable")

        self.client = _OpenAIClient(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            max_retries=5,
            timeout=600,
            default_headers={"X-Title": "babybench"},
        )

    # ------------------------------------------------------------------ core
    def _build_kwargs(self, messages):
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        extra_body: Dict = {}

        if self.reasoning_effort == "off":
            # Mirror the native providers exactly: no reasoning parameter at all,
            # sampling temperature as configured.
            kwargs["temperature"] = self.temperature
        else:
            if self.model_name.startswith("anthropic/"):
                budget = self._ANTHROPIC_EFFORT_TO_BUDGET.get(self.reasoning_effort, 8192)
                extra_body["reasoning"] = {"max_tokens": budget}
                # Anthropic requires temperature=1 with extended thinking, and
                # max_tokens must exceed the thinking budget (see llms/anthropic.py).
                kwargs["temperature"] = 1
                if kwargs["max_tokens"] < budget + 4096:
                    kwargs["max_tokens"] = budget + 4096
            else:
                extra_body["reasoning"] = {"effort": self.reasoning_effort}
                # Reasoning models reject a non-default temperature.

        if self.provider_order:
            extra_body["provider"] = {"order": list(self.provider_order),
                                      "allow_fallbacks": bool(self.allow_fallbacks)}
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def generate(self, prompt: str, use_history: bool = False) -> str:
        try:
            self.add_user_message(prompt)
            messages = self._prepare_messages(use_history)

            response = self.client.chat.completions.create(**self._build_kwargs(messages))

            # OpenRouter can return a 200 whose body is an error object; the SDK
            # then yields an object with `error` set and `choices` None.
            err = getattr(response, "error", None)
            if err:
                raise RuntimeError(f"OpenRouter returned error: {err}")
            if not response.choices:
                raise RuntimeError(f"OpenRouter returned no choices: {response}")

            content = response.choices[0].message.content
            assistant_response = (content or "").strip()

            if use_history:
                self.add_assistant_message(assistant_response)
            else:
                self.conversation_history.pop()

            return assistant_response

        except Exception as e:
            raise Exception(f"OpenRouter API error: {str(e)}")

    # -------------------------------------------------------------- helpers
    def _prepare_messages(self, use_history: bool = True) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if use_history:
            messages.extend(self.conversation_history)
        elif self.conversation_history:
            messages.append(self.conversation_history[-1])
        return messages

    def chat(self, message: str) -> str:
        return self.generate(message, use_history=True)

    def single_prompt(self, prompt: str) -> str:
        return self.generate(prompt, use_history=False)

    def add_user_message(self, content: str) -> None:
        self.conversation_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.conversation_history.append({"role": "assistant", "content": content})

    def clear_history(self) -> None:
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()
