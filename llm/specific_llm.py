from .base_llm import BaseLLM

class Default(BaseLLM):
    def format_message(self, role, content):
        """
        No format applied.
        """
        return content

class QwenLLM(BaseLLM):
    def format_message(self, role, content):
        """
        Formats the message using Qwen's expected template.
        """
        return f"<|im_start|>{role}\n{content}<|im_end|>"

class GPTLLM(BaseLLM):
    def format_message(self, role, content):
        """
        Formats the message using OpenAI's GPT-like template.
        """
        if role == "system":
            return f"System: {content}\n"
        elif role == "user":
            return f"User: {content}\n"
        elif role == "assistant":
            return f"Assistant: {content}\n"
