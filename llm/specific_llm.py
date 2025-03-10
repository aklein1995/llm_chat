from .base_llm import BaseLLM, BaseVLM

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
class QwenVLM(BaseVLM):
    def format_message(self, role, content, image_path=None):
        """
        Formats messages according to the Qwen2.5-VL template.
        
        - role: "system", "user", or "assistant".
        - content: The text prompt.
        - image_path: Path to an image (optional).
        
        If an image is provided, it is referenced as `<image>` within the prompt.
        """

        message = f"<|im_start|>{role}\n{content}"

        if image_path:
            message += "\n<image>"  # Qwen2.5-VL uses <image> to indicate an image is attached.

        message += "<|im_end|>"

        return message
