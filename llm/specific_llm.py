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
    def format_message(self, text, image_paths=None):
        """
            Formats the message using `apply_chat_template()`.
        """
        messages = [
            # message1
            {"role": "system", "content": "You are a helpful assistant."},
            # message2 where we attach attach images
            {
                "role": "user", 
                "content": [ 
                    {"type":"text", "text": text},
                    # {"type":"image","image": image_path},]
                ] + [{"type": "image", "image": img} for img in image_paths]  # Dynamically add images

            }
        ]

        formatted_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            return_tensors=None  # Return as string for `pipeline()`
        )
        print("\nFormatted text:", formatted_text)
        return formatted_text

