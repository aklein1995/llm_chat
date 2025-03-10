from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForVision2Seq, AutoProcessor
from langchain_huggingface import HuggingFacePipeline
from PIL import Image
import torch

class BaseLLM(ABC):
    def __init__(self, 
                 model_path, 
                 temperature=0.9, 
                 device='cuda', 
                 max_new_tokens=2000, 
                 repetition_penalty=1.1, 
                 truncation=True, 
                 padding=False):
        """
        Base LLM class that initializes a Hugging Face pipeline with customizable parameters.

        Parameters:
        - model_path (str): Path to the model or Hugging Face model identifier.
        - temperature (float): Sampling temperature; higher values make output more random.
        - device (str): Device to run the model on ('cuda' or 'cpu').
        - max_new_tokens (int): Maximum number of tokens to generate.
        - repetition_penalty (float): Controls the likelihood of repeated text.
        - truncation (bool): Whether to truncate inputs.
        - padding (bool): Whether to apply padding.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.hf_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=temperature,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            truncation=truncation,
            padding=padding,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.llm = HuggingFacePipeline(pipeline=self.hf_pipeline)

        self.raw_memory = []
        self.formatted_memory = ""

    @abstractmethod
    def format_message(self, role, content):
        """Abstract method for formatting messages. Must be implemented by subclasses."""
        pass

    def __call__(self, prompt, system_message=None):
        """
        Generates a response from the model, using an optional system message.
        """
        if system_message:
            formatted_system_message = self.format_message("system", system_message)
            self.formatted_memory += formatted_system_message

        user_message = self.format_message("user", prompt)
        self.formatted_memory += user_message

        response = self.llm.invoke(self.formatted_memory)
        assistant_message = self.format_message("assistant", response)
        self.formatted_memory += assistant_message

        return response

    def clear_memory(self, boolean=True):
        """
        Clears conversation memory if boolean is True.
        """
        if boolean:
            print("Cleaning...")
            self.raw_memory = []
            self.formatted_memory = ""


class BaseVLM(ABC):
    def __init__(self, model_path, temperature=0.9, device="cuda", max_new_tokens=200):
        """
        Base Vision-Language Model (VLM) class.
        Supports both text and images as input.
        """
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.hf_pipeline = pipeline(
            # "image-to-text",
            "image-text-to-text",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            image_processor=self.processor,
            processor=self.processor,
            max_new_tokens=max_new_tokens
            # temperature=temperature,
        )
        
        self.memory = ""

    @abstractmethod
    def format_message(self, role, content):
        """Abstract method for formatting messages. Must be implemented by subclasses."""
        pass

    def __call__(self, text, image_paths=None):
        """
        Generates a response using the VLM.
        """
        inputs={}
        
        # add image
        if image_paths:
            inputs["images"] = [Image.open(i).convert("RGB") for i in image_paths]

        # add text
        formatted_text = self.format_message(text,image_paths)
        inputs["text"] = formatted_text

        print("\nInput to pipeline:", inputs)

        # process info
        response = self.hf_pipeline(**inputs)
        return response[0]["generated_text"]
    
    def clear_memory(self):
        """Clears conversation memory."""
        print("Cleaning memory...")
        self.memory = ""
