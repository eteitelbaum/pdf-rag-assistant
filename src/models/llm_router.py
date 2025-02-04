from typing import List, Dict, Any, Literal
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain_community.llms import HuggingFacePipeline
from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError
import os
from metrics_tracker import MetricsTracker
import psutil

# Global variables to cache model and tokenizer
_GLOBAL_MODEL = None
_GLOBAL_TOKENIZER = None

class LLMRouter:
    """Handles LLM initialization and routing"""
    
    def _verify_hf_token(self):
        """Verify HuggingFace token is valid."""
        try:
            token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
            if not token:
                raise ValueError("No HuggingFace token found in environment variables")
            
            # Set token for huggingface_hub
            login(token)
            
            # Verify token works
            api = HfApi(token=token)
            api.whoami()
            return True
        except Exception as e:
            print("\nError: HuggingFace token verification failed")
            print("Please ensure you have:")
            print("1. Set HUGGINGFACE_TOKEN or HUGGINGFACEHUB_API_TOKEN in your environment")
            print("2. Accepted the model terms at https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3")
            print(f"\nDetailed error: {str(e)}")
            return False
    
    def __init__(self, model_path: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        """Initialize LLM Router with in-memory caching."""
        global _GLOBAL_MODEL, _GLOBAL_TOKENIZER
        
        # Verify token before attempting to load model
        if not self._verify_hf_token():
            raise ValueError("HuggingFace token verification failed")
            
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if model is already loaded in memory
        if _GLOBAL_MODEL is None or _GLOBAL_TOKENIZER is None:
            print(f"Using device: {self.device}")
            print(f"Attempting to load model: {model_path}")
            
            print("\nStep 1: Initializing tokenizer...")
            _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
            # Explicitly set padding token for Mistral
            if _GLOBAL_TOKENIZER.pad_token is None:
                _GLOBAL_TOKENIZER.pad_token = _GLOBAL_TOKENIZER.eos_token
                _GLOBAL_TOKENIZER.pad_token_id = _GLOBAL_TOKENIZER.eos_token_id
            print("Tokenizer initialized successfully")
            
            print("\nStep 2: Loading model...")
            _GLOBAL_MODEL = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                #load_in_4bit=True,
                #bnb_4bit_compute_dtype=torch.bfloat16
            )
            print("Model loaded successfully")
        else:
            print("Using already loaded model and tokenizer")
            
        self.model = _GLOBAL_MODEL
        self.tokenizer = _GLOBAL_TOKENIZER
    
    def generate_response(self, query, relevant_docs):
        try:
            # Debug print to see what documents we're getting
            print("\nNumber of relevant docs:", len(relevant_docs))
            
            doc_texts = []
            for doc in relevant_docs:
                if isinstance(doc, str):
                    doc_texts.append(doc)
                elif hasattr(doc, 'page_content'):
                    doc_texts.append(doc.page_content)
                else:
                    print(f"Warning: Skipping document with unexpected format: {type(doc)}")
            
            # Debug print the first bit of context
            context = "\n\n".join(doc_texts[:3])
            print("\nFirst 200 chars of context:")
            print(context[:200])
            
            prompt = f"""Based on the following academic literature excerpts, {query}

Context:
{context}

Response:"""
            
            print("\nPrompt length:", len(prompt))
            
            # Generate with explicit truncation settings
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):]
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I encountered an error while processing your query."
