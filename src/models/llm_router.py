from typing import List, Dict, Any, Literal
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain_community.llms import HuggingFacePipeline
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
import os
from metrics_tracker import MetricsTracker
import psutil

class LLMRouter:
    """Handles LLM initialization and routing"""
    
    def _verify_hf_token(self):
        """Verify HuggingFace token is valid."""
        try:
            token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
            if not token:
                raise ValueError("No HuggingFace token found in environment")
            api = HfApi(token=token)
            api.whoami()
            return True
        except Exception as e:
            print("\nError: Invalid HuggingFace token")
            print("Please check your .env file contains a valid token:")
            print("HUGGINGFACEHUB_API_TOKEN=your-token-here")
            if isinstance(e, HfHubHTTPError):
                print(f"HuggingFace Error: {e.response.text}")
            return False
    
    def __init__(self, llm_type="local", llm_name="mistralai/Mistral-7B-Instruct-v0.3"):
        load_dotenv()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Verify token before initializing model
        if not self._verify_hf_token():
            raise ValueError("Invalid or missing HuggingFace token")
        
        # Initialize model with explicit truncation settings
        self.model, self.tokenizer = self._initialize_llm(llm_type, llm_name)
    
    def _initialize_llm(self, llm_type="local", llm_name="mistralai/Mistral-7B-Instruct-v0.3"):
        print(f"Attempting to load model: {llm_name}")
        print(f"Using device: {self.device}")
        
        try:
            hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
            
            # Initialize tokenizer
            print("\nStep 1: Initializing tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                llm_name,
                token=hf_token
            )
            # Set padding token
            tokenizer.pad_token = tokenizer.eos_token
            print("Tokenizer initialized successfully")
            
            # Initialize model
            print("\nStep 2: Loading model...")
            model = AutoModelForCausalLM.from_pretrained(
                llm_name,
                device_map="auto",
                torch_dtype=torch.float16,
                token=hf_token
            )
            print("Model loaded successfully")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"\nDetailed error information:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise ValueError(f"Error initializing model: {str(e)}")
    
    def generate_response(self, query, relevant_docs):
        try:
            # Debug print to see what documents we're getting
            print("\nNumber of relevant docs:", len(relevant_docs))
            
            doc_texts = []
            for doc in relevant_docs:
                if isinstance(doc, str):
                    doc_texts.append(doc)
                elif hasattr(doc, 'content'):
                    doc_texts.append(doc.content)
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
