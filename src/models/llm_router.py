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
    
    def __init__(self, 
                 llm_type: str = "huggingface", 
                 llm_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 temperature: float = 0.1):
        """
        Initialize the LLM router.
        
        Args:
            llm_type: Type of LLM ("openai", "anthropic", "huggingface", "local")
            llm_name: Name/path of the specific model
            temperature: Temperature for text generation
        """
        self.temperature = temperature
        # Check for Apple Silicon GPU
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f"Using device: {self.device}")
        self.llm = self._initialize_llm(llm_type, llm_name)
    
    def _verify_hf_token(self):
        """Verify HuggingFace token is valid."""
        try:
            token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
            if not token:
                raise ValueError("No HuggingFace token found in environment")
            api = HfApi(token=token)
            # Try to get user info - this will fail if token is invalid
            api.whoami()
            return True
        except Exception as e:
            print("\nError: Invalid HuggingFace token")
            print("Please check your .env file contains a valid token:")
            print("HUGGINGFACEHUB_API_TOKEN=your-token-here")
            if isinstance(e, HfHubHTTPError):
                print(f"HuggingFace Error: {e.response.text}")
            return False
    
    def _initialize_llm(self, llm_type: str, llm_name: str):
        """Initialize and return the specified LLM backend."""
        load_dotenv()
        token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
        if not token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment")
        
        if llm_type == "openai":
            return ChatOpenAI(
                model_name=llm_name,
                temperature=self.temperature
            )
            
        elif llm_type == "anthropic":
            return ChatAnthropic(
                model=llm_name,
                temperature=self.temperature
            )
            
        elif llm_type == "local":
            llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
            llm_model = AutoModelForCausalLM.from_pretrained(llm_name)
            pipe = pipeline(
                "text-generation",
                model=llm_model,
                tokenizer=llm_tokenizer,
                max_length=2048,
                temperature=self.temperature,
                device=self.device
            )
            return HuggingFacePipeline(pipeline=pipe)
            
        elif llm_type == "huggingface":
            if not self._verify_hf_token():
                raise ValueError("Invalid HuggingFace token")
            metrics = MetricsTracker()
            try:
                print(f"\nAttempting to load model: {llm_name}")
                print(f"Using device: {self.device}")
                
                # First load tokenizer with specific config
                tokenizer = AutoTokenizer.from_pretrained(
                    llm_name,
                    token=token,
                    trust_remote_code=True,
                    force_download=True
                )
                
                # Then load model with specific config
                model = AutoModelForCausalLM.from_pretrained(
                    llm_name,
                    token=token,
                    trust_remote_code=True,
                    # Mistral recommends bfloat16, but it's not supported on Apple Silicon (M1/M2).
                    # Using torch_dtype="auto" to let the model choose the best available format.
                    # This should handle MPS (Apple GPU) compatibility automatically.
                    device_map={"": self.device},  # Explicitly set device
                    torch_dtype="auto"             # Let the model decide
                )
                
                # Create pipeline with loaded components
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    token=token,
                    trust_remote_code=True,
                    max_length=2048,
                    temperature=self.temperature,
                    top_p=0.95,
                    do_sample=True,
                    device_map="auto"
                )
                
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                print("\nError: Could not access model.")
                print("Please make sure you:")
                print("1. Have a valid HuggingFace token in .env")
                print("2. Have accepted the model terms at:")
                print(f"   https://huggingface.co/{llm_name}")
                raise
                
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    def generate_response(self, query: str, relevant_docs: List[Any]) -> str:
        """
        Generate response using relevant documents.
        
        Args:
            query: User's question
            relevant_docs: List of relevant document chunks
            
        Returns:
            str: Generated response
        """
        system_prompt = "<s>[INST] You are a research assistant analyzing academic papers about labor and human rights compliance in global supply chains. Provide detailed, academically-oriented answers based on the provided sources. Always cite the source documents in your response. [/INST]"
        
        # Prepare context from relevant documents
        context = "\n\n".join([
            f"From '{doc.metadata.get('source', 'unknown')}': {doc.page_content}"
            for doc in relevant_docs
        ])
        
        prompt = f"<s>[INST] Here are relevant academic sources:\n\n{context}\n\nBased on these sources, {query} [/INST]"
        
        # Combine system prompt and user prompt
        full_prompt = system_prompt + prompt
        
        response = self.llm.invoke(full_prompt)
        return response.content
