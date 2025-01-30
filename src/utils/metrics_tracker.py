"""
LLM Performance Metrics Tracker
-----------------------------

This module provides functionality to track, compare, and analyze the performance
of different Language Models (LLMs) in a RAG system. It captures metrics such as:
- Response time
- Cost estimates
- Success/failure rates
- Response content

The metrics are saved to JSON files in a 'model_comparisons' directory for later analysis.

Key features:
- Timing measurement for queries
- Cost estimation based on token count
- Error handling and logging
- Response storage and comparison
- Performance summary generation

Usage:
    metrics = MetricsTracker()
    result = metrics.track_query("Model-Name", "Query", query_function)
    comparison = metrics.compare_responses("Query")

The system automatically creates a 'model_comparisons' directory to store results
and provides methods to analyze and compare performance across different models.

Note: Cost estimates are approximate and based on published API rates.
"""

import time
from datetime import datetime
import json
from typing import Dict, Any
import os
import psutil

class MetricsTracker:
    def __init__(self):
        """Initialize the metrics tracker and create storage directory."""
        self.results_dir = "model_comparisons"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def track_query(self, model_name: str, query: str, func):
        """
        Track the performance metrics of a query execution.
        
        Args:
            model_name (str): Name of the LLM being tested
            query (str): The query being processed
            func: Function that executes the query
            
        Returns:
            Dict containing timing, cost, and response information
        """
        start_time = time.time()
        try:
            response = func()
            duration = time.time() - start_time
            
            # Estimate cost (rough approximations)
            cost = self._estimate_cost(model_name, query, response)
            
            result = {
                "model": model_name,
                "query": query,
                "response": response,
                "duration_seconds": duration,
                "estimated_cost_usd": cost,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
        except Exception as e:
            result = {
                "model": model_name,
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
        
        self._save_result(result)
        return result
    
    def _estimate_cost(self, model_name: str, query: str, response: str) -> float:
        """
        Estimate the cost of a query based on token count and model rates.
        
        Args:
            model_name (str): Name of the LLM
            query (str): Input query
            response (str): Model's response
            
        Returns:
            float: Estimated cost in USD
        """
        # Rough cost estimates per 1K tokens
        cost_rates = {
            "GPT-4": 0.03,
            "Claude-3": 0.015,
            "Mistral-7B": 0.0,  # Free when self-hosted
            "Llama-2": 0.0,     # Free when self-hosted
            "Gemma": 0.0        # Free when self-hosted
        }
        
        # Rough token count (characters / 4)
        total_chars = len(query) + len(response)
        token_estimate = total_chars / 4
        
        return (token_estimate / 1000) * cost_rates.get(model_name, 0.0)
    
    def _save_result(self, result: Dict[str, Any]):
        """
        Save a query result to a JSON file.
        
        Args:
            result (Dict[str, Any]): The result data to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result['model']}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    def compare_responses(self, query: str) -> Dict[str, Any]:
        """
        Compare all responses for a given query across different models.
        
        Args:
            query (str): The query to compare responses for
            
        Returns:
            Dict containing comparison results and summary statistics
        """
        results = []
        for file in os.listdir(self.results_dir):
            if file.endswith('.json'):
                with open(os.path.join(self.results_dir, file), 'r') as f:
                    result = json.load(f)
                    if result.get('query') == query:
                        results.append(result)
        
        return {
            "query": query,
            "comparisons": results,
            "summary": {
                "fastest": min(results, key=lambda x: x.get('duration_seconds', float('inf')))['model'],
                "total_cost": sum(r.get('estimated_cost_usd', 0) for r in results)
            }
        }

    def track_system_resources(self) -> Dict[str, float]:
        """Track detailed system resources"""
        process = psutil.Process()
        memory = process.memory_info()
        
        return {
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "memory_used_mb": memory.rss / 1024 / 1024,  # Convert to MB
            "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
            "cpu_count": psutil.cpu_count()
        }