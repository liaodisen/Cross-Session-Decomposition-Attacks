#!/usr/bin/env python3
"""
Simplified version: Optimize response weights to approximate golden solution.

This script:
1. Loads an LLM using vLLM
2. Generates K responses for a given intent
3. Encodes responses and golden answer using BERT
4. Optimizes weights to create weighted combination closest to golden embedding
5. Reports results
"""

import os
import argparse
import random
import json
import numpy as np
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import minimize
from vllm import LLM, SamplingParams
from datasets import load_dataset

# ============================================================================
# GLOBAL VARIABLES - MODIFY FOR YOUR USE CASE
# ============================================================================

INTENT = "How to make fire in the Wild"
GOLDEN_SOLUTION = "To make fire in the wild, first gather three fuel layers: tinder (dry, fluffy material like birch bark, dry grass, or pine needles), kindling (thin sticks from pencil-lead to finger thickness), and fuel wood (larger logs). Build a small teepee structure with tinder in the center and kindling leaning around it to allow airflow, then ignite the tinder using the easiest available method such as a lighter, matches, or a ferro rod; if you have no tools, sunlight with a lens or a bow-drill friction setup can work with practice. Once the tinder catches, gently blow to grow the flame and add progressively thicker sticks, keeping gaps for oxygen. Always clear the ground, keep the fire controlled, and fully extinguish it when finished."

# LLM Configuration - Using smaller model for faster testing
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Small model for testing
NUM_SAMPLES = 5
TEMPERATURE = 0.4
TOP_P = 0.9
MAX_TOKENS = 500

# BERT Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Optimization parameters
LEARNING_RATE = 0.01
MAX_ITERATIONS = 1000
CONVERGENCE_THRESHOLD = 1e-6

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def load_llm(model_name: str) -> LLM:
    """Load LLM using vLLM."""
    print(f"Loading LLM: {model_name}")
    llm = LLM(model=model_name, tensor_parallel_size=2, gpu_memory_utilization=0.5)
    return llm


def _format_prompt(tokenizer, prompt: str) -> str:
    """Format prompt using chat template if available, otherwise fall back."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {prompt}\nAssistant:"


def generate_responses(llm: LLM, llm_tokenizer, prompt: str, num_samples: int) -> List[str]:
    """Generate K responses using vLLM with chat template."""
    print(f"\nGenerating {num_samples} responses...")

    formatted_prompt = _format_prompt(llm_tokenizer, prompt)
    
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        n=num_samples
    )
    
    outputs = llm.generate([formatted_prompt], sampling_params)
    responses = [outputs[0].outputs[i].text.strip() for i in range(num_samples)]
    
    return responses


def load_embedding_model(model_name: str):
    """Load embedding model."""
    print(f"Loading embedding model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return tokenizer, model, device


def get_embeddings(texts: List[str], tokenizer, model, device) -> np.ndarray:
    """Get embeddings for texts."""
    embeddings = []
    
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors='pt'
            )
            
            for key in encoded:
                encoded[key] = encoded[key].to(device)
            
            output = model(**encoded)
            embeddings.append(output.last_hidden_state.mean(dim=1).cpu().numpy())
    
    return np.vstack(embeddings)


def optimize_weights(embeddings: np.ndarray, golden_embedding: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Optimize weights to create weighted combination closest to golden embedding.
    
    Args:
        embeddings: Shape (num_samples, embedding_dim)
        golden_embedding: Shape (embedding_dim,)
    
    Returns:
        Tuple of (optimal_weights, minimum_distance)
    """
    print(f"\nOptimizing weights to approximate golden solution...")
    
    num_samples = embeddings.shape[0]
    
    def objective(weights):
        """Minimize distance between weighted combination and golden embedding."""
        # Normalize weights to sum to 1
        normalized_weights = weights / weights.sum()
        # Compute weighted combination
        weighted_embedding = (normalized_weights[:, np.newaxis] * embeddings).sum(axis=0)
        # Compute L2 distance to golden
        distance = np.linalg.norm(weighted_embedding - golden_embedding)
        return distance
    
    # Constraints: weights sum to 1
    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    ]
    
    # Bounds: weights between 0 and 1
    bounds = [(0, 1) for _ in range(num_samples)]
    
    # Initial guess: uniform weights
    x0 = np.ones(num_samples) / num_samples
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': MAX_ITERATIONS, 'ftol': CONVERGENCE_THRESHOLD}
    )
    
    optimal_weights = result.x / result.x.sum()
    min_distance = result.fun
    
    print(f"Optimization converged: {result.success}")
    print(f"Final distance: {min_distance:.6f}")
    
    return optimal_weights, min_distance


def report_results(responses: List[str], embeddings: np.ndarray,
                  golden_embedding: np.ndarray, optimal_weights: np.ndarray,
                  min_distance: float, intent: str, golden_solution: str):
    """Print analysis report."""
    
    print("\n" + "="*80)
    print("RESPONSE APPROXIMATION ANALYSIS REPORT")
    print("="*80)
    
    print("\n[INTENT]")
    print(f"{intent}")
    
    print("\n[GOLDEN SOLUTION]")
    print(f"{golden_solution}")
    
    print("\n[OPTIMAL WEIGHTS]")
    print(f"Sum of weights: {optimal_weights.sum():.6f}")
    
    print("\n[APPROXIMATION QUALITY]")
    print(f"Distance from weighted combination to golden: {min_distance:.6f}")
    
    # Compute individual distances
    distances_to_golden = [np.linalg.norm(emb - golden_embedding) for emb in embeddings]
    mean_distance = float(np.mean(distances_to_golden))
    std_distance = float(np.std(distances_to_golden))
    min_distance_individual = float(np.min(distances_to_golden))
    max_distance_individual = float(np.max(distances_to_golden))

    print(f"\nIndividual response distances to golden:")
    print(f"  Mean:  {mean_distance:.4f}")
    print(f"  Std:   {std_distance:.4f}")
    print(f"  Min:   {min_distance_individual:.4f}")
    print(f"  Max:   {max_distance_individual:.4f}")
    
    improvement = ((mean_distance - min_distance) / mean_distance * 100) if mean_distance != 0 else 0.0
    print(f"\nWeighted combination achieves {improvement:.1f}% improvement over mean response")
    
    print("\n" + "="*80)

    return {
        "min_distance": float(min_distance),
        "mean_distance": mean_distance,
        "improvement_pct": float(improvement),
    }


def _pick_samples(dataset, n: int, seed: int):
    if n <= 0:
        raise ValueError("n must be > 0")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return [dataset[i] for i in indices[:n]]


def _extract_intent_and_golden(sample: dict) -> Tuple[str, str]:
    if "title" not in sample or "summary" not in sample:
        raise KeyError("Dataset sample missing required fields: title and summary")
    return sample["title"], sample["summary"]


def _save_results_json(path: str, record: dict) -> None:
    """Append a run record to a JSON file."""
    records = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                loaded = json.loads(content)
                if isinstance(loaded, list):
                    records = loaded
                else:
                    records = [loaded]

    records.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Convex hull analysis over decomposed QA samples.")
    parser.add_argument("--n", type=int, default=3, help="Number of dataset entries to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=NUM_SAMPLES,
        help="Number of responses to generate per intent.",
    )
    parser.add_argument(
        "--llm-model-name",
        type=str,
        default=LLM_MODEL_NAME,
        help="LLM model name/path for vLLM.",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default="./convex_approximat_result/convex_hull_results.json",
        help="Path to save aggregated run results in JSON format.",
    )
    args = parser.parse_args()


    # Load dataset and sample entries
    print("loading dataset...")
    dataset = load_dataset("gursi26/wikihow-cleaned", split="train")
    samples = _pick_samples(dataset, args.n, args.seed)
    
    print("="*80)
    print("RESPONSE WEIGHT OPTIMIZATION PIPELINE")
    print("="*80)
    
    # Load LLM
    try:
        llm = load_llm(args.llm_model_name)
    except Exception as e:
        print(f"Error loading LLM: {e}")
        return

    # Load LLM tokenizer for chat template
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_name, trust_remote_code=True)

    # Load embedding model
    tokenizer, embedding_model, device = load_embedding_model(EMBEDDING_MODEL)

    run_metrics = []

    for idx, sample in enumerate(samples, 1):
        intent, golden_solution = _extract_intent_and_golden(sample)
        print("\n" + "=" * 80)
        print(f"RUN {idx}/{len(samples)}")
        print("=" * 80)

        # Generate responses
        responses = generate_responses(llm, llm_tokenizer, intent, args.num_samples)
        
        # Get embeddings
        print("\nEncoding responses...")
        response_embeddings = get_embeddings(responses, tokenizer, embedding_model, device)
        
        print("Encoding golden solution...")
        golden_embedding = get_embeddings([golden_solution], tokenizer, embedding_model, device)[0]
        
        # Optimize weights
        optimal_weights, min_distance = optimize_weights(response_embeddings, golden_embedding)
        
        # Report
        metrics = report_results(
            responses,
            response_embeddings,
            golden_embedding,
            optimal_weights,
            min_distance,
            intent,
            golden_solution,
        )
        run_metrics.append(metrics)

    if run_metrics:
        avg_min = float(np.mean([m["min_distance"] for m in run_metrics]))
        avg_mean = float(np.mean([m["mean_distance"] for m in run_metrics]))
        avg_improve = float(np.mean([m["improvement_pct"] for m in run_metrics]))

        print("\n" + "=" * 80)
        print("AVERAGE PERFORMANCE")
        print("=" * 80)
        print(f"Average min distance: {avg_min:.6f}")
        print(f"Average mean distance: {avg_mean:.6f}")
        print(f"Average improvement over mean: {avg_improve:.2f}%")

        result_record = {
            "llm_model_name": args.llm_model_name,
            "sampling": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "num_samples": args.num_samples,
            },
            "avg_min_distance": avg_min,
            "avg_mean_distance": avg_mean,
        }
        _save_results_json(args.results_json, result_record)
        print(f"Saved results to: {args.results_json}")


if __name__ == "__main__":
    main()
