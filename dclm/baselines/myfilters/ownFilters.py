from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Callable
from core.factory_utils import factory_function
import torch
import json

import ray

from core.constants import CONTENT

import json
import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer


@ray.remote(num_gpus=1)
class RayWorker:
    def __init__(self, model_name: str, centroid_path: str):
        """Initialize model and centroid on the GPU."""
        self.model = SentenceTransformer(model_name).to("cuda")
        
        # Load centroid as a PyTorch tensor directly on the GPU
        with open(centroid_path, "r") as f:
            centroid_array = np.array(json.load(f))
        self.centroid = torch.tensor(centroid_array, dtype=torch.float32, device="cuda")

    def filter_page(self, page: Dict, threshold_mean: float, threshold_std: float, std_multiplier: float) -> List[Dict]:
        """Process a single page and filter based on distance to the centroid."""
        with torch.no_grad():
            embedding = self.model.encode(page[CONTENT], convert_to_tensor=True, device="cuda")

        # Compute distance using GPU tensor operations
        distance = torch.norm(embedding - self.centroid)

        return [page] if distance.item() <= threshold_mean + threshold_std * std_multiplier else []

@ray.remote(num_gpus=1)
class segmentRayWorker:
    
    def __init__(self, model_name: str):
        """Initialize model and centroid on the GPU."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name).to("cuda")
        
    
    def split_text_into_parts_with_overlap(self, text, n, overlap):
        if not text.strip():
            return []  # Handle empty text case
        
        words = text.split()
        total_words = len(words)
        n = max(1, int(n))  # Ensure n is at least 1
        avg = max(1, total_words // n)  # Prevent division errors
        overlap_size = int(overlap * float(avg))  # Ensure multiplication is valid

        parts = []
        for i in range(n):
            start = max(0, i * avg - overlap_size * (i > 0))
            end = min(total_words, (i + 1) * avg + overlap_size * (i < n - 1))
            parts.append(" ".join(words[start:end]))
        
        return parts


    def embed_texts(self, model, device, texts: List[str]) -> List[np.ndarray]:
        """
        Generates embeddings for a list of texts using the given model.
        """
        return [model.encode(text, device=device) for text in texts]

    def calculate_distances(self, embeddings: List[np.ndarray]) -> List[float]:
        """
        Calculates angles between consecutive embeddings in radians.
        """
        distances = []
        for doc_embeddings in embeddings:
            for i in range(len(doc_embeddings) - 1):
                norm_i = np.linalg.norm(doc_embeddings[i])
                norm_next = np.linalg.norm(doc_embeddings[i + 1])

                # Skip comparison if either embedding is a zero vector
                if norm_i == 0 or norm_next == 0:
                    distances.append(np.pi)  # Assign max possible angle (180 degrees)
                    continue

                cosine_similarity = np.dot(doc_embeddings[i], doc_embeddings[i + 1]) / (norm_i * norm_next)
                angle = np.arccos(np.clip(cosine_similarity, -1, 1))  # Ensure valid input range
                distances.append(abs(angle))
        
        return distances

    def filter_segmented_page(
        self,
        page: Dict,
        n_parts: int = 5,
        threshold_mean: float = 0.5,
        threshold_std: float = 0.1,
        std_multiplier: float = 1,
        ) -> List[Dict]:
        # Segment the page content and compute embeddings
        segments = self.split_text_into_parts_with_overlap(page[CONTENT], n_parts, 0.4)
        if not segments:
            return []  # Skip empty pages
        embeddings = self.embed_texts(self.model, self.device, segments)
        if not embeddings:
            return []
        # Calculate angles and statistics
        angles = self.calculate_distances(embeddings)
        if not angles:
            return []
        mean_angle = np.mean(angles)
        # std_angle = np.std(angles)

        # Filter based on thresholds
        if mean_angle <= threshold_mean + threshold_std * std_multiplier:
            return [page]
        else:
            return []


@factory_function
def embedding_angle_filter(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    n_parts: int = 5,
    threshold_mean: float = 0.5,
    threshold_std: float = 0.1,
    std_multiplier: float = 1,
) -> List[Dict]:
    """
    Filters pages based on the mean and standard deviation of angles between segment embeddings.

    Arguments:
    model_name -- The pre-defined sentence transformer model to use for embeddings.
                  Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
    n_parts -- Number of segments to split each page into. Defaults to 5.
    threshold_mean -- Maximum mean angle allowed for a page to pass. Defaults to 0.5.
    threshold_std -- Maximum standard deviation of angles allowed for a page to pass. Defaults to 0.1.

    Returns:
    A callable filter function that takes a page dictionary and returns the page
    if it passes the filter or an empty list if it doesn't.
    """
    # Load the pre-defined model and move to the appropriate device
    worker = segmentRayWorker.remote(model_name)

    def filter_fn(page: Dict) -> List[Dict]:
        """
        Filters a single page based on the mean and standard deviation of angles.
        """
        return ray.get(worker.filter_segmented_page.remote(page, threshold_mean, threshold_std, std_multiplier))

    return filter_fn



@factory_function
def embedding_centroid_filter(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    centroid_path: str = "centroid.json",
    threshold_mean: float = 0.5,
    threshold_std: float = 0.1,
    std_multiplier: float = 1,
) -> List[Dict]:
    """
    Factory function that initializes a Ray-based parallel filtering function.
    This ensures that the model and centroid are only loaded once.
    """
    worker = RayWorker.remote(model_name, centroid_path)

    def filter_fn(page: Dict) -> List[Dict]:
        return ray.get(worker.filter_page.remote(page, threshold_mean, threshold_std, std_multiplier))

    return filter_fn