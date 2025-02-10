import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import os

class ImageCoherenceTester:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
    def load_image(self, image_path: str) -> Image:
        """Load image from local path using PIL."""
        return Image.open(image_path)
    
    def create_test_cases(self, image_dir: str = "images") -> List[Dict]:
        """Create test cases with varying degrees of relevance."""
        test_cases = [
            {
                'image_path': f"{image_dir}/cat.jpg",  # Local path to cat image
                'matching_text': "A cute cat sitting on a windowsill, watching birds outside.",
                'unrelated_text': "The economic impact of industrial revolution on urban development.",
                'somewhat_related': "Pets make great companions for people living alone.",
                'label': 'cat'
            },
            {
                'image_path': f"{image_dir}/car.jpg",  # Local path to car image
                'matching_text': "A red sports car parked in front of a modern building.",
                'unrelated_text': "Traditional recipes for baking sourdough bread.",
                'somewhat_related': "Modern transportation has revolutionized our lives.",
                'label': 'car'
            }
        ]
        return test_cases
    
    def compute_similarity_scores(self, image: Image, texts: List[str]) -> List[float]:
        """Compute CLIP similarity scores between an image and multiple texts."""
        inputs = self.processor(
            text=texts,
            images=[image] * len(texts),
            return_tensors="pt",
            padding=True
        )
        
        outputs = self.model(**inputs)
        scores = outputs.logits_per_image.detach().numpy()
        return scores[0].tolist()
    
    def run_test_suite(self) -> Dict:
        """Run complete test suite and return results."""
        test_cases = self.create_test_cases()
        results = {}
        
        for case in test_cases:
            try:
                # Load the image from local path
                image = self.load_image(case['image_path'])
                
                texts = [
                    case['matching_text'],
                    case['unrelated_text'],
                    case['somewhat_related']
                ]
                
                scores = self.compute_similarity_scores(image, texts)
                results[case['label']] = {
                    'matching_score': scores[0],
                    'unrelated_score': scores[1],
                    'somewhat_related_score': scores[2]
                }
                
            except Exception as e:
                print(f"Error processing test case {case['label']}: {str(e)}")
                
        return results
    
    def visualize_results(self, results: Dict, out_dir: str):
        """Create visualization of test results."""
        labels = list(results.keys())
        matching_scores = [results[label]['matching_score'] for label in labels]
        unrelated_scores = [results[label]['unrelated_score'] for label in labels]
        somewhat_scores = [results[label]['somewhat_related_score'] for label in labels]
        
        x = np.arange(len(labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width, matching_scores, width, label='Matching')
        ax.bar(x, somewhat_scores, width, label='Somewhat Related')
        ax.bar(x + width, unrelated_scores, width, label='Unrelated')
        
        ax.set_ylabel('Similarity Score')
        ax.set_title('CLIP Similarity Scores by Text Type')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        plt.tight_layout()
        
        # save figure to out dir
        fig_path = f"{out_dir}/results.png"
        fig.savefig(fig_path)
        
        return fig

def main():
    # Initialize tester
    tester = ImageCoherenceTester()
    
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Run tests
    print("Running image coherence tests...")
    results = tester.run_test_suite()
    
    # Print results
    print("\nTest Results:")
    for label, scores in results.items():
        print(f"\n{label.upper()}:")
        print(f"Matching text score: {scores['matching_score']:.3f}")
        print(f"Somewhat related score: {scores['somewhat_related_score']:.3f}")
        print(f"Unrelated text score: {scores['unrelated_score']:.3f}")
    
    # Create visualization
    fig = tester.visualize_results(results, out_dir)
    
    return results, fig

if __name__ == "__main__":
    main()