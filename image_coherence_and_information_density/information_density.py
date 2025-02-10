import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class InformationDensityChecker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
    def compute_sentence_embeddings(self, text):
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 3]
        embeddings = []
        
        for sent in sentences:
            inputs = self.tokenizer(sent, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(embedding)
            
        return sentences, embeddings
    
    def compute_similarity_matrix(self, embeddings):
        n = len(embeddings)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                sim = torch.cosine_similarity(embeddings[i], embeddings[j], dim=1)
                sim_matrix[i, j] = sim.item()
                
        return sim_matrix
    
    def analyze_text(self, text):
        sentences, embeddings = self.compute_sentence_embeddings(text)
        sim_matrix = self.compute_similarity_matrix(embeddings)
        
        # Calculate density metrics
        token_count = len(self.tokenizer.encode(text))
        avg_similarity = (sim_matrix.sum() - len(sentences)) / (len(sentences) * len(sentences) - len(sentences))
        unique_info = np.sum(sim_matrix < 0.8) / 2  # Count pairs below threshold
        density_score = unique_info / token_count
        
        return {
            'sentences': sentences,
            'similarity_matrix': sim_matrix,
            'token_count': token_count,
            'avg_similarity': avg_similarity,
            'density_score': density_score
        }

def visualize_comparison(results_concise, results_verbose, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot similarity matrices
    sns.heatmap(results_concise['similarity_matrix'], ax=ax1, cmap='YlOrRd')
    ax1.set_title('Concise Text\nSentence Similarities')
    
    sns.heatmap(results_verbose['similarity_matrix'], ax=ax2, cmap='YlOrRd')
    ax2.set_title('Verbose Text\nSentence Similarities')
    
    plt.tight_layout()
    
    # Save figure
    fig_path = f"{out_dir}/similarity_comparison.png"
    fig.savefig(fig_path)
    
    return fig

def main():
    # Test cases
    concise_text = """Machine learning models learn patterns from data. They use these patterns to make predictions. Good models generalize well to new data. Training requires careful validation to avoid overfitting."""
    
    verbose_text = """Machine learning is a fascinating field that involves teaching computers to learn from data. As you might imagine, these incredible systems analyze data to find patterns. By looking at these patterns in the data, which is really quite remarkable, they can make predictions about new information. It's worth noting that when we train these amazing models, which is a really interesting process, we want them to work well on new data. The training process, which is super important and really crucial to get right, needs to be carefully validated to make sure we don't overfit the model, which would be problematic and not ideal."""
    
    # Initialize checker
    checker = InformationDensityChecker()
    
    # Create output directory
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Analyze texts
    results_concise = checker.analyze_text(concise_text)
    results_verbose = checker.analyze_text(verbose_text)
    
    # Print results
    print("\nConcise Text Analysis:")
    print(f"Token count: {results_concise['token_count']}")
    print(f"Average similarity: {results_concise['avg_similarity']:.3f}")
    print(f"Density score: {results_concise['density_score']:.3f}")
    
    print("\nVerbose Text Analysis:")
    print(f"Token count: {results_verbose['token_count']}")
    print(f"Average similarity: {results_verbose['avg_similarity']:.3f}")
    print(f"Density score: {results_verbose['density_score']:.3f}")
    
    # Create visualization
    fig = visualize_comparison(results_concise, results_verbose, out_dir)
    
    return results_concise, results_verbose, fig

if __name__ == "__main__":
    main()