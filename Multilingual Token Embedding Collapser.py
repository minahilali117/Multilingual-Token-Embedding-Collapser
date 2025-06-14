import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class WordEmbedding:
    """Data class to store word and its embedding"""
    word: str
    embedding: np.ndarray
    language: str
    subword_tokens: List[str]
    token_positions: List[int]

class MultilingualTokenizer:
    """Abstraction layer for multilingual tokenization"""
    
    def __init__(self, model_name: str = "xlm-roberta-base"):
        """
        Initialize the multilingual tokenizer
        
        Args:
            model_name: HuggingFace model name (default: xlm-roberta-base)
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Language-specific patterns for word boundary detection
        self.language_patterns = {
            'urdu': r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+',
            'arabic': r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+',
            'chinese': r'[\u4e00-\u9fff]+',
            'spanish': r'[a-záéíóúñü]+',
            'english': r'[a-z]+',
            'default': r'\S+'
        }
    
    def detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns"""
        text_lower = text.lower()
        
        # Check for specific language patterns
        if re.search(r'[\u0600-\u06FF]', text):
            # Could be Arabic or Urdu, check for Urdu-specific characters
            if re.search(r'[\u0679\u067E\u0686\u0688\u0691\u06BA\u06BE\u06C1\u06C3\u06CC\u06D2]', text):
                return 'urdu'
            return 'arabic'
        elif re.search(r'[\u4e00-\u9fff]', text):
            return 'chinese'
        elif re.search(r'[ñáéíóúü]', text_lower):
            return 'spanish'
        elif re.search(r'[a-z]', text_lower):
            return 'english'
        
        return 'default'
    
    def tokenize_with_alignment(self, text: str) -> Tuple[List[str], List[int], List[str]]:
        """
        Tokenize text and return tokens with word alignment
        
        Returns:
            tokens: List of subword tokens
            word_ids: List of word IDs for each token
            words: List of original words
        """
        # Tokenize with return_offsets_mapping for alignment
        encoded = self.tokenizer(
            text, 
            return_tensors="pt", 
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        offsets = encoded['offset_mapping'][0]
        
        # Extract words and create alignment
        language = self.detect_language(text)
        pattern = self.language_patterns.get(language, self.language_patterns['default'])
        
        words = []
        word_boundaries = []
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            words.append(match.group())
            word_boundaries.append((match.start(), match.end()))
        
        # Create word_ids for each token
        word_ids = []
        for i, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:  # Special tokens
                word_ids.append(None)
                continue
                
            # Find which word this token belongs to
            token_word_id = None
            for word_idx, (word_start, word_end) in enumerate(word_boundaries):
                if start >= word_start and end <= word_end:
                    token_word_id = word_idx
                    break
                elif start < word_end and end > word_start:  # Overlapping
                    token_word_id = word_idx
                    break
            
            word_ids.append(token_word_id)
        
        return tokens, word_ids, words

class EmbeddingAggregator:
    """Handles embedding extraction and aggregation logic"""
    
    def __init__(self, tokenizer: MultilingualTokenizer):
        self.tokenizer = tokenizer
    
    def extract_embeddings(self, text: str) -> torch.Tensor:
        """Extract embeddings from the model"""
        inputs = self.tokenizer.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.tokenizer.model(**inputs)
            # Use last hidden state
            embeddings = outputs.last_hidden_state[0]  # Remove batch dimension
        
        return embeddings
    
    def aggregate_word_embeddings(self, text: str, aggregation_method: str = 'mean') -> List[WordEmbedding]:
        """
        Aggregate subword embeddings into word-level embeddings
        
        Args:
            text: Input text
            aggregation_method: 'mean', 'max', 'first', 'last'
        
        Returns:
            List of WordEmbedding objects
        """
        # Get tokens, word alignments, and words
        tokens, word_ids, words = self.tokenizer.tokenize_with_alignment(text)
        
        # Extract embeddings
        embeddings = self.extract_embeddings(text)
        
        # Detect language
        language = self.tokenizer.detect_language(text)
        
        # Group tokens by word
        word_embeddings = []
        
        for word_idx, word in enumerate(words):
            # Find all tokens belonging to this word
            token_indices = [i for i, wid in enumerate(word_ids) if wid == word_idx]
            
            if not token_indices:
                continue
            
            # Get embeddings for these tokens
            word_token_embeddings = embeddings[token_indices]
            word_tokens = [tokens[i] for i in token_indices]
            
            # Aggregate embeddings
            if aggregation_method == 'mean':
                aggregated_embedding = torch.mean(word_token_embeddings, dim=0)
            elif aggregation_method == 'max':
                aggregated_embedding = torch.max(word_token_embeddings, dim=0)[0]
            elif aggregation_method == 'first':
                aggregated_embedding = word_token_embeddings[0]
            elif aggregation_method == 'last':
                aggregated_embedding = word_token_embeddings[-1]
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation_method}")
            
            word_embeddings.append(WordEmbedding(
                word=word,
                embedding=aggregated_embedding.numpy(),
                language=language,
                subword_tokens=word_tokens,
                token_positions=token_indices
            ))
        
        return word_embeddings

class EmbeddingVisualizer:
    """Interactive 2D visualization of embeddings"""
    
    def __init__(self, reduction_method: str = 'tsne'):
        """
        Initialize visualizer
        
        Args:
            reduction_method: 'tsne' or 'pca'
        """
        self.reduction_method = reduction_method
    
    def reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduce embedding dimensions for visualization"""
        if self.reduction_method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(embeddings)-1))
        elif self.reduction_method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown reduction method: {self.reduction_method}")
        
        return reducer.fit_transform(embeddings)
    
    def create_interactive_plot(self, word_embeddings: List[WordEmbedding], 
                              title: str = "Multilingual Word Embeddings") -> go.Figure:
        """Create interactive plotly visualization"""
        if not word_embeddings:
            raise ValueError("No word embeddings provided")
        
        # Extract embeddings and metadata
        embeddings = np.array([we.embedding for we in word_embeddings])
        words = [we.word for we in word_embeddings]
        languages = [we.language for we in word_embeddings]
        subword_info = [f"Subwords: {', '.join(we.subword_tokens)}" for we in word_embeddings]
        
        # Reduce dimensions
        reduced_embeddings = self.reduce_dimensions(embeddings)
        
        # Create color mapping for languages
        unique_languages = list(set(languages))
        colors = px.colors.qualitative.Set1[:len(unique_languages)]
        color_map = dict(zip(unique_languages, colors))
        
        # Create the plot
        fig = go.Figure()
        
        for lang in unique_languages:
            lang_indices = [i for i, l in enumerate(languages) if l == lang]
            lang_words = [words[i] for i in lang_indices]
            lang_embeddings = reduced_embeddings[lang_indices]
            lang_subwords = [subword_info[i] for i in lang_indices]
            
            fig.add_trace(go.Scatter(
                x=lang_embeddings[:, 0],
                y=lang_embeddings[:, 1],
                mode='markers+text',
                text=lang_words,
                textposition='top center',
                name=f'{lang.capitalize()} ({len(lang_indices)} words)',
                marker=dict(
                    color=color_map[lang],
                    size=10,
                    line=dict(width=1, color='white')
                ),
                hovertemplate='<b>%{text}</b><br>' +
                             f'Language: {lang}<br>' +
                             'X: %{x:.2f}<br>' +
                             'Y: %{y:.2f}<br>' +
                             '%{customdata}<extra></extra>',
                customdata=lang_subwords
            ))
        
        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>Method: {self.reduction_method.upper()}</sub>",
                x=0.5
            ),
            xaxis_title=f"{self.reduction_method.upper()} Component 1",
            yaxis_title=f"{self.reduction_method.upper()} Component 2",
            showlegend=True,
            hovermode='closest',
            width=900,
            height=700,
            font=dict(size=12)
        )
        
        return fig

class MultilingualEmbeddingCollapser:
    """Main class that orchestrates the entire pipeline"""
    
    def __init__(self, model_name: str = "xlm-roberta-base", 
                 aggregation_method: str = 'mean',
                 reduction_method: str = 'tsne'):
        """
        Initialize the multilingual embedding collapser
        
        Args:
            model_name: HuggingFace model name
            aggregation_method: How to aggregate subword embeddings
            reduction_method: Dimensionality reduction method for visualization
        """
        self.tokenizer = MultilingualTokenizer(model_name)
        self.aggregator = EmbeddingAggregator(self.tokenizer)
        self.visualizer = EmbeddingVisualizer(reduction_method)
        self.aggregation_method = aggregation_method
        
    def process_sentence(self, sentence: str) -> List[WordEmbedding]:
        """Process a single sentence and return word embeddings"""
        return self.aggregator.aggregate_word_embeddings(sentence, self.aggregation_method)
    
    def process_multiple_sentences(self, sentences: Dict[str, str]) -> Dict[str, List[WordEmbedding]]:
        """
        Process multiple sentences from different languages
        
        Args:
            sentences: Dictionary with language names as keys and sentences as values
        
        Returns:
            Dictionary with language names as keys and word embeddings as values
        """
        results = {}
        for lang_name, sentence in sentences.items():
            results[lang_name] = self.process_sentence(sentence)
        
        return results
    
    def visualize_embeddings(self, word_embeddings: List[WordEmbedding], 
                           title: str = "Multilingual Word Embeddings") -> go.Figure:
        """Visualize word embeddings"""
        return self.visualizer.create_interactive_plot(word_embeddings, title)
    
    def analyze_multilingual_text(self, sentences: Dict[str, str], 
                                show_individual_plots: bool = False) -> Tuple[go.Figure, Dict]:
        """
        Complete analysis pipeline for multilingual text
        
        Args:
            sentences: Dictionary with language names as keys and sentences as values
            show_individual_plots: Whether to create individual plots for each language
        
        Returns:
            Combined plot figure and analysis results
        """
        # Process all sentences
        results = self.process_multiple_sentences(sentences)
        
        # Combine all word embeddings
        all_embeddings = []
        for lang_name, embeddings in results.items():
            all_embeddings.extend(embeddings)
        
        # Create combined visualization
        combined_fig = self.visualize_embeddings(
            all_embeddings, 
            "Multilingual Word Embeddings - Combined View"
        )
        
        # Create analysis summary
        analysis = {
            'total_words': len(all_embeddings),
            'languages_detected': len(set(we.language for we in all_embeddings)),
            'language_distribution': {},
            'subword_stats': {}
        }
        
        for lang_name, embeddings in results.items():
            detected_lang = embeddings[0].language if embeddings else 'unknown'
            analysis['language_distribution'][lang_name] = {
                'word_count': len(embeddings),
                'detected_language': detected_lang,
                'avg_subwords_per_word': np.mean([len(we.subword_tokens) for we in embeddings]) if embeddings else 0
            }
            
            # Subword statistics
            for we in embeddings:
                num_subwords = len(we.subword_tokens)
                if num_subwords not in analysis['subword_stats']:
                    analysis['subword_stats'][num_subwords] = 0
                analysis['subword_stats'][num_subwords] += 1
        
        return combined_fig, analysis

# Example usage and testing functions
def save_plot_as_html(fig, filename="multilingual_embeddings.html"):
    """Save the plot as an HTML file that can be opened in a browser"""
    fig.write_html(filename)
    print(f"Plot saved as {filename}")
    print(f"Open this file in your web browser to view the interactive plot")

def display_plot_inline():
    """Display plot inline for Jupyter notebooks"""
    try:
        import plotly.offline as pyo
        pyo.init_notebook_mode(connected=True)
        return True
    except ImportError:
        return False

def run_example():
    """Run the complete example with proper plot handling"""
    print("Initializing Multilingual Embedding Collapser...")
    
    # Initialize the collapser
    collapser = MultilingualEmbeddingCollapser(
        model_name="xlm-roberta-base",
        aggregation_method='mean',
        reduction_method='tsne'
    )
    
    # Example sentences in different languages
    test_sentences = {
        'English': "The computer processes information quickly and efficiently.",
        'Spanish': "La computadora procesa información rápidamente y eficientemente.",
        'Urdu': "کمپیوٹر معلومات کو تیزی سے اور مؤثر طریقے سے پروسیس کرتا ہے۔",
        'Arabic': "يعالج الحاسوب المعلومات بسرعة وكفاءة.",
        'Chinese': "计算机快速高效地处理信息。"
    }
    
    print("Processing multilingual sentences...")
    
    # Analyze the text
    fig, analysis = collapser.analyze_multilingual_text(test_sentences)
    
    print("\nAnalysis Results:")
    print(f"Total words processed: {analysis['total_words']}")
    print(f"Languages detected: {analysis['languages_detected']}")
    
    print("\nLanguage Distribution:")
    for lang, stats in analysis['language_distribution'].items():
        print(f"  {lang}: {stats['word_count']} words, "
              f"detected as '{stats['detected_language']}', "
              f"avg {stats['avg_subwords_per_word']:.1f} subwords/word")
    
    print("\nSubword Token Distribution:")
    for num_subwords, count in sorted(analysis['subword_stats'].items()):
        print(f"  {num_subwords} subwords: {count} words")
    
    print("\nExample of processed word embeddings:")
    english_embeddings = collapser.process_sentence(test_sentences['English'])
    for we in english_embeddings[:3]:  # Show first 3 words
        print(f"Word: '{we.word}', Subwords: {we.subword_tokens}, "
              f"Embedding shape: {we.embedding.shape}, Language: {we.language}")
    
    # Handle plot display based on environment
    print("\n" + "="*50)
    print("PLOT DISPLAY OPTIONS:")
    print("="*50)
    
    # Try to display inline first (for Jupyter)
    if display_plot_inline():
        print("Displaying plot inline (Jupyter notebook detected)")
        fig.show()
    else:
        print("Jupyter notebook not detected.")
        
        # Save as HTML file
        save_plot_as_html(fig, "multilingual_embeddings.html")
        
        # Try to show in browser
        try:
            fig.show()
            print("Attempting to open plot in default browser...")
        except Exception as e:
            print(f"Could not open plot automatically: {e}")
            print("Please open 'multilingual_embeddings.html' in your web browser")
    
    return fig, analysis

# Alternative display methods
def create_static_plot(word_embeddings, title="Multilingual Word Embeddings"):
    """Create a static matplotlib plot as fallback"""
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        # Extract data
        embeddings = np.array([we.embedding for we in word_embeddings])
        words = [we.word for we in word_embeddings]
        languages = [we.language for we in word_embeddings]
        
        # Reduce dimensions
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        reduced_embeddings = tsne.fit_transform(embeddings)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color map for languages
        unique_languages = list(set(languages))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_languages)))
        color_map = dict(zip(unique_languages, colors))
        
        # Plot points
        for lang in unique_languages:
            lang_indices = [i for i, l in enumerate(languages) if l == lang]
            lang_embeddings = reduced_embeddings[lang_indices]
            lang_words = [words[i] for i in lang_indices]
            
            ax.scatter(lang_embeddings[:, 0], lang_embeddings[:, 1], 
                      c=[color_map[lang]], label=f'{lang.capitalize()} ({len(lang_indices)} words)',
                      s=60, alpha=0.7)
            
            # Add word labels
            for i, word in enumerate(lang_words):
                ax.annotate(word, (lang_embeddings[i, 0], lang_embeddings[i, 1]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multilingual_embeddings_static.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Static plot saved as 'multilingual_embeddings_static.png'")
        
    except ImportError:
        print("Matplotlib not available for static plots")
    except Exception as e:
        print(f"Error creating static plot: {e}")

if __name__ == "__main__":
    # Run the example
    fig, analysis = run_example()
    
    # If interactive plot didn't work, try static plot
    print("\nIf the interactive plot didn't display properly, creating static plot...")
    try:
        # Get all embeddings for static plot
        collapser = MultilingualEmbeddingCollapser()
        test_sentences = {
            'English': "The computer processes information quickly and efficiently.",
            'Spanish': "La computadora procesa información rápidamente y eficientemente.",
            'Urdu': "کمپیوٹر معلومات کو تیزی سے اور مؤثر طریقے سے پروسیس کرتا ہے۔",
            'Arabic': "يعالج الحاسوب المعلومات بسرعة وكفاءة.",
            'Chinese': "计算机快速高效地处理信息。"
        }
        
        all_embeddings = []
        for sentence in test_sentences.values():
            all_embeddings.extend(collapser.process_sentence(sentence))
        
        create_static_plot(all_embeddings)
        
    except Exception as e:
        print(f"Static plot creation failed: {e}")
        print("\nTo view the visualization:")
        print("1. Install required packages: pip install plotly matplotlib")
        print("2. Open 'multilingual_embeddings.html' in your web browser")
        print("3. Or run in Jupyter notebook for inline display")