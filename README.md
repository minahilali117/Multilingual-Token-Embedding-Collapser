# Multilingual Token Embedding Collapser

A comprehensive Python toolkit for processing multilingual text, collapsing subword tokens into word-level representations, and visualizing embeddings across different languages including Urdu, Arabic, Chinese, Spanish, and English.

## Features

- **Multilingual Support**: Handles tokenization for Urdu, Arabic, Chinese, Spanish, and English with language-specific patterns
- **Subword Token Collapsing**: Aggregates subword tokens (like "▁com", "puter") into coherent word-level embeddings
- **Multiple Aggregation Methods**: Support for mean, max, first, and last token aggregation strategies
- **Interactive Visualization**: t-SNE and PCA dimensionality reduction with interactive Plotly visualizations
- **Cross-Language Analysis**: Compare and analyze embeddings across different languages simultaneously

## Installation

```bash
pip install torch transformers scikit-learn plotly numpy
```

## Quick Start

```python
from multilingual_embedder import MultilingualEmbeddingCollapser

# Initialize the collapser
collapser = MultilingualEmbeddingCollapser(
    model_name="xlm-roberta-base",
    aggregation_method='mean',
    reduction_method='tsne'
)

# Example sentences in different languages
sentences = {
    'English': "The computer processes information quickly.",
    'Spanish': "La computadora procesa información rápidamente.",
    'Urdu': "کمپیوٹر معلومات کو تیزی سے پروسیس کرتا ہے۔",
    'Arabic': "يعالج الحاسوب المعلومات بسرعة وكفاءة.",
    'Chinese': "计算机快速高效地处理信息。"
}

# Analyze and visualize
fig, analysis = collapser.analyze_multilingual_text(sentences)
fig.show()
```

## Architecture

### 1. MultilingualTokenizer
**Tokenizer abstraction layer** that handles:
- Language detection based on Unicode character ranges
- XLM-RoBERTa tokenization with word alignment
- Language-specific tokenization patterns
- Subword token tracking and mapping

```python
tokenizer = MultilingualTokenizer("xlm-roberta-base")
tokens, word_ids, words = tokenizer.tokenize_with_alignment("Hello world")
```

### 2. EmbeddingAggregator
**Embedding aggregation logic** that provides:
- Subword embedding extraction from transformer models
- Multiple aggregation strategies (mean, max, first, last)
- Word-level embedding computation
- Cross-lingual embedding consistency

```python
aggregator = EmbeddingAggregator(tokenizer)
word_embeddings = aggregator.aggregate_word_embeddings(
    "Hello world", 
    aggregation_method='mean'
)
```

### 3. EmbeddingVisualizer
**Interactive 2D visualization** featuring:
- t-SNE and PCA dimensionality reduction
- Interactive Plotly scatter plots
- Language-specific color coding
- Hover information with subword details
- Customizable plot styling

```python
visualizer = EmbeddingVisualizer(reduction_method='tsne')
fig = visualizer.create_interactive_plot(word_embeddings)
```

## Language-Specific Handling

### Tokenization Patterns
- **Urdu**: `[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+` with Urdu-specific character detection
- **Arabic**: `[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+` 
- **Chinese**: `[\u4e00-\u9fff]+` for CJK ideographs
- **Spanish**: `[a-záéíóúñü]+` including accented characters
- **English**: `[a-z]+` basic Latin characters

### Language Detection
Automatic language detection based on:
- Unicode character range analysis
- Script-specific character patterns
- Language-specific diacritics and special characters

## Subword Token Handling

The system handles various subword tokenization schemes:

### XLM-RoBERTa Subwords
- **Prefix tokens**: `▁` indicates word boundaries
- **Continuation tokens**: No prefix for word continuations
- **Special tokens**: `<s>`, `</s>`, `<pad>`, `<unk>`

### Example Tokenization
```
Input: "computer"
Tokens: ["▁com", "puter"]
Aggregated: Single embedding for "computer"
```

## Aggregation Methods

### Mean Aggregation (Default)
```python
aggregated_embedding = torch.mean(subword_embeddings, dim=0)
```

### Max Pooling
```python
aggregated_embedding = torch.max(subword_embeddings, dim=0)[0]
```

### First/Last Token
```python
# First subword token
aggregated_embedding = subword_embeddings[0]
# Last subword token  
aggregated_embedding = subword_embeddings[-1]
```

## Visualization Features

### Interactive Elements
- **Hover Information**: Word, language, subword tokens, coordinates
- **Language Legends**: Toggle visibility by language
- **Zoom and Pan**: Interactive exploration of embedding space
- **Text Labels**: Words displayed directly on plot

### Customization Options
```python
# Different reduction methods
visualizer_tsne = EmbeddingVisualizer('tsne')
visualizer_pca = EmbeddingVisualizer('pca')

# Custom plot styling
fig.update_layout(
    title="Custom Embedding Visualization",
    width=1200,
    height=800
)
```

## API Reference

### MultilingualEmbeddingCollapser

#### `__init__(model_name, aggregation_method, reduction_method)`
- `model_name`: HuggingFace model identifier (default: "xlm-roberta-base")
- `aggregation_method`: 'mean', 'max', 'first', 'last' (default: 'mean')
- `reduction_method`: 'tsne', 'pca' (default: 'tsne')

#### `process_sentence(sentence)`
Process a single sentence and return word embeddings.

#### `process_multiple_sentences(sentences)`
Process multiple sentences from different languages.

#### `analyze_multilingual_text(sentences, show_individual_plots)`
Complete analysis pipeline with visualization and statistics.

### WordEmbedding (Data Class)
```python
@dataclass
class WordEmbedding:
    word: str                    # Original word
    embedding: np.ndarray        # Aggregated embedding vector
    language: str               # Detected language
    subword_tokens: List[str]   # Original subword tokens
    token_positions: List[int]  # Token positions in sequence
```

## Performance Considerations

### Memory Usage
- XLM-RoBERTa base model: ~560MB
- Embedding storage: ~768 dimensions per word
- Batch processing recommended for large datasets

### Speed Optimization
```python
# Process multiple sentences efficiently
results = collapser.process_multiple_sentences(sentences)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
collapser.tokenizer.model.to(device)
```

## Advanced Usage

### Custom Model Integration
```python
# Use different multilingual models
collapser = MultilingualEmbeddingCollapser(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

### Batch Processing
```python
# Process large datasets
def process_batch(sentences_batch):
    results = []
    for sentence in sentences_batch:
        embeddings = collapser.process_sentence(sentence)
        results.extend(embeddings)
    return results
```

### Export Embeddings
```python
# Save embeddings for later use
import pickle

word_embeddings = collapser.process_sentence("Your text here")
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(word_embeddings, f)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or use smaller model variants
2. **Language Detection**: Manually specify language if auto-detection fails
3. **Tokenization Misalignment**: Check text preprocessing and special characters

### Debug Mode
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{multilingual_embedder,
  title={Multilingual Token Embedding Collapser},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/multilingual-embedder}
}
```

## Acknowledgments

- HuggingFace Transformers library
- XLM-RoBERTa model by Facebook AI
- Plotly for interactive visualizations
- scikit-learn for dimensionality reduction
