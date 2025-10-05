# Quick Start Guide

## 🚀 Getting Started

The LongPage Dataset Generator is now fully implemented.

### OVERVIEW

● This is a comprehensive LongPage Dataset Generator - an enterprise-grade production system for converting book texts into
  structured datasets designed to train AI models. Here's my analysis:

  Architecture Overview

  Purpose: Transforms raw book texts into structured LongPage-format datasets containing:
  - Chapter-level scene analysis with embeddings
  - Character archetypes and development arcs
  - World-building rules and story structure analysis
  - Synthetic user prompts for AI training

  Pipeline Flow:
  1. Preprocessing (src/utils/preprocessor.py) - Text cleaning, chapter detection, Gutenberg boilerplate removal
  2. Scene Segmentation (src/agents/scene_agent.py) - Rule-based + LLM-validated scene boundary detection
  3. Scene Analysis (src/agents/scene_analysis_agent.py) - Embedding generation and narrative analysis
  4. Chapter Summarization (src/agents/chapter_agent.py) - Multi-scene synthesis
  5. Book-level Metadata (src/agents/book_agent.py) - World rules, characters, story arcs, writing style
  6. Synthetic Generation (src/agents/synthetic_prompt_generator.py) - Professional-grade client briefs

  Technical Implementation

  LLM Integration: Uses Groq API with Llama3-70B for all analysis tasks, includes:
  - Rate limiting with exponential backoff
  - Think-tag stripping for model outputs
  - JSON parsing with multiple fallback strategies
  - Robust error handling

  Text Processing:
  - Unicode normalization and encoding fixes
  - Table of contents detection and removal
  - Multiple chapter detection regex patterns with fallbacks
  - Scene boundary detection using narrative cues

  Output Format: Generates JSON/JSONL files with comprehensive metadata:
  {
    "use_prompt": "synthetic client brief",
    "book_highlight": "concise hook",
    "book_title": "extracted title",
    "world_rules": ["fundamental world differences"],
    "story_arcs": ["major plotlines"],
    "character_archetypes": {"character": "archetype"},
    "book_chapters": {
      "chapter_title": {
        "embedding_space": {...},
        "scene_breakdown": [...]
      }
    }
  }

  Code Quality Assessment

  Strengths:
  - Comprehensive logging throughout pipeline
  - Multiple fallback strategies for each processing step
  - Clean separation of concerns between agents
  - Production-ready error handling
  - Enterprise-grade metadata extraction

  Areas for Consideration:
  - Heavy reliance on external LLM API (cost/latency implications)
  - Complex multi-stage processing (potential bottlenecks)
  - Windows-specific hardcoded paths in some places

  This is clearly designed for generating high-quality training data for narrative AI systems, with sophisticated analysis that
  goes far beyond simple text processing. The synthetic prompt generation is particularly impressive - it creates realistic client
  briefs that capture the essence of each book.

### 1. Prerequisites

- Python 3.8+
- Groq API key (sign up at https://console.groq.com)
- Book files in .txt format

### 2. Setup

```bash
# Navigate to the project directory
cd D:\local\tahdari\FINALAPP\ien\rag\longpage

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env

# Edit .env with your LLM provider settings
# For Groq (default):
# LLM_API_KEY=your_groq_key_here
# LLM_MODEL=llama3-70b-8192
# LLM_BASE_URL=https://api.groq.com/openai/v1

# For local Ollama:
# LLM_API_KEY=any-value
# LLM_MODEL=llama3:70b
# LLM_BASE_URL=http://localhost:11434/v1
```

### 3. Prepare Your Books

Place your book files (.txt format) in the books directory:

```bash
# Create books directory if it doesn't exist
mkdir -p books

# Copy your books here
# Example: cp /path/to/books/*.txt books/
```

### 4. Run the Pipeline

```bash
# Test with 5 books first
python main.py --max-books 5 --validate

# Process all books
python main.py

# Custom options
python main.py --max-books 10 --output-format json --validate
```

### 5. Check Results

Output files are saved in `data/output/` with timestamps:
- `longpage_dataset_YYYYMMDD_HHMMSS.jsonl` - Main dataset
- `pipeline.log` - Processing log

## 📊 Output Format

Each line in the JSONL file contains a complete book object with:

```json
{
  "use_prompt": "Generated synthetic user prompt...",
  "book_highlight": "Concise book summary...",
  "book_title": "Book Title",
  "book_tags": ["fiction", "adventure"],
  "book_archetype": "Adventure Fiction",
  "world_rules": ["Rule 1", "Rule 2"],
  "story_arcs": ["Arc 1", "Arc 2"],
  "character_archetypes": {"Character": "Archetype"},
  "writing_style": ["Style 1", "Style 2"],
  "book_characters": {
    "main_characters": ["Char1", "Char2"],
    "side_characters": ["Char3", "Char4"]
  },
  "book_chapters": {
    "Chapter 1": {
      "chapter": "Full chapter text...",
      "embedding_space": {"action": 45, "dialog": 30, ...},
      "chapter_summary": ["Summary point 1", "Summary point 2"],
      "scene_breakdown": [...]
    }
  }
}
```

## 🧪 Testing

Run the test suite to verify everything works:

```bash
python test_system.py
```

This will:
1. Process a single test book
2. Run batch processing on 3 books
3. Validate the output format
4. Show a summary of results

## 🔧 Configuration

### Environment Variables

Edit these values in your `.env` file:

- `LLM_API_KEY`: Your API key for the LLM provider
- `LLM_MODEL`: Model name to use (e.g., llama3-70b-8192, gpt-4-turbo)
- `LLM_BASE_URL`: Base URL for the API
- `MAX_RETRIES`: Number of API retry attempts (default: 3)
- `RATE_LIMIT_DELAY`: Delay between retries in seconds (default: 1)

### Supported Providers

- **Groq** (default): Fast cloud-based inference
- **Ollama**: Local models via http://localhost:11434/v1
- **LM Studio**: Local models via http://localhost:1234/v1  
- **OpenAI**: GPT models via official API
- **Any OpenAI-compatible API**: Just change the base URL and model

### Code Configuration

For path configurations, edit these values when initializing the pipeline:

```python
pipeline = BookPipeline(
    books_dir="/path/to/your/books",
    output_dir="/path/to/output"
)
```

## 📈 Production Tips

1. **Start Small**: Test with 5-10 books first
2. **Monitor Logs**: Check `pipeline.log` for errors
3. **Validate Output**: Use `--validate` flag
4. **Rate Limits**: Groq has generous limits, but monitor usage
5. **Batch Size**: Adjust `--max-books` based on your needs

## 🎯 Next Steps

1. Add your actual book collection
2. Run the full pipeline
3. Analyze the generated datasets
4. Fine-tune prompts if needed
5. Integrate with your downstream applications

## 🆘 Troubleshooting

### Common Issues

- **API Key Errors**: Check your .env file and variable names
- **Rate Limits**: Reduce batch size or increase RATE_LIMIT_DELAY
- **Memory Issues**: Process books in smaller batches
- **Encoding Errors**: Ensure books are UTF-8 encoded

### Provider-Specific Tips

**Groq:**
- Sign up at https://console.groq.com
- Free tier has generous limits
- Models: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768

**Ollama (Local):**
- Install Ollama from https://ollama.com
- Pull models: `ollama pull llama3:70b`
- Base URL: http://localhost:11434/v1
- API key can be any value

**LM Studio (Local):**
- Download from https://lmstudio.ai
- Load your preferred model
- Check the server tab for correct port (usually 1234)
- Base URL: http://localhost:1234/v1

**OpenAI:**
- Use your official API key
- Models: gpt-4, gpt-4-turbo, gpt-3.5-turbo
- Higher cost but excellent quality

## 📝 Notes

- The system uses NO mock or fallback data
- All analysis is performed by actual LLM calls
- The output format exactly matches LongPage schema
- Enterprise-grade error handling and logging included

Happy generating! 🎉