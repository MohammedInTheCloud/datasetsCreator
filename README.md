

# Book Dataset Generator

This tool helps you generate datasets compatible with the LongPage format from your own book texts. LongPage is a dataset designed to train AI models for long-form creative writing with structured reasoning.

original dataset https://huggingface.co/datasets/Pageshift-Entertainment/LongPage 
# Story behind this repo 

I wanted to put two new AI models to the test: Qwen3-Max as a planner and GLM-4.5 with CLAUDE CODE as an executor. While browsing r/LocalLLaMA, I came across a mention of a dataset called LongPage — designed to help models write full-length books with structured reasoning. The original poster didn’t share how they generated it, so I decided to build my own generator from scratch. I’m using Qwen3-Max to plan then handing that plan off to GLM-4.5 with CLAUDE CODE to execute and generate the actual content. This repo is the result of that experiment. 

## Getting Started

### What You Need

- Python 3.8 or higher
- An API key from a supported LLM provider (Groq, OpenAI, Ollama, or LM Studio)
- Your book files in plain text (.txt) format

### Setup

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Create your environment configuration file:

```bash
cp .env.example .env
```

3. Open the `.env` file and add your API credentials. For example, if you're using Groq:

```
LLM_API_KEY=your_actual_groq_api_key_here
LLM_MODEL=llama3-70b-8192
LLM_BASE_URL=https://api.groq.com/openai/v1
```

You can also configure it for local models like Ollama or LM Studio by changing the base URL and model name.

### Prepare Your Books

Create a folder named `books` and place your plain text book files inside it. The tool will automatically process all .txt files found in this directory.

### Run the Tool

To start processing your books, simply run:

```bash
python main.py
```

If you want to test the system first, you can limit it to process only a few books:

```bash
python main.py --max-books 5 --validate
```

This will process 5 books and validate the output format.

### Check the Results

After processing, you'll find your generated dataset in the `data/output/` folder. The main output file will be named something like `longpage_dataset_YYYYMMDD_HHMMSS.jsonl`.

Each line in this file represents one book, fully structured with metadata, chapter breakdowns, character information, and scene analysis — all formatted to match the LongPage specification.

## Configuration

You can customize the tool by editing the `.env` file:

- `LLM_API_KEY`: Your API key
- `LLM_MODEL`: The model you want to use (e.g., `llama3-70b-8192`, `gpt-4-turbo`)
- `LLM_BASE_URL`: The API endpoint
- `MAX_RETRIES`: How many times to retry if an API call fails
- `RATE_LIMIT_DELAY`: Delay between retries to avoid hitting rate limits

## Supported Providers

The tool works with several LLM providers:

- **Groq**: Fast cloud-based service. Sign up at console.groq.com.
- **Ollama**: Run models locally. Install from ollama.com.
- **LM Studio**: Another option for local models. Download from lmstudio.ai.
- **OpenAI**: Use GPT models with your official API key.

## Tips for Success

1. Start by testing with just a few books to make sure everything works.
2. Keep an eye on the `pipeline.log` file for any errors or warnings.
3. If you encounter rate limits, try reducing the number of books processed at once or increasing the delay between requests.
4. Make sure your book files are saved in UTF-8 encoding to avoid text processing errors.

## Troubleshooting

If you run into problems:

- Double-check that your API key and base URL are correct in the `.env` file.
- If you're using a local service like Ollama or LM Studio, make sure the application is running and the server is active.
- For memory issues, process fewer books at a time by using the `--max-books` option.

The system performs real analysis using actual LLM calls — no mock data is used. The output is designed to be directly compatible with the LongPage dataset format for training or fine-tuning language models.
