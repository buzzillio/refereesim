# RefereeSim: AI Reviewer Evaluation Platform

## Overview

RefereeSim evaluates AI models' ability to detect errors in research papers using **REAL API calls**. The system generates synthetic papers with controlled errors and tests 11 working AI models from Cohere, Gemini, and Hyperbolic (NO OpenAI required).

## Features

✅ **Real API Integration** - No simulations, actual API calls to Cohere, Gemini, and Hyperbolic  
📊 **Confusion Matrices** - Terminal output and saved visualizations  
🤖 **11 Working AI Models** - Cohere, Gemini, Meta-Llama, DeepSeek (NO OpenAI required)  
📈 **Comprehensive Reports** - HTML reports, CSV data, performance plots  
🎯 **Controlled Evaluation** - Synthetic papers with known ground-truth errors  

## Quick Start

### Secure Setup (Environment Variables)
1. Download `run_refereesim.py`
2. Set environment variables:
   ```bash
   export COHERE_API_KEY=your_cohere_key
   export HYPERBOLIC_API_KEY=your_hyperbolic_key  
   export GENAI_API_KEY=your_gemini_key
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python run_refereesim.py`

## API Keys Required

- **Cohere**: Get from https://dashboard.cohere.ai/api-keys
- **Hyperbolic**: Get from https://hyperbolic.xyz/
- **Gemini**: Get from https://ai.google.dev/gemini-api/docs/api-key

## Output

The system generates:
- **Terminal Output**: Real-time confusion matrices for each model
- **HTML Report**: Comprehensive analysis with visualizations
- **CSV Files**: Raw performance data
- **Confusion Matrix Images**: Saved as PNG files
- **Performance Plots**: Model comparison charts

## Models Tested (11 Working Models)

### Cohere Models (2)
- command-a-03-2025
- command-r

### Gemini Models (2)  
- gemini-2.5-flash
- gemini-2.5-pro

### Hyperbolic Models (7)
- openai/gpt-oss-120b
- meta-llama/Meta-Llama-3.1-405B-Instruct
- meta-llama/Meta-Llama-3.1-70B-Instruct
- meta-llama/Meta-Llama-3.1-8B-Instruct
- deepseek-ai/DeepSeek-R1-0528
- deepseek-ai/DeepSeek-R1
- deepseek-ai/DeepSeek-V3

## Project Structure

```
RefereeSim_Download/
├── run_refereesim.py           # Secure main file (environment variables)
├── requirements.txt            # Dependencies
├── refereesim/                 # Core package
│   ├── generators/             # Paper generation
│   ├── reviewers/              # AI reviewer API integration
│   ├── scorers/                # Performance evaluation
│   ├── seeders/                # Error injection
│   └── utils/                  # Reporting and utilities
└── results/                    # Generated after running
    └── refereesim_YYYYMMDD_HHMMSS/
        ├── data/               # Generated papers
        ├── reviews/            # AI review results
        └── reports/            # HTML reports and plots
```

## Example Output

```
🏆 FINAL RESULTS:
📊 Papers Evaluated: 15
🤖 Models Tested: 11
🎯 Ground Truth Errors: 13
📝 Total Reviews: 165
🥇 Best Model: command-a-03-2025
   F1-Score: 0.847

📊 CONFUSION MATRIX - 🏆 WINNER: command-a-03-2025
==================================================
                 Predicted
               Error  No Error
Actual Error     38        7    (TP)  (FN)
   No Error      12      213    (FP)  (TN)
--------------------------------------------------
Accuracy:  0.930 (251/270)
Precision: 0.760 (38/50)
Recall:    0.844 (38/45)
F1-Score:  0.800
==================================================
```

## Stanford Agents4Science 2025

This platform was designed for the Stanford Agents4Science 2025 conference to evaluate AI reviewers on scientific paper error detection using real API calls and controlled synthetic datasets.