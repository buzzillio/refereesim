# 🚀 RefereeSim: Ready to Run!

## Two Ways to Run:

### 🔑 With Your Own API Keys (Recommended)
```bash
python run_refereesim_NO_KEYS.py
```
**First time setup:**
1. Edit `run_refereesim_NO_KEYS.py` 
2. Find the `setup_environment()` function
3. Replace the placeholder keys with your real API keys

### 🏃‍♂️ Quick Test (Embedded Keys)
```bash  
python run_refereesim.py
```
**Note:** Contains pre-embedded keys - only use for testing

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🎯 What You'll Get:
- Real-time confusion matrices in terminal
- HTML report with full analysis  
- Performance comparison across 18 AI models
- Saved confusion matrix images
- CSV data files for further analysis

**Expected runtime:** 10-15 minutes for complete evaluation

## 🤖 Models Tested:
- **OpenAI**: GPT-5, GPT-4.1, GPT-4o series (9 models)
- **Cohere**: command-a-03-2025, command-r (2 models)  
- **Hyperbolic**: Meta-Llama, DeepSeek, GPT-OSS (7 models)

Total: **18 latest AI models** with **real API calls**