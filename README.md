# Ollama Copilot Fixer 🔧🤖

**Fix and enable custom GGUF models (including sharded files) from HuggingFace/Unsloth for GitHub Copilot with full Tool support**

GitHub Copilot's local model integration requires the "Tool" capability to function properly. This PowerShell script solves multiple common issues:

1. ✅ **Sharded GGUF files** - Automatically detects and merges split models
2. ✅ **Missing Tool capability** - Applies proper templates for Copilot compatibility  
3. ✅ **HuggingFace downloads** - Downloads models directly from HuggingFace repos
4. ✅ **Local file support** - Works with files already on your system

---

## 🎯 Problems This Solves

### Problem 1: Sharded GGUF Files
Large models on HuggingFace are often split into multiple files (`model-00001-of-00005.gguf`, etc.). Ollama cannot load these directly—they must be merged first.

### Problem 2: Missing Tool Capability
Custom GGUF imports to Ollama only show "Inference" capability by default. GitHub Copilot requires "Tool" capability to function. 

### Problem 3: Manual Downloads
Downloading models from HuggingFace and configuring them manually is tedious and error-prone.

**This script fixes all three issues automatically. ** 🚀

---

## ✨ Features

- 🔗 **Download from HuggingFace** - Provide a repo URL or ID
- 📁 **Local file support** - Use GGUF files already on disk
- 🧩 **Auto-merge sharded files** - Detects and combines split models
- 🔍 **Auto-detect architecture** (Llama 3, Mistral, Phi-3, Gemma 2, Nemotron)
- 🛠️ **Apply tool templates** automatically
- ✅ **Validate and test** models after setup
- 🎨 **Color-coded output** for easy troubleshooting
- 🧹 **Automatic cleanup** of temporary files

---

## 📋 Prerequisites

- **Windows** with PowerShell 5.1+ (or PowerShell Core 7+)
- **[Ollama](https://ollama.ai/download)** installed and running
- **[llama.cpp](https://github.com/ggerganov/llama.cpp/releases)** (for merging sharded files - script can auto-download)
- **[HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)** (optional, for HF downloads)
- **VS Code** with GitHub Copilot extension

---

## 🚀 Quick Start

### Install HuggingFace CLI (Optional but Recommended)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

### Usage Examples

#### 1. Download from HuggingFace

```powershell
.\Setup-OllamaToolModel.ps1 -ModelSource "unsloth/Llama-3.2-3B-Instruct-GGUF" -ModelName "llama3-copilot"
```

#### 2. Use a Local GGUF File

```powershell
.\Setup-OllamaToolModel.ps1 -ModelSource "C:\Models\nemotron-nano-Q4_K_M.gguf"
```

#### 3. Merge Sharded Files and Configure

```powershell
.\Setup-OllamaToolModel.ps1 -ModelSource "C:\Models\llama-405b-00001-of-00008.gguf" -ModelName "llama-405b"
```

#### 4. Download Specific Quantization from HuggingFace

```powershell
.\Setup-OllamaToolModel.ps1 -ModelSource "bartowski/Llama-3.3-70B-Instruct-GGUF" -QuantizationType "Q4_K_M"
```

---

## 📖 Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `ModelSource` | ✅ Yes | - | Local path, HF repo ID, or HF URL |
| `ModelName` | ❌ No | Auto-generated | Custom name for the model in Ollama |
| `Architecture` | ❌ No | `auto` | Force architecture:  `llama3`, `mistral`, `phi3`, `gemma2` |
| `ContextLength` | ❌ No | `8192` | Maximum context window size |
| `Temperature` | ❌ No | `0.7` | Default sampling temperature |
| `QuantizationType` | ❌ No | First found | Specific quant (e.g., `Q4_K_M`, `Q5_K_M`) |
| `LlamaCppPath` | ❌ No | Auto-detect | Path to llama.cpp installation |
| `KeepDownloads` | ❌ No | `false` | Keep downloaded files after import |

---

## 🏗️ Supported Architectures

- ✅ **Llama 3 / 3.1 / 3.2 / 3.3** (including Nemotron variants)
- ✅ **Mistral / Mixtral**
- ✅ **Phi-3 / Phi-4**
- ✅ **Gemma 2**
- ✅ **Qwen 2 / 2.5**

---

## 🔧 How It Works

### For Local Files: 
1. Detects if file is part of a sharded set
2. Merges shards using llama.cpp's `gguf-split` tool
3. Applies architecture-specific tool template
4. Creates model in Ollama with Tool capability

### For HuggingFace URLs:
1. Downloads model using HuggingFace CLI
2. Auto-selects specified quantization or first GGUF found
3. Handles sharded downloads automatically
4. Proceeds with merge and configuration

---

## 📊 Example Output

```
╔════════════════════════════════════════════════════════════╗
║          Ollama Copilot Fixer - Model Setup Tool           ║
╚════════════════════════════════════════════════════════════╝

ℹ Step 1: Checking dependencies... 
✓ Ollama is installed
✓ llama.cpp found at C:\llama.cpp\bin
✓ HuggingFace CLI is installed

ℹ Step 2: Processing model source... 
ℹ Detected HuggingFace repository:  unsloth/Llama-3.2-3B-Instruct-GGUF
ℹ Downloading model files... 
✓ Downloaded:  Llama-3.2-3B-Instruct-Q4_K_M.gguf (2.1 GB)

ℹ Step 3: Checking for sharded files... 
✓ Model is a single file (no merging needed)

ℹ Step 4: Determining model architecture...
✓ Using architecture: llama3

ℹ Step 5: Generating Modelfile with Tool capability...
✓ Modelfile generated with tool support

ℹ Step 6: Creating model in Ollama...
✓ Model created successfully:  llama3-copilot

ℹ Step 7: Testing model...
✓ Model responded successfully

╔════════════════════════════════════════════════════════════╗
║                    SETUP COMPLETE!                          ║
╚════════════════════════════════════════════════════════════╝

Your model is ready for use with GitHub Copilot!  🎉
```

---

## 🤔 Troubleshooting

### Sharded File Merging Fails

**Error:** "gguf-split not found" or merge errors

**Solutions:**
1. Install llama.cpp from [releases](https://github.com/ggerganov/llama.cpp/releases)
2. Specify path manually: `-LlamaCppPath "C:\path\to\llama.cpp"`
3. Ensure all shard files are in the same directory
4. Verify files downloaded completely (check file sizes)

### HuggingFace Download Issues

**Error:** "hf command not found"

**Solution:**
```powershell
# Install HF CLI
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"

# Or use Python
pip install -U huggingface_hub
```

### Model Not Appearing in Copilot

1.  Restart VS Code completely
2. Verify:  `ollama list` shows the model with "tool" capability
3. Check: `ollama show MODEL_NAME --modelfile`
4. Ensure Ollama is running:  `ollama serve`

### Out of Disk Space

For large models (70B+), ensure you have:
- 2x model size for sharded files during merge
- Use `-KeepDownloads $false` to auto-cleanup

---

## 🎓 Technical Background

### Why Sharded Files Exist
Models over ~50GB are often split by HuggingFace due to:
- Git LFS file size limits
- Easier parallel downloads
- Repository storage constraints

### Why Tool Capability Matters
GitHub Copilot uses "tool calling" (function calling) to:
- Execute code completions
- Access context from your workspace
- Invoke language model APIs properly

The tool template tells Ollama how to format these special requests.

### Merging Process
The `gguf-split` utility from llama.cpp:
1. Reads shard metadata from first file
2. Concatenates tensor data in correct order
3. Rebuilds GGUF header with full model info
4. Outputs single monolithic file

---

## 📚 Resources

- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md)
- [llama.cpp GGUF Split Guide](https://github.com/ggml-org/llama.cpp/discussions/6404)
- [HuggingFace CLI Documentation](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli)
- [GitHub Copilot Local Models](https://code.visualstudio.com/docs/copilot/copilot-customization)
- [Ollama Multi-file GGUF Issue](https://github.com/ollama/ollama/issues/5245)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional architecture support
- Better error recovery
- Cross-platform support (Linux/macOS)
- GUI wrapper

Please open an issue or pull request! 

---

## 📝 License

MIT License - free to use, modify, and distribute. 

---

## 🙏 Acknowledgments

- **Ollama team** - Excellent local LLM runtime
- **llama.cpp** - GGUF tools and quantization
- **Unsloth** - Optimized model quantizations
- **HuggingFace** - Model hosting and distribution
- **GitHub** - Copilot's local model support

---

**Made with ❤️ for the local AI community**

**Problems fixed:  3 | Stars deserved:  ⭐⭐⭐**