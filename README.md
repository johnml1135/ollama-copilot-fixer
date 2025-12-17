# Ollama Copilot Fixer 🔧🤖

**Fix and enable custom GGUF models (including sharded files) from HuggingFace/Unsloth for GitHub Copilot with full Tool support**

GitHub Copilot's local model integration requires the **"Tool"** capability to function properly. This repo provides a **Python CLI** that:

1. ✅ **Downloads GGUFs from Hugging Face**
2. ✅ **Detects + merges sharded GGUFs** (via `llama-gguf-split`)
3. ✅ **Generates an Ollama Modelfile** with tool-capable chat templates
4. ✅ **Runs `ollama create`** so the model is usable by Copilot

---

## 🎯 Problems This Solves

### Problem 1: Sharded GGUF Files
Large models on HuggingFace are often split into multiple files (`model-00001-of-00005.gguf`, etc.). Ollama cannot load these directly—they must be merged first.

### Problem 2: Missing Tool Capability
Custom GGUF imports to Ollama only show "Inference" capability by default. GitHub Copilot requires "Tool" capability to function. 

### Problem 3: Manual Downloads
Downloading models from HuggingFace and configuring them manually is tedious and error-prone.

**This script fixes all three issues automatically.** 🚀

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

- **Python 3.10+**
- **[Ollama](https://ollama.ai/download)** installed and running (`ollama serve`)
- **[llama.cpp](https://github.com/ggml-org/llama.cpp/releases)** (only needed for merging sharded GGUFs)
- **Python deps**: `python -m pip install -r requirements.txt`
- **Hugging Face CLI** (optional fallback for HF downloads): `python -m pip install -U huggingface_hub`
- **VS Code** with GitHub Copilot extension

---

## 🚀 Quick Start

### 1) (Optional) Install Hugging Face CLI

```bash
python -m pip install -r requirements.txt
```

If the repo is gated/private, authenticate first:

```bash
hf auth login
```

### 2) Run the tool

The CLI is runnable directly from the repo:

```bash
python -m ollama_copilot_fixer --help
```

### Usage Examples

#### 1. Download from Hugging Face

```bash
python -m ollama_copilot_fixer \
	--model-source "unsloth/Llama-3.2-3B-Instruct-GGUF" \
	--model-name "llama3-copilot"
```

#### 1b. Download using Ollama-style HF syntax (recommended)

```bash
python -m ollama_copilot_fixer \
	--model-source "hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q4_0" \
	--model-name "nemotron-copilot"
```

Notes:
- `hf.co/<owner>/<repo>` is treated as the Hugging Face repo id.
- The `:<QUANT>` suffix is treated like `--quantization-type` (e.g. `Q4_0`, `Q4_K_M`, `IQ2_XXS`).

#### 2. Use a local GGUF file

```bash
python -m ollama_copilot_fixer --model-source "C:\\Models\\nemotron-nano-Q4_K_M.gguf"
```

#### 3. Merge sharded files and configure

Provide the **first shard** and a path to `llama-gguf-split`:

```bash
python -m ollama_copilot_fixer \
	--model-source "C:\\Models\\llama-405b-00001-of-00008.gguf" \
	--llama-cpp-path "C:\\llama.cpp\\bin" \
	--model-name "llama-405b"
```

#### 4. Download a specific quant

```bash
python -m ollama_copilot_fixer \
	--model-source "bartowski/Llama-3.3-70B-Instruct-GGUF" \
	--quantization-type "Q4_K_M"
```

#### 5. Works with Unsloth Dynamic 2.0 quants

```bash
python -m ollama_copilot_fixer --model-source "hf.co/unsloth/<MODEL-REPO>:IQ2_XXS"
```

---

## 📖 CLI Options

Run `python -m ollama_copilot_fixer --help` for the full list.

Common options:

| Option | Required | Default | Description |
|---|---:|---:|---|
| `--model-source` | ✅ Yes | - | Local path, HF repo id/URL, or `hf.co/<owner>/<repo>:<QUANT>` |
| `--model-name` | ❌ No | Derived | Name to register in Ollama |
| `--architecture` | ❌ No | `auto` | `llama3`, `mistral`, `phi3`, `gemma2`, `qwen`, or `auto` |
| `--context-length` | ❌ No | (auto) | Context window (`num_ctx`). If omitted, this tool will not set `num_ctx` in the Modelfile and Ollama/model defaults apply (Ollama may cap the maximum, e.g. 256k). |
| `--temperature` | ❌ No | `0.7` | Default sampling temperature |
| `--quantization-type` | ❌ No | - | Quant filter for HF downloads |
| `--llama-cpp-path` | ❌ No | - | Path to llama.cpp folder or `llama-gguf-split.exe` |
| `--keep-downloads` | ❌ No | off | Keep temp working directory |
| `--skip-test` | ❌ No | off | Skip `ollama run` smoke test |

Tip: if you want to explicitly request a large context window (subject to Ollama limits), pass it directly:

```bash
python -m ollama_copilot_fixer \
	--model-source "hf.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF:Q4_0" \
	--model-name "nemotron-copilot" \
	--context-length 262144
```

### Caching + config

This tool now uses an app-managed cache directory (downloads, HF cache, merged GGUFs) to reduce repeated downloads and merges.

- Default config path:
	- Windows: `%APPDATA%\ollama-copilot-fixer\config.json`
	- Linux: `$XDG_CONFIG_HOME/ollama-copilot-fixer/config.json` (or `~/.config/...`)
- Default cache root:
	- Windows: `%LOCALAPPDATA%\ollama-copilot-fixer\cache`
	- Linux: `$XDG_CACHE_HOME/ollama-copilot-fixer` (or `~/.cache/...`)

Override config and cache locations:

```bash
python -m ollama_copilot_fixer --config "C:\path\to\config.json" --cache-root "D:\LLMCache"
```

An example config is provided in [config.example.json](config.example.json).

Cache commands:

```bash
python -m ollama_copilot_fixer cache info
python -m ollama_copilot_fixer cache clear
```

---

## 🏗️ Supported Architectures

- ✅ **Llama 3 / 3.1 / 3.2 / 3.3** (including Nemotron variants)
- ✅ **Mistral / Mixtral**
- ✅ **Phi-3 / Phi-4**
- ✅ **Gemma 2**
- ✅ **Qwen 2 / 2.5**

---

## 🔧 How It Works

### For Local Files
1. Detects if file is part of a sharded set
2. Merges shards using llama.cpp's `llama-gguf-split` tool
3. Applies architecture-specific tool template
4. Creates model in Ollama with Tool capability

### For HuggingFace URLs:
1. Downloads model using HuggingFace CLI
2. Auto-selects specified quantization or first GGUF found
3. Handles sharded downloads automatically
4. Proceeds with merge and configuration

---

## ✅ Recommended Fix for Nemotron Tool Calling: NVIDIA NIM

Some Nemotron GGUF builds (notably `Nemotron-3-Nano`) can emit tool calls as **plain text** (for example, XML-ish blocks like `<function=read_file>...`) when run via Ollama. GitHub Copilot expects **OpenAI-style structured tool calls** (`tool_calls`), so it may render that markup verbatim instead of executing tools.

NVIDIA NIM provides an OpenAI-compatible `/v1` API and supports tool calling for **Nemotron 3 Nano**.

### If you are using an LLM-specific NIM container (recommended)

Tool calling is enabled automatically for supported models (Nemotron 3 Nano is listed as supported), and NVIDIA’s docs explicitly recommend **not** setting tool-calling environment variables externally for LLM-specific NIM containers.

Configure Copilot to point at the NIM endpoint:
- Base URL: `http://localhost:8000/v1`

### If you are using a generic LLM NIM deployment

Enable tool calling and select a post-processor:

```bash
NIM_ENABLE_AUTO_TOOL_CHOICE=1
NIM_TOOL_CALL_PARSER=pythonic
```

If the chat response contains an empty `tool_calls` field but the function call appears in `content`, the parser/template combination is mismatched. Per NVIDIA’s guidance, fix this by switching parsers (options include `mistral`, `llama3_json`, `granite`, `hermes`, `jamba`) and/or overriding the chat template via `NIM_CHAT_TEMPLATE`.

---

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

### Sharded file merging fails

**Error:** "gguf-split not found" or merge errors

**Solutions:**
1. Install llama.cpp from [releases](https://github.com/ggml-org/llama.cpp/releases)
2. Specify the path to `llama-gguf-split` manually: `--llama-cpp-path "C:\\path\\to\\llama.cpp\\bin"`
3. Ensure all shard files are in the same directory
4. Verify files downloaded completely (check file sizes)

### Hugging Face download issues

**Error:** `hf` not found

```bash
python -m pip install -U huggingface_hub
```

**Error:** no GGUFs found after download

- The repo may be gated: run `hf auth login`.
- Your quant filter may be too strict: omit `--quantization-type` and try again.

### Model not appearing in Copilot

1. Restart VS Code completely
2. Verify `ollama list` shows the model
3. Check `ollama show MODEL_NAME --modelfile`
4. Ensure Ollama is running: `ollama serve`

### Copilot shows `<function=...>` or other tool markup

This usually means the model is outputting tool calls as plain text, not as structured `tool_calls`.

- For Nemotron models, prefer running them via NVIDIA NIM (see the section above).
- For Ollama + GGUF imports, see “Future Improvements” below for the proxy approach.

### Out of Disk Space

For large models (70B+), ensure you have:
- 2x model size for sharded files during merge
- Use the default cleanup behavior (omit `--keep-downloads`) to remove temporary downloads/merges

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
- [NVIDIA NIM Function (Tool) Calling](https://docs.nvidia.com/nim/large-language-models/1.15.0/function-calling.html)

---

## 🔮 Future Improvements: ToolBridge / OpenAI↔Ollama Proxy

When using Ollama with certain GGUF imports, the model may not consistently produce structured tool calls even with a good Modelfile template. In those cases, a lightweight proxy can translate “tool call as text” into OpenAI-compatible `tool_calls` so GitHub Copilot can execute tools instead of displaying markup.

### Why it would be needed

- Ollama can expose an OpenAI-like API, but tool calling still depends on the model emitting a parseable tool-call dialect.
- Some models (including certain Nemotron GGUF builds) emit tool calls in a non-standard textual format.

### How it could be implemented

- Run a local proxy (for example, ToolBridge) that accepts OpenAI `/v1` requests from Copilot.
- Proxy forwards prompts to Ollama, then:
	- Detects tool-call text patterns in the assistant content.
	- Converts them into structured `tool_calls` in the OpenAI response.
	- Optionally strips special tokens / turn markers.
- Update this repo to add:
	- A probe step to detect whether a model emits structured tool calls.
	- Optional commands to install/run the proxy as a background service (Windows Scheduled Task, Linux systemd).
	- Minimal config to point Copilot at the proxy endpoint when needed.

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