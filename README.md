# Unsloth Llama Helper Scripts

PowerShell helpers for running the Reddit-tested Qwen3.6 27B MTP IQ4_KS setup
on a single 24 GB NVIDIA GPU via [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)'s
`llama-server`, exposed as an OpenAI-compatible endpoint for **VS Code GitHub
Copilot Chat (BYOK)**, Cline, OpenCode, Claude Code Router, etc.

No GGUF rebuilding. No Modelfiles. No Ollama. The active catalog intentionally
has one supported choice: `ubergarm/Qwen3.6-27B-GGUF` /
`Qwen3.6-27B-MTP-IQ4_KS.gguf` with the ik_llama.cpp parameters from the
LocalLLaMA RTX 3090 recipe.

## Why not Ollama?

Ollama's bundled runner does not register the `qwen35moe` architecture used by
Qwen3.6-35B-A3B
([ollama/ollama#15747](https://github.com/ollama/ollama/issues/15747)) and
Unsloth's own docs state
[*"Currently no Qwen3.6 GGUF works in Ollama due to separate mmproj vision
files. Use llama.cpp compatible backends."*](https://unsloth.ai/docs/models/qwen3.6)

Use the **[GitHub Copilot LLM Gateway](https://marketplace.visualstudio.com/items?itemName=AndrewButson.github-copilot-llm-gateway)**
extension by Andrew Butson to connect Copilot Chat to your local llama-server.
The extension registers your server's models as a first-class provider inside
Copilot Chat, with full agent mode and tool calling support.

> Caveat: This powers **chat + agent** only. Inline ghost-text completions stay
> on GitHub-hosted models regardless of which inference backend you pick.

## Quick start

```powershell
# 1. (first run only) clone and build ik_llama.cpp into tools\ik_llama.cpp\
.\scripts\install-llama.ps1            # CUDA build for RTX 3090 by default

# 2. start the single supported profile in the background
.\scripts\start-server.ps1             # auto-builds ik_llama.cpp if missing

# 3. check it
.\scripts\status-server.ps1

# 4. stop it
.\scripts\stop-server.ps1
```

The server listens on `http://127.0.0.1:8080/v1` (OpenAI-compatible).
First launch downloads the GGUF model and CPU mmproj sidecar into `models\`.
Set `HF_TOKEN` first if Hugging Face requires authentication in your
environment.

## Wire VS Code Copilot Chat to it

### 1. Install the GitHub Copilot LLM Gateway extension

Install **[GitHub Copilot LLM Gateway](https://marketplace.visualstudio.com/items?itemName=AndrewButson.github-copilot-llm-gateway)**
(`AndrewButson.github-copilot-llm-gateway`) from the VS Code Marketplace. This
workspace also lists it as a recommended extension — accept the prompt when you
open the folder, or install it from the Extensions view.

### 2. Configure the extension

The extension settings are under `github.copilot.llm-gateway.*`. The workspace
`.vscode/settings.json` already includes the defaults below, but you can
override them in your user settings if needed:

```jsonc
// .vscode/settings.json (already included in this repo)
"github.copilot.llm-gateway.serverUrl": "http://127.0.0.1:8080",
"github.copilot.llm-gateway.requestTimeout": 600000,
"github.copilot.llm-gateway.defaultMaxTokens": 16384,
"github.copilot.llm-gateway.defaultMaxOutputTokens": 4096,
"github.copilot.llm-gateway.enableToolCalling": true,
"github.copilot.llm-gateway.parallelToolCalling": false,
"github.copilot.llm-gateway.agentTemperature": 0.0
```

> **Important:** Set the `serverUrl` to the **base URL only** — do NOT include
> `/v1` or a trailing slash. The extension appends `/v1/models` itself.

On this 24 GB Qwen setup, keep the advertised Copilot context inside the
effective local budget rather than the model's theoretical max. The repo now
defaults Copilot to **16K** tokens for tool-heavy local agent work. The Qwen
profile allocates much larger native context, but cold Copilot agent
prompts with large tool schemas slow down sharply before they reach those
limits: a 7.5K-token synthetic tool-schema probe took about 3 minutes on the RTX
3090, and real 36K/58K-token Copilot prompts were canceled before prefill
completed. You can raise this to **32K** for lighter chat or smaller tool
surfaces once first turns are stable. Parallel tool calling is disabled by
default because serial calls are slower but more reliable on local Qwen; turn it
back on after your tool calls are stable.

### 3. Start the llama-server

```powershell
./scripts/start-server.ps1
```

The single supported model is started automatically. The server listens on
`http://127.0.0.1:8080/v1`.

### 4. Verify the connection

Open the Command Palette (`Ctrl+Shift+P`) and run:

- **GitHub Copilot LLM Gateway: Test Server Connection** — confirms connectivity
  and lists discovered models

### 5. Select a model in Copilot Chat

1. Open Copilot Chat (`Ctrl+Alt+I`)
2. Click the **model selector** dropdown at the bottom
3. Click **"Manage Models..."**
4. Select **"LLM Gateway"** as the provider
5. Enable the local `qwen3.6-27b-mtp-iq4-ks` model

The extension auto-discovers whatever model is currently loaded on llama-server.
See [the extension docs](https://github.com/arbs-io/github-copilot-llm-gateway)
for more details on tool calling, agent mode, and troubleshooting.

## Initialize Copilot for local agents

This repo includes a small Copilot customization bundle for people using a
single 24 GB GPU with a local LLM Gateway model. It installs five skills that
teach Copilot to keep context small, delegate focused research to subagents, and
debug local tool-call failures without stuffing the whole session into Qwen.

Run the initializer and choose whether to install into your **user profile** or
into a **repository**:

```powershell
.\scripts\initialize-copilot-local-agent.ps1
```

Non-interactive examples:

```powershell
# Install skills to ~\.copilot\skills and settings to VS Code User settings
.\scripts\initialize-copilot-local-agent.ps1 -Scope User

# Install skills/settings into another repo
.\scripts\initialize-copilot-local-agent.ps1 -Scope Repo -RepoPath C:\src\my-repo
```

Repo installs write:

- `.github\skills\...`
- `.vscode\settings.json`

User installs write:

- `~\.copilot\skills\...`
- `%APPDATA%\Code\User\settings.json` on Windows

Existing settings files are backed up before being rewritten. Existing skill
folders are skipped unless you pass `-Force`.

Installed skills:

| Skill | Use it for |
| ----- | ---------- |
| `local-context-budget` | Inspecting prompt bloat, context reserve, and token-heavy Copilot sessions. |
| `local-repo-triage` | Finding only the files needed for a coding task before implementation. |
| `local-subagent-delegation` | Using same-model fresh-context subagents for focused research or review. |
| `local-tool-reliability` | Stabilizing Qwen/llama-server/LLM Gateway tool calls. |
| `local-session-handoff` | Compressing a long local-agent session into a fresh-chat checkpoint. |

After installing, restart or reload VS Code if the skills do not immediately
appear in the `/` menu. The skills are intentionally task-specific rather than
always-on instructions, so they do not burn context unless Copilot loads them for
a relevant request.

The skills are now structured for progressive loading instead of carrying their
rationale in the main instruction body. Discovery stays in the frontmatter
`description`, action steps stay in each `SKILL.md`, and the supporting research
lives in per-skill `references/evidence.md` files plus the shared
[.github/skills/REFERENCES.md](.github/skills/REFERENCES.md). This matches the
VS Code skill-loading model: concise procedural bodies, heavier references only
when needed.

Headline numbers that drive the defaults:

- RULER puts Qwen3-30B-A3B's *effective* context at **64K** despite a 128K
  claim — accuracy drops from 96.5 @ 4K to 79.2 @ 128K. The repo's installer
  therefore defaults `defaultMaxTokens` to **16384** for tool-heavy local agent
  turns, while the skills treat **32K as the high-quality upper comfort zone**
  and **64K as a soft research ceiling**.
- Anthropic's multi-agent post shows token usage alone explains ~80% of agent
  performance variance, and that multi-agent systems use ~15× more tokens than
  chat — so on a single GPU with no parallelism, subagents are for *context
  isolation*, not speed.
- The Berkeley Function-Calling Leaderboard shows every model degrades with
  more tools (a quantized Llama-3.1-8B failed with 46 tools, succeeded with
  19), which is why `parallelToolCalling` defaults to `false` and the tool
  reliability skill recommends trimming the tool surface before debugging the
  parser.

## Single Qwen3.6 27B Profile

Defined in [scripts/models.ps1](scripts/models.ps1). The catalog intentionally
contains exactly one entry:

| Key | Backend | Model file | KV | Context | MTP | Vision |
| --- | ------- | ---------- | -- | ------- | --- | ------ |
| `qwen36-27b-mtp-iq4-ks` | `ik_llama.cpp` | `ubergarm/Qwen3.6-27B-GGUF` / `Qwen3.6-27B-MTP-IQ4_KS.gguf` | `q8_0` / `q8_0` | 156,000 | ik built-in MTP | `mmproj-BF16.gguf` on CPU |

The profile uses the LocalLLaMA RTX 3090 recipe:

```text
--ctx-size 156000
--cache-type-k q8_0
--cache-type-v q8_0
--flash-attn on
--multi-token-prediction
--draft-max 4
--draft-p-min 0.0
--merge-qkv
--merge-up-gate-experts
--cache-ram 32768
--ctx-checkpoints 32
--ctx-checkpoints-interval 512
--ctx-checkpoints-tolerance 5
--cache-ram-similarity 0.50
--cache-ram-n-min 0
--threads 8
--threads-batch 8
--threads-mtmd 8
--batch-size 2048
--ubatch-size 512
--gpu-layers 99
--split-mode none
--main-gpu 0
--parallel 1
--cont-batching
--reasoning on
--reasoning-format deepseek
--chat-template-kwargs {"tool_parser":"qwen3_coder","preserve_thinking":true}
--mmproj <mmproj-BF16.gguf>
--no-mmproj-offload
--image-min-tokens 1024
--image-max-tokens 4096
```

The model file is about 16.2 GB. The mmproj sidecar is downloaded from
`unsloth/Qwen3.6-27B-MTP-GGUF` because the ubergarm quant repo contains the
IQ4_KS weight files but not the projector file.

If 156K context is too tight on your Windows desktop, pass
`-ContextOverride 128000` to [scripts/start-server.ps1](scripts/start-server.ps1).
The Reddit thread also notes that q4 KV is the next fallback if you need more
VRAM headroom, but this repo now keeps the requested q8 KV recipe as the only
catalog choice.

## Sampler defaults

From the LocalLLaMA recipe and Qwen coding defaults:

| Family | temp | top_p | top_k | min_p | presence_penalty | repeat_penalty |
| ------ | ---- | ----- | ----- | ----- | ---------------- | -------------- |
| Qwen3.6 (precise coding, thinking) | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |

Disable thinking with `-NoThink`. The script then switches to
`--reasoning off` and merges `enable_thinking:false` into the chat-template
kwargs without dropping the Qwen tool-call fixes.

### Qwen3.6 tool calling fixes

Qwen3.6 models can leak reasoning content or fail to close `<thinking>` tags
before outputting tool calls, causing strict XML-style parsing to fail with
"Request failed" errors. For the single Qwen3.6 profile,
[scripts/start-server.ps1](scripts/start-server.ps1) uses the local fixed
template at [scripts/templates/qwen36-tool-fix.jinja](scripts/templates/qwen36-tool-fix.jinja)
and the Reddit reasoning-on settings:

```powershell
LLAMA_CHAT_TEMPLATE_KWARGS={"tool_parser":"qwen3_coder","preserve_thinking":true}
--reasoning on
--reasoning-format deepseek
```

`-NoThink` is still available if a client starts losing tool-call state across
reasoning turns:

```powershell
LLAMA_CHAT_TEMPLATE_KWARGS={"tool_parser":"qwen3_coder","enable_thinking":false}
--reasoning off
```

The no-think path deliberately does **not** seed an empty `<think></think>`
assistant prefix. That earlier pattern could trigger llama-server's reasoning
parser while reasoning was disabled and leave generation spinning at high CPU
with little GPU progress.

When Qwen still fails after a follow-up tool call, capture the boundary between
Copilot and `llama-server` instead of guessing which side dropped the tool call:

```powershell
# Terminal 1: keep llama-server on 8080
.\scripts\start-server.ps1

# Terminal 2: proxy Copilot traffic through 8090 and log raw request/response JSON
node .\scripts\trace-openai-proxy.js --listen 8090 --target http://127.0.0.1:8080

# Point Copilot's custom model URL at http://127.0.0.1:8090/v1, reproduce once,
# then summarize what the server actually returned.
.\scripts\analyze-openai-trace.ps1
```

If the analyzer reports `server-returned-structured-tool-calls` for the failing
second turn, `llama-server` did its job and the remaining bug is in the Copilot
or gateway agent loop. If it reports `server-returned-tool-xml-outside-tool_calls`
or `server-returned-reasoning-without-tool_calls`, the failure is still on the
model/template/parser side. The trace log contains full prompts and tool
results, so treat `logs\openai-proxy-trace.jsonl` as sensitive.

To reproduce model-side behavior without involving Copilot, use the Python
OpenAI-compatible probe. It runs a tiny chat, a two-step structured tool loop, a
synthetic bloated tool schema, and an optional prompt-prefill stress, then
summarizes the new llama-server timing lines:

```powershell
python .\scripts\probe-openai-models.py --tests basic,two-tools,schema-bloat --schema-kb 16
python .\scripts\probe-openai-models.py --tests schema-bloat --schema-kb 32 --timeout-sec 420
```

Old upstream llama.cpp Q5 probes with a 32 KB synthetic schema pointed to cold tool-schema prefill as
the limiting factor for giant Copilot agent prompts. After the ik migration,
rerun [scripts/probe-openai-models.py](scripts/probe-openai-models.py) against
the new profile before drawing conclusions from the older Q5 timings.

## Scripts

| Script                          | Purpose                                                              |
| ------------------------------- | -------------------------------------------------------------------- |
| `scripts\install-llama.ps1`     | Clone and build `ikawrakow/ik_llama.cpp` from source (CUDA/CPU).     |
| `scripts\start-server.ps1`      | Resolve the single profile, auto-build ik if missing, launch `llama-server`. |
| `scripts\status-server.ps1`     | PID, model, /health probe, `nvidia-smi` snapshot.                    |
| `scripts\stop-server.ps1`       | Stop the background server.                                          |
| `scripts\trace-openai-proxy.js` | Logs raw OpenAI-compatible traffic between Copilot and `llama-server`. |
| `scripts\analyze-openai-trace.ps1` | Summarizes whether traced responses contained structured tool calls. |
| `scripts\probe-openai-models.py` | Sends controlled OpenAI-compatible chat/tool/schema probes and correlates them with llama-server timings. |
| `scripts\models.ps1`            | Single-profile model catalog & sampler defaults.                     |
| `scripts\benchmark-models.ps1`  | Loads each catalog model, measures VRAM, runs a one-shot inference.  |
| `scripts\inspect-copilot-context.ps1` | Summarize Copilot prompt sizes and llama-server prompt tokens. |
| `scripts\initialize-copilot-local-agent.ps1` | Install minimal-context skills and LLM Gateway settings into a repo or user profile. |

## Measured GPU RAM

The old upstream llama.cpp measurements have been removed from the active docs
because the launcher now uses ik_llama.cpp, IQ4_KS weights, q8 KV, MTP, and a
CPU mmproj sidecar. After the first ik run, use the status script to inspect
actual Windows load behavior on this machine:

```powershell
.\scripts\status-server.ps1
```

The source Reddit recipe reported roughly **1261 tok/s prefill** and
**72.9 tok/s decode** on an RTX 3090-class setup for a ~5.9K prompt and ~1K
output. Treat those as target recipe numbers, not local verification. If the
156K/q8 setup bumps into Windows CUDA system-memory fallback, first try
`-ContextOverride 128000`; q4 KV is the next manual fallback, but it is no
longer a separate catalog choice.

## VS Code tasks
