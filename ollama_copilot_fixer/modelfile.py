from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelTemplate:
    template: str
    stop: list[str]


_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant with tool calling capabilities. "
    "You can help with code, answer questions, and use tools when needed."
)


_TOOLS_PREAMBLE = (
    "You have access to the following tools. Use them when helpful. "
    "When you call a tool, call it using the tool calling mechanism (not as plain text in the chat content).\n"
)


_TEMPLATES: dict[str, ModelTemplate] = {
    "llama3": ModelTemplate(
        template=(
            "{{ if .Messages }}\n"
            "{{- if or .System .Tools }}<|start_header_id|>system<|end_header_id|>\n"
            "{{- if .System }}\n\n"
            "{{ .System }}\n"
            "{{- end }}\n"
            "{{- if .Tools }}\n\n"
            f"{_TOOLS_PREAMBLE}"
            "{{ .Tools }}\n"
            "{{- end }}<|eot_id|>\n"
            "{{- end }}\n"
            "{{- range .Messages }}\n"
            "<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n"
            "{{ .Content }}<|eot_id|>\n"
            "{{- end }}\n"
            "{{- else }}\n"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "{{ .System }}<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{{ .Prompt }}<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
            "{{- end }}"
        ),
        stop=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
    ),
    "mistral": ModelTemplate(
        template=(
            "{{ if .Messages }}\n"
            "{{- if or .System .Tools }}[INST]\n"
            "{{- if .System }}{{ .System }}\n"
            "{{- end }}\n"
            "{{- if .Tools }}\n\n"
            f"{_TOOLS_PREAMBLE}"
            "{{ .Tools }}\n"
            "{{- end }}[/INST]\n"
            "{{- end }}\n"
            "{{- range .Messages }}\n"
            "{{- if eq .Role \"user\" }}[INST] {{ .Content }} [/INST]\n"
            "{{- else if eq .Role \"assistant\" }}{{ .Content }}</s>\n"
            "{{- end }}\n"
            "{{- end }}\n"
            "{{- else }}[INST] {{ if .System }}{{ .System }}\n\n"
            "{{ end }}{{ .Prompt }} [/INST]\n"
            "{{- end }}"
        ),
        stop=["</s>", "[INST]", "[/INST]"],
    ),
    "phi3": ModelTemplate(
        template=(
            "{{ if .Messages }}\n"
            "{{- if or .System .Tools }}<|system|>\n"
            "{{- if .System }}{{ .System }}\n"
            "{{- end }}\n"
            "{{- if .Tools }}\n\n"
            f"{_TOOLS_PREAMBLE}"
            "{{ .Tools }}\n"
            "{{- end }}<|end|>\n"
            "{{- end }}\n"
            "{{- range .Messages }}\n"
            "<|{{ .Role }}|>\n"
            "{{ .Content }}<|end|>\n"
            "{{- end }}\n"
            "<|assistant|>\n"
            "{{- else }}<|system|>\n"
            "{{ .System }}<|end|>\n"
            "<|user|>\n"
            "{{ .Prompt }}<|end|>\n"
            "<|assistant|>\n"
            "{{- end }}"
        ),
        stop=["<|end|>", "<|system|>", "<|user|>", "<|assistant|>"],
    ),
    "gemma2": ModelTemplate(
        template=(
            "{{ if .Messages }}\n"
            "{{- if or .System .Tools }}<start_of_turn>model\n"
            "{{- if .System }}{{ .System }}\n"
            "{{- end }}\n"
            "{{- if .Tools }}\n\n"
            f"{_TOOLS_PREAMBLE}"
            "{{ .Tools }}\n"
            "{{- end }}<end_of_turn>\n"
            "{{- end }}\n"
            "{{- range .Messages }}\n"
            "<start_of_turn>{{ .Role }}\n"
            "{{ .Content }}<end_of_turn>\n"
            "{{- end }}\n"
            "<start_of_turn>model\n"
            "{{- else }}<start_of_turn>system\n"
            "{{ .System }}<end_of_turn>\n"
            "<start_of_turn>user\n"
            "{{ .Prompt }}<end_of_turn>\n"
            "<start_of_turn>model\n"
            "{{- end }}"
        ),
        stop=["<end_of_turn>", "<start_of_turn>"],
    ),
    "qwen": ModelTemplate(
        template=(
            "{{ if .Messages }}\n"
            "{{- if or .System .Tools }}<|im_start|>system\n"
            "{{- if .System }}\n"
            "{{ .System }}\n"
            "{{- end }}\n"
            "{{- if .Tools }}\n\n"
            f"{_TOOLS_PREAMBLE}"
            "{{ .Tools }}\n"
            "{{- end }}<|im_end|>\n"
            "{{- end }}\n"
            "{{- range .Messages }}\n"
            "<|im_start|>{{ .Role }}\n"
            "{{ .Content }}<|im_end|>\n"
            "{{- end }}\n"
            "<|im_start|>assistant\n"
            "{{- else }}<|im_start|>system\n"
            "{{ .System }}<|im_end|>\n"
            "<|im_start|>user\n"
            "{{ .Prompt }}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "{{- end }}"
        ),
        stop=["<|im_start|>", "<|im_end|>"],
    ),
}


def supported_architectures() -> list[str]:
    return sorted(_TEMPLATES.keys())


def generate_modelfile(
    *,
    absolute_model_path: str,
    architecture: str,
    context_length: int | None,
    temperature: float,
    extra_stop: list[str] | None = None,
    system_message: str | None = None,
) -> str:
    if architecture not in _TEMPLATES:
        raise ValueError(
            f"Unsupported architecture: {architecture}. Supported: {', '.join(supported_architectures())}"
        )

    mt = _TEMPLATES[architecture]
    stop = list(mt.stop)
    if extra_stop:
        for s in extra_stop:
            if s not in stop:
                stop.append(s)

    # Ollama Modelfile syntax
    out = [
        "# Auto-generated Modelfile with Tool capability for GitHub Copilot",
        f"# Architecture: {architecture}",
        "",
        f"FROM {absolute_model_path}",
        "",
        "# Template with Tool support",
        f'TEMPLATE """{mt.template}"""',
        "",
        "# Stop sequences",
    ]

    for seq in stop:
        out.append(f'PARAMETER stop "{seq}"')

    out += [
        "",
        "# Model parameters",
        f"PARAMETER temperature {temperature}",
        "PARAMETER num_predict -1",
        "",
        "# System message",
        f'SYSTEM """{system_message or _SYSTEM_MESSAGE}"""',
        "",
    ]

    if context_length is not None:
        if context_length <= 0:
            raise ValueError("context_length must be a positive integer")
        out.insert(out.index("PARAMETER num_predict -1"), f"PARAMETER num_ctx {context_length}")

    return "\n".join(out)
