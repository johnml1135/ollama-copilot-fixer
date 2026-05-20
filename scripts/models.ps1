<#
.SYNOPSIS
    Single-model catalog for the RTX 3090 ik_llama.cpp Qwen3.6 setup.

.DESCRIPTION
    The catalog intentionally exposes one supported choice: the Reddit-tested
    ik_llama.cpp profile for Qwen3.6 27B MTP IQ4_KS on a 24 GB NVIDIA GPU.

    Model weights are resolved from Hugging Face by scripts\start-server.ps1
    and then launched through the local ik_llama.cpp llama-server binary.
#>

# Profile schema:
#   Key        : menu key (short id)
#   Name       : human label
#   HFRepo     : huggingface repo for the GGUF weight file
#   HFFile     : explicit GGUF filename in that repo
#   Quant      : human-readable quant tag (display only)
#   Alias      : value advertised to OpenAI clients (the "model" field)
#   Context    : --ctx-size
#   MaxContext : the model's native max (for documentation only)
#   Size       : on-disk weight size at the chosen quant
#   Family     : sampler default group
#   CacheTypeK/CacheTypeV
#              : KV cache quantization for this profile
#   Batch/UBatch/Threads/ThreadsBatch/ThreadsMtmd
#              : ik_llama.cpp launch parameters from the Reddit recipe
#   MmprojHFRepo/MmprojHFFile
#              : vision projector sidecar, kept on CPU via --no-mmproj-offload
#   Reasoning  : default --reasoning mode
#   ExtraArgs  : array of ik_llama.cpp llama-server args appended verbatim
#   Notes      : free-form caveats shown in the menu

$global:LlamaModelCatalog = [ordered]@{

    'qwen36-27b-mtp-iq4-ks' = @{
        Name             = 'Qwen3.6 27B MTP (IQ4_KS, ik_llama.cpp, 156K context)'
        HFRepo           = 'ubergarm/Qwen3.6-27B-GGUF'
        HFFile           = 'Qwen3.6-27B-MTP-IQ4_KS.gguf'
        Quant            = 'IQ4_KS + MTP'
        Alias            = 'qwen3.6-27b-mtp-iq4-ks'
        Context          = 156000
        MaxContext       = 262144
        Size             = '16.2 GB'
        Family           = 'qwen36'
        CacheTypeK       = 'q8_0'
        CacheTypeV       = 'q8_0'
        Batch            = 2048
        UBatch           = 512
        Threads          = 8
        ThreadsBatch     = 8
        ThreadsMtmd      = 8
        GpuLayers        = 99
        SplitMode        = 'none'
        MainGpu          = 0
        Parallel         = 1
        Predict          = -1
        Reasoning        = 'on'
        ReasoningFormat  = 'deepseek'
        MmprojHFRepo     = 'unsloth/Qwen3.6-27B-MTP-GGUF'
        MmprojHFFile     = 'mmproj-BF16.gguf'
        NoMmprojOffload  = $true
        ImageMinTokens   = 1024
        ImageMaxTokens   = 4096
        Speculative      = 'ik built-in MTP, draft_max=4, draft_p_min=0.0'
        ExtraArgs        = @(
            '--multi-token-prediction',
            '--draft-max', '4',
            '--draft-p-min', '0.0',
            '--merge-qkv',
            '--merge-up-gate-experts',
            '--cache-ram', '32768',
            '--ctx-checkpoints', '32',
            '--ctx-checkpoints-interval', '512',
            '--ctx-checkpoints-tolerance', '5',
            '--cache-ram-similarity', '0.50',
            '--cache-ram-n-min', '0',
            '--cont-batching'
        )
        Notes            = 'Single supported profile from the LocalLLaMA RTX 3090 recipe: ubergarm IQ4_KS MTP, 156K context, q8_0 KV, ik MTP, CPU vision projector, and RAM context cache.'
    }

}

$global:LlamaFamilyDefaults = @{
    'qwen36' = @{
        Temp            = '0.6'
        TopP            = '0.95'
        TopK            = '20'
        MinP            = '0.0'
        PresencePenalty = '0.0'
        RepeatPenalty   = '1.0'
    }
}