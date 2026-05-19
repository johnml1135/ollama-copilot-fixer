<#
.SYNOPSIS
    Catalog of Unsloth GGUF models tuned for a single 24 GB NVIDIA GPU.

.DESCRIPTION
    Each entry is a profile of well-known-good llama-server arguments derived
    from Unsloth's official "How to Run Locally" pages:
      https://unsloth.ai/docs/models/qwen3.6
      https://unsloth.ai/docs/models/gemma-4

    Files are pulled via llama.cpp's `-hf user/repo --hf-file <name>.gguf`
    form, cached under $env:LLAMA_CACHE.

        Context-window choices are SIZED FOR 24 GB VRAM:
            - Qwen3.6-27B: HYBRID -- 16 of 64 layers full-attention (GQA 24:4,
        head_dim 256), other 48 are Gated DeltaNet linear-attention with
                constant-size state. The active catalog keeps only four Qwen profiles:
                MTP/non-MTP crossed with 150K-class and 256K-class context targets.
      - Gemma 4 26B-A4B (MoE): kv_heads=4 with sliding-window pattern;
        very modest KV.
      - Gemma 4 31B (dense): 60 layers w/ 1-in-6 full-attention pattern
        (10 full + 50 sliding-window-1024). KV at 128K @ q8_0 ~= 6 GB.

    Sizes/contexts here have been verified by scripts\benchmark-models.ps1.
    See README.md "Measured GPU RAM" section for actual numbers.
#>

# Profile schema:
#   Key        : menu key (short id)
#   Name       : human label
#   HFRepo     : huggingface repo for `-hf` shortcut
#   HFFile     : explicit GGUF filename in that repo (avoids HF preset 404s)
#   Quant      : human-readable quant tag (display only)
#   Alias      : value advertised to OpenAI clients (the "model" field)
#   Context    : --ctx-size (sized for 24 GB GPU, NOT the model max)
#   MaxContext : the model's native max (for documentation only)
#   Size       : on-disk weight size at the chosen quant (verified vs HF)
#   Family     : 'qwen36' | 'gemma4' (controls samplers)
#   Temp/TopP/TopK/MinP/PresencePenalty/RepeatPenalty
#              : optional per-profile sampler overrides
#   CacheTypeK/CacheTypeV
#              : KV cache quantization for this profile; default q8_0
#   NoMmproj   : pass --no-mmproj to avoid auto-loading unused vision projector
#   Batch/UBatch
#              : optional logical/physical batch sizes to reduce compute buffer
#   ExtraArgs  : array of llama-server args appended verbatim
#   Notes      : free-form caveats shown in the menu

$global:LlamaModelCatalog = [ordered]@{

    'qwen36-27b-mtp-q5' = @{
        Name        = 'Qwen3.6 27B MTP (Q5_K_M, 160K context)'
        HFRepo      = 'unsloth/Qwen3.6-27B-MTP-GGUF'
        HFFile      = 'Qwen3.6-27B-Q5_K_M.gguf'
        Quant       = 'Q5_K_M + MTP'
        Alias       = 'qwen3.6-27b-mtp-q5'
        Context     = 160000
        MaxContext  = 262144
        Size        = '18.5 GB'
        Family      = 'qwen36'
        CacheTypeK  = 'q4_1'
        CacheTypeV  = 'q4_1'
        NoMmproj    = $true
        Speculative = 'draft-mtp, draft_n_max=2'
        ExtraArgs   = @(
            '--spec-type', 'draft-mtp',
            '--spec-draft-n-max', '2'
        )
        Notes       = 'Highest-quality MTP profile in the 150K-class slot. Live probe loaded 66/66 GPU layers at 160K with q4_1 KV, using 22,182 MiB of 22,854 MiB initially free; tight, so do not raise context without re-testing.'
    }

    'qwen36-27b-mtp-quality-max' = @{
        Name        = 'Qwen3.6 27B MTP (Dynamic Q4, near-native context)'
        HFRepo      = 'unsloth/Qwen3.6-27B-MTP-GGUF'
        HFFile      = 'Qwen3.6-27B-UD-Q4_K_XL.gguf'
        Quant       = 'UD-Q4_K_XL + MTP'
        Alias       = 'qwen3.6-27b-mtp-quality-max'
        Context     = 245760
        MaxContext  = 262144
        Size        = '16.7 GB'
        Family      = 'qwen36'
        CacheTypeK  = 'q4_1'
        CacheTypeV  = 'q4_1'
        NoMmproj    = $true
        Speculative = 'draft-mtp, draft_n_max=2'
        ExtraArgs   = @(
            '--spec-type', 'draft-mtp',
            '--spec-draft-n-max', '2'
        )
        Notes       = 'Best 256K-class MTP tradeoff: Unsloth Dynamic 4-bit weights with near-native context. Verified 66/66 GPU layers at 245,760 tokens with q4_1 KV, using 22,424 MiB of 22,854 MiB initially free.'
    }

    'qwen36-27b-q5' = @{
        Name       = 'Qwen3.6 27B (Q5_K_M, 200K context)'
        HFRepo     = 'unsloth/Qwen3.6-27B-GGUF'
        HFFile     = 'Qwen3.6-27B-Q5_K_M.gguf'
        Quant      = 'Q5_K_M'
        Alias      = 'qwen3.6-27b-q5'
        Context    = 200000
        MaxContext = 262144
        Size       = '18.2 GB'
        Family     = 'qwen36'
        CacheTypeK = 'q4_1'
        CacheTypeV = 'q4_1'
        NoMmproj   = $true
        Notes      = 'Highest-quality non-MTP 150K-class profile. Live probe loaded 65/65 GPU layers at 200K with q4_1 KV, using 22,476 MiB of 23,154 MiB initially free; tight, but comparable to the MTP Q5_K_M profile.'
    }

    'qwen36-27b-quality-max' = @{
        Name       = 'Qwen3.6 27B (Dynamic Q4, native context)'
        HFRepo     = 'unsloth/Qwen3.6-27B-GGUF'
        HFFile     = 'Qwen3.6-27B-UD-Q4_K_XL.gguf'
        Quant      = 'UD-Q4_K_XL'
        Alias      = 'qwen3.6-27b-quality-max'
        Context    = 262144
        MaxContext = 262144
        Size       = '16.4 GB'
        Family     = 'qwen36'
        CacheTypeK = 'q4_1'
        CacheTypeV = 'q4_1'
        NoMmproj   = $true
        Notes      = 'Best non-MTP 256K-class profile. Verified 65/65 GPU layers at 262,144 tokens with q4_1 KV, using 22,210 MiB of 23,154 MiB initially free.'
    }

    'gemma4-26b-a4b' = @{
        Name       = 'Gemma 4 26B-A4B (MoE)'
        HFRepo     = 'unsloth/gemma-4-26B-A4B-it-GGUF'
        HFFile     = 'gemma-4-26B-A4B-it-UD-Q5_K_S.gguf'
        Quant      = 'UD-Q5_K_S'
        Alias      = 'gemma-4-26b-a4b'
        Context    = 200000
        MaxContext = 262144
        Size       = '17.5 GB'
        Family     = 'gemma4'
        CacheTypeK = 'q8_0'
        CacheTypeV = 'q8_0'
        ExtraArgs  = @()
        Notes      = 'MoE w/ sliding-window. Measured 22.4 GiB @ 200K (~1.6 GiB free). mmproj vision sidecar not loaded.'
    }

    'gemma4-31b' = @{
        Name       = 'Gemma 4 31B (dense, sliding-window)'
        HFRepo     = 'unsloth/gemma-4-31B-it-GGUF'
        HFFile     = 'gemma-4-31B-it-IQ4_XS.gguf'
        Quant      = 'IQ4_XS'
        Alias      = 'gemma-4-31b'
        Context    = 131072
        MaxContext = 262144
        Size       = '15.3 GB'
        Family     = 'gemma4'
        CacheTypeK = 'q8_0'
        CacheTypeV = 'q8_0'
        ExtraArgs  = @()
        Notes      = 'Dense 60-layer with 1-in-6 full-attn (10 full + 50 sliding-1024). Measured 23.0 GiB @ 128K (no headroom for 200K).'
    }

}

# Family-level sampler defaults from Unsloth docs (precise-coding profile for
# Qwen3.6 thinking; Google defaults for Gemma 4).
$global:LlamaFamilyDefaults = @{
    'qwen36' = @{
        Temp            = '0.6'
        TopP            = '0.95'
        TopK            = '20'
        MinP            = '0.0'
        PresencePenalty = '0.0'
        RepeatPenalty   = '1.0'
    }
    'gemma4' = @{
        Temp            = '1.0'
        TopP            = '0.95'
        TopK            = '64'
        MinP            = '0.0'
        PresencePenalty = '0.0'
        RepeatPenalty   = '1.0'
    }
}
