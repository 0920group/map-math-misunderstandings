import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


MODEL_CONFIGS = {
    "modernbert_base": {
        "model_id": "answerdotai/ModernBERT-base",
        "batch_size": 32,
        "eval_batch_size": 64,
        "is_modernbert": True,
    },
    "modernbert_large": {
        "model_id": "answerdotai/ModernBERT-large",
        "batch_size": 16,
        "eval_batch_size": 32,
        "is_modernbert": True,
    },
    "ettin_400m": {
        "model_id": "jhu-clsp/ettin-encoder-400m",
        "batch_size": 32,
        "eval_batch_size": 64,
        "is_modernbert": False,
    },
}


def load_tokenizer(model_name):
    cfg = MODEL_CONFIGS[model_name]
    return AutoTokenizer.from_pretrained(cfg["model_id"])


def load_model(model_name, num_labels, label2id, id2label):
    cfg = MODEL_CONFIGS[model_name]

    # Build config first so we can set model-specific attributes
    config = AutoConfig.from_pretrained(
        cfg["model_id"],
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    if cfg["is_modernbert"]:
        # Disable torch.compile inside the model — incompatible with HF Trainer
        if hasattr(config, "reference_compile"):
            config.reference_compile = False
        # Use mean pooling for stability
        if hasattr(config, "classifier_pooling"):
            config.classifier_pooling = "mean"

    common_kwargs = dict(
        config=config,
        dtype=torch.bfloat16,
    )

    # Try flash_attention_2, fall back to sdpa, then default
    for attn_impl in ("flash_attention_2", "sdpa", None):
        try:
            kwargs = dict(common_kwargs)
            if attn_impl is not None:
                kwargs["attn_implementation"] = attn_impl
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg["model_id"], **kwargs
            )
            label = attn_impl if attn_impl else "default"
            print(f"[model] loaded {model_name} with {label} attention")
            return model
        except Exception as e:
            print(f"[model] {attn_impl} failed: {e}, trying next...")

    raise RuntimeError(f"Failed to load model {model_name}")
