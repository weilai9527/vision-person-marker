import json

from config import (
    LLM_API_URL,
    LLM_API_KEY,
    LLM_MODEL,
    API_CONFIG_PATH,
    API_PROVIDERS,
)

_api_config_cache = None
_api_config_cache_mtime = 0


def mask_api_key(key: str) -> str:
    if not key or len(key) < 8:
        return ""
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def is_masked_api_key(key: str) -> bool:
    return "*" in (key or "")


def normalize_chat_endpoint(api_url: str) -> str:
    cleaned = api_url.strip().rstrip("/")
    if not cleaned:
        return API_PROVIDERS["openai"]["api_url"]
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return cleaned


def infer_provider(api_url: str) -> str:
    if "dashscope-intl" in api_url:
        return "qwen_intl"
    if "dashscope-us" in api_url:
        return "qwen_us"
    if "dashscope" in api_url:
        return "qwen"
    if "moonshot" in api_url:
        return "kimi"
    if "api.openai.com" in api_url:
        return "openai"
    return "custom"


def normalize_model_name(provider: str, model: str) -> str:
    cleaned = model.strip()
    if provider.startswith("qwen"):
        return cleaned.lower()
    return cleaned


def load_api_config() -> dict:
    global _api_config_cache, _api_config_cache_mtime

    try:
        current_mtime = API_CONFIG_PATH.stat().st_mtime if API_CONFIG_PATH.exists() else 0
    except OSError:
        current_mtime = 0

    if _api_config_cache is not None and _api_config_cache_mtime == current_mtime:
        return _api_config_cache.copy()

    config = {
        "provider": "openai",
        "api_url": LLM_API_URL,
        "api_key": LLM_API_KEY,
        "model": LLM_MODEL,
    }
    if API_CONFIG_PATH.exists():
        try:
            saved_config = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
            config.update({key: saved_config.get(key) or value for key, value in config.items()})
        except (OSError, json.JSONDecodeError):
            pass
    config["api_url"] = normalize_chat_endpoint(config["api_url"])
    if config["provider"] not in API_PROVIDERS or config["provider"] == "openai":
        config["provider"] = infer_provider(config["api_url"])
    config["model"] = normalize_model_name(config["provider"], config["model"])
    if is_masked_api_key(config.get("api_key", "")):
        config["api_key"] = ""

    _api_config_cache = config.copy()
    _api_config_cache_mtime = current_mtime
    return config


def load_api_config_for_display() -> dict:
    config = load_api_config()
    if config.get("api_key"):
        config["api_key_masked"] = mask_api_key(config["api_key"])
        config["api_key"] = ""
    else:
        config["api_key_masked"] = ""
    return config


def save_api_config(provider: str, api_url: str, api_key: str, model: str) -> None:
    global _api_config_cache, _api_config_cache_mtime
    provider = provider if provider in API_PROVIDERS else "custom"
    provider_defaults = API_PROVIDERS[provider]
    normalized_model = normalize_model_name(provider, model) or provider_defaults["model"]
    current_config = load_api_config()
    cleaned_api_key = api_key.strip()
    if not cleaned_api_key or is_masked_api_key(cleaned_api_key):
        cleaned_api_key = current_config.get("api_key", "")
    config = {
        "provider": provider,
        "api_url": normalize_chat_endpoint(api_url or provider_defaults["api_url"]),
        "api_key": cleaned_api_key,
        "model": normalized_model,
    }
    API_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    API_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _api_config_cache = None
    _api_config_cache_mtime = 0
