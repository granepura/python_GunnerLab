"""Multi-provider LLM streaming backends for mcce-chat."""

import configparser
import os
import sys
from pathlib import Path

SYSTEM_PROMPT = (
    "You are MCCE Chatbox, an expert assistant for MCCE4 (Multi-Conformation "
    "Continuum Electrostatics) by the Gunner Lab at CCNY. Help users set up runs, "
    "diagnose failures, interpret pKa results, understand .ftpl topology files, and "
    "fix common problems. Always cite specific files/functions from the retrieved "
    "context. If suggesting a run.prm fix, show the exact line."
)

DEFAULT_PROVIDER = "ollama"
DEFAULT_MODEL = "llama3"

KNOWN_PROVIDERS = {
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "pip": "anthropic",
        "default_model": "claude-sonnet-4-20250514",
    },
    "ollama": {
        "env_host": "OLLAMA_HOST",
        "pip": "ollama",
        "default_model": "llama3",
    },
    "groq": {
        "env_key": "GROQ_API_KEY",
        "pip": "openai",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "pip": "openai",
        "default_model": "gpt-4o-mini",
    },
    "openai_compat": {
        "env_base_url": "OPENAI_COMPAT_BASE_URL",
        "env_key": "OPENAI_COMPAT_API_KEY",
        "pip": "openai",
        "default_model": "default",
    },
    "gemini": {
        "env_key": "GOOGLE_API_KEY",
        "pip": "google-genai",
        "default_model": "gemini-2.5-flash",
    },
}

CONF_PATH = Path.home() / ".mcce_chat.conf"


def load_config() -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    if CONF_PATH.exists():
        config.read(str(CONF_PATH))
    return config


def save_config(provider: str, model: str):
    config = load_config()
    if "defaults" not in config:
        config["defaults"] = {}
    config["defaults"]["provider"] = provider
    config["defaults"]["model"] = model
    with open(str(CONF_PATH), "w") as f:
        config.write(f)


def resolve_provider_model(cli_provider: str = None, cli_model: str = None) -> tuple:
    config = load_config()
    provider = cli_provider
    model = cli_model

    if not provider:
        provider = config.get("defaults", "provider", fallback=None)
    if not provider:
        provider = DEFAULT_PROVIDER

    if not model:
        model = config.get("defaults", "model", fallback=None)
    if not model:
        model = KNOWN_PROVIDERS.get(provider, {}).get("default_model", DEFAULT_MODEL)

    return provider, model


def _get_api_key(provider: str) -> str:
    config = load_config()
    info = KNOWN_PROVIDERS.get(provider, {})

    env_key_name = info.get("env_key", "")
    if env_key_name:
        key = os.environ.get(env_key_name)
        if key:
            return key

    if config.has_section(provider):
        key = config.get(provider, "api_key", fallback=None)
        if key:
            return key

    return None


def _get_ollama_host() -> str:
    config = load_config()
    host = os.environ.get("OLLAMA_HOST")
    if host:
        return host
    if config.has_section("ollama"):
        host = config.get("ollama", "host", fallback=None)
        if host:
            return host
    return "http://localhost:11434"


def _require_import(module_name: str, pip_name: str = None):
    try:
        return __import__(module_name)
    except ImportError:
        pip_name = pip_name or module_name
        print(f"\nError: '{module_name}' is not installed.", file=sys.stderr)
        print(f"Install it with:  pip install {pip_name}", file=sys.stderr)
        sys.exit(1)


def stream_anthropic(messages: list, model: str, run_context_text: str = ""):
    anthropic = _require_import("anthropic")
    api_key = _get_api_key("anthropic")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set. Export it or add to ~/.mcce_chat.conf [anthropic] api_key=...",
              file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    system = SYSTEM_PROMPT
    if run_context_text:
        system += f"\n\nCurrent MCCE4 run context:\n{run_context_text}"

    with client.messages.stream(
        model=model,
        max_tokens=4096,
        system=system,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            yield text


def stream_ollama(messages: list, model: str, run_context_text: str = ""):
    host = _get_ollama_host()
    system_msg = SYSTEM_PROMPT
    if run_context_text:
        system_msg += f"\n\nCurrent MCCE4 run context:\n{run_context_text}"

    full_messages = [{"role": "system", "content": system_msg}] + messages

    try:
        ollama = _require_import("ollama")
        client = ollama.Client(host=host)
        response = client.chat(model=model, messages=full_messages, stream=True)
        for chunk in response:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
    except ImportError:
        pass
    except Exception:
        import requests
        url = f"{host}/api/chat"
        payload = {"model": model, "messages": full_messages, "stream": True}
        resp = requests.post(url, json=payload, stream=True)
        resp.raise_for_status()
        import json
        for line in resp.iter_lines():
            if line:
                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content


def stream_openai_like(messages: list, model: str, run_context_text: str = "",
                       base_url: str = None, api_key: str = None):
    openai = _require_import("openai")
    system_msg = SYSTEM_PROMPT
    if run_context_text:
        system_msg += f"\n\nCurrent MCCE4 run context:\n{run_context_text}"

    full_messages = [{"role": "system", "content": system_msg}] + messages

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    client = openai.OpenAI(**kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        stream=True,
        max_tokens=4096,
    )
    for chunk in response:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            yield delta.content


def stream_groq(messages: list, model: str, run_context_text: str = ""):
    api_key = _get_api_key("groq")
    if not api_key:
        print("Error: GROQ_API_KEY not set. Export it or add to ~/.mcce_chat.conf [groq] api_key=...",
              file=sys.stderr)
        sys.exit(1)
    yield from stream_openai_like(
        messages, model, run_context_text,
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )


def stream_openai(messages: list, model: str, run_context_text: str = ""):
    api_key = _get_api_key("openai")
    if not api_key:
        print("Error: OPENAI_API_KEY not set. Export it or add to ~/.mcce_chat.conf [openai] api_key=...",
              file=sys.stderr)
        sys.exit(1)
    yield from stream_openai_like(messages, model, run_context_text, api_key=api_key)


def stream_gemini(messages: list, model: str, run_context_text: str = ""):
    genai = _require_import("google", pip_name="google-genai")
    from google import genai as genai_module
    from google.genai import types

    api_key = _get_api_key("gemini")
    if not api_key:
        print("Error: GOOGLE_API_KEY not set. Export it or add to ~/.mcce_chat.conf [gemini] api_key=...",
              file=sys.stderr)
        sys.exit(1)

    client = genai_module.Client(api_key=api_key)
    system_msg = SYSTEM_PROMPT
    if run_context_text:
        system_msg += f"\n\nCurrent MCCE4 run context:\n{run_context_text}"

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    response = client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_msg,
            max_output_tokens=4096,
        ),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


def stream_openai_compat(messages: list, model: str, run_context_text: str = ""):
    base_url = os.environ.get("OPENAI_COMPAT_BASE_URL")
    api_key = os.environ.get("OPENAI_COMPAT_API_KEY", "no-key")
    config = load_config()
    if not base_url and config.has_section("openai_compat"):
        base_url = config.get("openai_compat", "base_url", fallback=None)
        if not api_key or api_key == "no-key":
            api_key = config.get("openai_compat", "api_key", fallback="no-key")
    if not base_url:
        print("Error: OPENAI_COMPAT_BASE_URL not set.", file=sys.stderr)
        sys.exit(1)
    yield from stream_openai_like(messages, model, run_context_text,
                                  base_url=base_url, api_key=api_key)


STREAM_FUNCS = {
    "anthropic": stream_anthropic,
    "ollama": stream_ollama,
    "groq": stream_groq,
    "openai": stream_openai,
    "openai_compat": stream_openai_compat,
    "gemini": stream_gemini,
}


def stream_chat(provider: str, model: str, messages: list, run_context_text: str = ""):
    func = STREAM_FUNCS.get(provider)
    if not func:
        print(f"Error: Unknown provider '{provider}'. Known: {', '.join(STREAM_FUNCS)}",
              file=sys.stderr)
        sys.exit(1)
    return func(messages, model, run_context_text)


def list_models():
    print("Known providers and default models:\n")
    for name, info in KNOWN_PROVIDERS.items():
        default = info.get("default_model", "?")
        env = info.get("env_key", info.get("env_host", ""))
        pip = info.get("pip", "")
        status = ""
        if env and os.environ.get(env):
            status = " [key set]"
        elif name == "ollama":
            status = " [no key needed]"
        print(f"  {name:16s} default_model={default:30s} pip={pip}{status}")
    print()
    config = load_config()
    if config.has_section("defaults"):
        p = config.get("defaults", "provider", fallback="?")
        m = config.get("defaults", "model", fallback="?")
        print(f"  Config default: provider={p}, model={m}  ({CONF_PATH})")
