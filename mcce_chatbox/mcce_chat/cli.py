"""CLI entry point and REPL loop for mcce-chat."""

import argparse
import os
import sys

from mcce_chat.scanner import scan_run_directory
from mcce_chat.llm import (
    resolve_provider_model, save_config, stream_chat, list_models, SYSTEM_PROMPT,
)
from mcce_chat.display import (
    print_banner, print_run_context, print_prompt, print_assistant_prefix,
    print_error, print_info, print_status_table,
)


def _get_rag_context(query: str, root_dir: str, no_rag: bool) -> str:
    if no_rag:
        return ""
    try:
        from mcce_chat.rag.retriever import hybrid_search, format_context
        chunks = hybrid_search(query, root_dir)
        return format_context(chunks)
    except Exception:
        return ""


def _build_user_message(user_input: str, rag_context: str) -> str:
    if rag_context:
        return f"{rag_context}\n\nUser question: {user_input}"
    return user_input


def _do_chat(user_input: str, messages: list, provider: str, model: str,
             run_context_text: str, root_dir: str, no_rag: bool):
    rag_context = _get_rag_context(user_input, root_dir, no_rag)
    content = _build_user_message(user_input, rag_context)
    messages.append({"role": "user", "content": content})

    print_assistant_prefix()
    full_response = []
    try:
        for token in stream_chat(provider, model, messages, run_context_text):
            sys.stdout.write(token)
            sys.stdout.flush()
            full_response.append(token)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print_error(str(e))
        messages.pop()
        return

    print()
    response_text = "".join(full_response)
    if response_text:
        messages.append({"role": "assistant", "content": response_text})


def main(root_dir: str = None):
    parser = argparse.ArgumentParser(
        prog="mcce-chat",
        description="MCCE4 AI assistant — ask questions about your MCCE4 run",
    )
    parser.add_argument("question", nargs="?", default=None,
                        help="Single-shot question (omit for interactive REPL)")
    parser.add_argument("--provider", "-p", default=None,
                        help="LLM provider (anthropic, ollama, groq, openai, openai_compat)")
    parser.add_argument("--model", "-m", default=None, help="Model name")
    parser.add_argument("--save-provider", action="store_true",
                        help="Save provider/model choice to ~/.mcce_chat.conf")
    parser.add_argument("--list-models", action="store_true",
                        help="List known providers and models")
    parser.add_argument("--rebuild-index", action="store_true",
                        help="Build or rebuild the RAG index from MCCE4 source")
    parser.add_argument("--status", action="store_true",
                        help="Print run directory context and exit")
    parser.add_argument("--no-rag", action="store_true",
                        help="Skip RAG retrieval (just use run context)")
    parser.add_argument("--show-context", action="store_true",
                        help="Print run context summary before chatting")
    parser.add_argument("--run-dir", default=None,
                        help="Override run directory (default: cwd)")

    args = parser.parse_args()

    if root_dir is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.list_models:
        list_models()
        return

    provider, model = resolve_provider_model(args.provider, args.model)

    if args.save_provider:
        save_config(provider, model)
        print(f"Saved default: provider={provider}, model={model}")
        return

    if args.rebuild_index:
        try:
            from mcce_chat.rag.indexer import build_index
            build_index(root_dir, force=True)
        except Exception as e:
            print_error(f"Failed to build index: {e}")
            sys.exit(1)
        return

    run_dir = args.run_dir or os.getcwd()
    ctx = scan_run_directory(run_dir)

    if args.status:
        print_status_table(ctx)
        return

    run_context_text = ctx.summary()
    pdb_id = ctx.protein.pdb_id if ctx.protein else ""

    if args.show_context:
        print_run_context(run_context_text)

    print_banner(run_dir, ctx.mode(), pdb_id, ctx.book_state, provider, model)

    if args.question:
        messages = []
        _do_chat(args.question, messages, provider, model,
                 run_context_text, root_dir, args.no_rag)
        return

    print_info("Type your question (Ctrl+D or 'exit' to quit)")
    print()
    messages = []

    while True:
        try:
            print_prompt()
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "/status":
            print_status_table(ctx)
            continue
        if user_input.lower() == "/context":
            print_run_context(run_context_text)
            continue
        if user_input.lower() == "/clear":
            messages.clear()
            print_info("Conversation cleared.")
            continue

        _do_chat(user_input, messages, provider, model,
                 run_context_text, root_dir, args.no_rag)
        print()
