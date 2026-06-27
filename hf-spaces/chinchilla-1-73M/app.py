"""
Chinchilla-1-73M — Chat interface
Brand: monochromatic, Poppins / Playfair Display, minimal.
"""

import json
import sys
import time
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import snapshot_download

# ── Ensure nanochat package is importable ──
REPO_DIR = Path(__file__).parent
sys.path.insert(0, str(REPO_DIR))

from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

# ── Paths ──
MODEL_REPO = "rajofearth/Chinchilla-1-73M"
CACHE_DIR = REPO_DIR / "model_cache"
CHECKPOINT_DIR = CACHE_DIR / "checkpoints" / "sft"
TOKENIZER_DIR = CACHE_DIR / "checkpoints" / "tokenizer"
MODEL_PATH = CHECKPOINT_DIR / "chinchilla-1-73M.pt"
META_PATH = CHECKPOINT_DIR / "chinchilla-1-73M.json"
TOKENIZER_PKL = TOKENIZER_DIR / "tokenizer.pkl"

DEVICE = torch.device("cpu")

_model = None
_tokenizer = None
_meta = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def download_model():
    if MODEL_PATH.exists():
        print("✓ Model already cached.")
        return
    print(f"↓ Downloading model from {MODEL_REPO}...")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=str(CACHE_DIR),
        local_dir_use_symlinks=False,
    )
    print("✓ Download complete.")


def load_model():
    global _model, _tokenizer, _meta

    if _model is not None:
        return _model, _tokenizer, _meta

    print("Loading tokenizer...")
    if not TOKENIZER_PKL.exists():
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PKL}.")
    _tokenizer = RustBPETokenizer.from_directory(str(TOKENIZER_DIR))

    print("Loading model config...")
    with open(META_PATH) as f:
        _meta = json.load(f)
    cfg_kwargs = _meta["model_config"]
    cfg_kwargs.setdefault("window_pattern", "L")
    model_config = GPTConfig(**cfg_kwargs)

    print(f"Loading weights from {MODEL_PATH.name}...")
    model_data = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    model_data = {
        k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()
    }

    print("Building model graph...")
    with torch.device("meta"):
        _model = GPT(model_config)
    _model.to_empty(device=DEVICE)
    _model.init_weights()
    _model.load_state_dict(model_data, strict=True, assign=True)
    _model.eval()

    n_params = sum(p.numel() for p in _model.parameters())
    print(f"✓ Model loaded ({n_params:,} params, {DEVICE})")
    return _model, _tokenizer, _meta


# (no system prompt — model runs raw)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_chat_prompt(tokenizer, history_tuples, new_message):
    """Build token prompt in nanochat conversation format."""
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    tokens = []

    for user_msg, bot_msg in history_tuples:
        tokens.extend([bos, user_start])
        tokens.extend(tokenizer.encode(user_msg))
        tokens.append(user_end)
        if bot_msg is not None and bot_msg.strip():
            tokens.append(assistant_start)
            tokens.extend(tokenizer.encode(bot_msg))
            tokens.append(assistant_end)

    # ── Current user message ──
    tokens.extend([bos, user_start])
    tokens.extend(tokenizer.encode(new_message))
    tokens.extend([user_end, assistant_start])

    return tokens


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_stream(message, history, temperature, max_tokens, top_k):
    """Yields partial response strings for streaming into the chatbot."""
    model, tokenizer, meta = _model, _tokenizer, _meta
    if model is None:
        yield "⚠ Model not loaded. Please wait and try again."
        return

    # Convert messages format (list of dicts) to tuples for build_chat_prompt
    history_tuples = []
    i = 0
    while i < len(history):
        if history[i]["role"] == "user":
            user_msg = history[i]["content"]
            bot_msg = ""
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                bot_msg = history[i + 1]["content"]
                i += 1
            history_tuples.append((user_msg, bot_msg))
        i += 1

    prompt_ids = build_chat_prompt(tokenizer, history_tuples, message)

    max_context = model.config.sequence_len
    prompt_max = max_context - max_tokens
    if len(prompt_ids) > prompt_max:
        prompt_ids = [prompt_ids[0]] + prompt_ids[-(prompt_max - 1) :]

    assistant_end_id = tokenizer.encode_special("<|assistant_end|>")
    bos_id = tokenizer.get_bos_token_id()
    stop_ids = {assistant_end_id, bos_id}

    engine = Engine(model, tokenizer)
    stream = engine.generate(
        prompt_ids,
        num_samples=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=42,
    )

    response = ""
    num_tokens = 0
    for token_column, _ in stream:
        token = token_column[0]
        if token in stop_ids:
            break
        response += tokenizer.decode([token])
        num_tokens += 1
        yield response, num_tokens


# ---------------------------------------------------------------------------
# CSS — brand: monochromatic, Poppins
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }

.gradio-container {
    background: #111111 !important;
    font-family: 'Poppins', sans-serif !important;
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding: 0 24px !important;
    width: 100% !important;
}

footer, .footer, .built-with { display: none !important; }

/* ────────────────────────────────
   Header
──────────────────────────────── */
#app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 0 22px;
    border-bottom: 1px solid #222;
    margin-bottom: 24px;
}

#logo-mark {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    border: 1.5px solid #333;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    overflow: hidden;
}
#logo-mark img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    filter: invert(1);
}
#logo-mark .logo-fallback { font-size: 20px; line-height: 1; }

#app-title {
    font-family: 'Poppins', sans-serif;
    font-size: 0.78rem;
    font-weight: 300;
    color: #F5F5F5;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    margin: 0;
    line-height: 1;
}
#app-wordmark-divider {
    width: 100%;
    height: 1px;
    background: #333;
    margin: 5px 0;
}
#app-subtitle-word {
    font-family: 'Poppins', sans-serif;
    font-size: 0.6rem;
    font-weight: 300;
    color: #555;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    margin: 0;
}
#app-meta {
    margin-top: 6px;
    font-size: 0.65rem;
    color: #444;
    letter-spacing: 0.04em;
}

/* ────────────────────────────────
   Chatbot
──────────────────────────────── */
#chatbot {
    background: #161616 !important;
    border: 1px solid #222 !important;
    border-radius: 3px !important;
}

/* ── User messages ── */
#chatbot .bubble-wrap.user .message-wrap,
#chatbot [data-testid="user"] .message-wrap,
#chatbot .bubble-wrap.user .prose,
#chatbot [data-testid="user"] .prose {
    background: #202020 !important;
    color: #E8E8E8 !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 3px !important;
    font-size: 0.875rem !important;
    line-height: 1.6 !important;
    font-family: 'Poppins', sans-serif !important;
    max-width: 88% !important;
}

/* ── Bot messages ── */
#chatbot .bubble-wrap.bot .message-wrap,
#chatbot [data-testid="bot"] .message-wrap,
#chatbot .bubble-wrap.bot .prose,
#chatbot [data-testid="bot"] .prose {
    background: transparent !important;
    color: #CCCCCC !important;
    border: 1px solid #1D1D1D !important;
    border-radius: 3px !important;
    font-size: 0.875rem !important;
    line-height: 1.65 !important;
    font-family: 'Poppins', sans-serif !important;
    max-width: 96% !important;
}

/* ────────────────────────────────
   Input row
──────────────────────────────── */
#input-row {
    margin-top: 8px !important;
    gap: 8px !important;
    align-items: flex-end !important;
}

#chat-input textarea {
    background: #161616 !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 3px !important;
    color: #F0F0F0 !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.875rem !important;
    padding: 12px 14px !important;
    line-height: 1.5 !important;
    transition: border-color 0.12s ease !important;
    resize: none !important;
    outline: none !important;
    box-shadow: none !important;
}
#chat-input textarea:focus {
    border-color: #555 !important;
}
#chat-input textarea::placeholder { color: #3A3A3A !important; }

#send-btn,
#send-btn > button {
    height: 46px !important;
    background: #F5F5F5 !important;
    color: #111111 !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    min-width: 76px !important;
    cursor: pointer !important;
    transition: background 0.12s !important;
    flex-shrink: 0 !important;
    box-shadow: none !important;
}
#send-btn:hover,
#send-btn > button:hover { background: #CCCCCC !important; }
#send-btn:active,
#send-btn > button:active { background: #AAAAAA !important; }

/* ── Action buttons row ── */
#action-row {
    margin-top: 6px !important;
    justify-content: flex-end !important;
    gap: 6px !important;
}

#clear-btn,
#clear-btn > button {
    background: transparent !important;
    border: 1px solid #1E1E1E !important;
    color: #444 !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 4px 12px !important;
    border-radius: 2px !important;
    cursor: pointer !important;
    transition: all 0.12s !important;
    box-shadow: none !important;
}
#clear-btn:hover,
#clear-btn > button:hover {
    border-color: #444 !important;
    color: #888 !important;
}

/* ── Generation stats (above input box) ── */
#gen-stats {
    font-size: 0.6rem !important;
    color: #3A3A3A !important;
    font-family: 'Poppins', sans-serif !important;
    text-align: right !important;
    padding: 2px 4px 0 0 !important;
    font-variant-numeric: tabular-nums !important;
}

/* ────────────────────────────────
   Sidebar
──────────────────────────────── */
#sidebar {
    padding-left: 28px !important;
    border-left: 1px solid #1E1E1E !important;
    padding-top: 0 !important;
}

.s-label {
    display: block;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3A3A3A;
    font-family: 'Poppins', sans-serif;
    padding-bottom: 7px;
    border-bottom: 1px solid #1E1E1E;
    margin-bottom: 14px;
    margin-top: 22px;
}
.s-label:first-child { margin-top: 0; }

/* ── Sliders ── */
.slider-wrap { margin-bottom: 16px !important; }
.slider-wrap > label > span:first-child {
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #BDBDBD !important;
    letter-spacing: 0.01em !important;
}
.slider-wrap .info {
    font-size: 0.66rem !important;
    color: #3A3A3A !important;
    margin-top: 2px !important;
}

.slider-wrap input[type="range"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    height: 1px !important;
    background: #2A2A2A !important;
    border-radius: 1px !important;
    outline: none !important;
    margin-top: 8px !important;
    width: 100% !important;
}
.slider-wrap input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    width: 11px !important;
    height: 11px !important;
    border-radius: 50% !important;
    background: #F5F5F5 !important;
    cursor: pointer !important;
    border: none !important;
}
.slider-wrap input[type="range"]::-moz-range-thumb {
    width: 11px !important;
    height: 11px !important;
    border-radius: 50% !important;
    background: #F5F5F5 !important;
    cursor: pointer !important;
    border: none !important;
}

.slider-wrap input[type="number"] {
    background: #161616 !important;
    border: 1px solid #222 !important;
    border-radius: 2px !important;
    color: #888 !important;
    font-size: 0.72rem !important;
    font-family: 'Poppins', sans-serif !important;
    padding: 2px 6px !important;
    text-align: center !important;
}

/* ── Model info table ── */
#model-info { margin-top: 0 !important; }
#model-info > .prose > p:first-child {
    font-size: 0.68rem !important;
    color: #444 !important;
    margin-bottom: 10px !important;
    line-height: 1.5 !important;
    font-family: 'Poppins', sans-serif !important;
}
#model-info strong { color: #777 !important; font-weight: 500 !important; }

#model-info table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-size: 0.7rem !important;
    font-family: 'Poppins', sans-serif !important;
}
#model-info th {
    text-align: left !important;
    padding: 6px 8px !important;
    background: #1A1A1A !important;
    color: #3A3A3A !important;
    font-weight: 400 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-size: 0.6rem !important;
    border-bottom: 1px solid #222 !important;
}
#model-info td {
    padding: 5px 8px !important;
    color: #777 !important;
    border-bottom: 1px solid #1A1A1A !important;
    font-size: 0.7rem !important;
    font-variant-numeric: tabular-nums !important;
}
#model-info tr:last-child td { border-bottom: none !important; }

/* ── Limitations ── */
#limitations { margin-top: 0 !important; }
#limitations > .prose > p {
    font-size: 0.68rem !important;
    color: #3A3A3A !important;
    line-height: 1.6 !important;
    font-family: 'Poppins', sans-serif !important;
    margin-bottom: 6px !important;
}
#limitations ul {
    padding-left: 12px !important;
    margin: 0 !important;
}
#limitations li {
    font-size: 0.68rem !important;
    color: #3A3A3A !important;
    line-height: 1.65 !important;
    font-family: 'Poppins', sans-serif !important;
    margin-bottom: 3px !important;
}
#limitations strong { color: #555 !important; font-weight: 500 !important; }
"""

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

MODEL_INFO_MD = """\
**Chinchilla · 1 · 73M** — nanochat d6 architecture

| Component | Value |
|---|---|
| Params | 73,531,646 |
| Layers | 6 |
| Attention | 6 heads, full MHA |
| Vocab | 32,768 BPE |
| Activation | ReLU² |
| Precision | float32 (CPU) |
"""

LIMITATIONS_MD = """\
Research-scale model — smaller than GPT-2 Small.

- **Hallucinates** — do not use for factual Q&A
- **Rambles** beyond ~50 tokens — keep Max Tokens low
- **Poor** at arithmetic and format constraints
- Incompatible with llama.cpp / Ollama / Transformers
"""

LOGO_HTML = """
<div id="app-header">
  <div id="logo-mark">
    <img src="/file=logo.png" alt="" onerror="this.style.display='none';this.nextSibling.style.display='flex';">
    <span class="logo-fallback" style="display:none;">▣</span>
  </div>
  <div id="header-text">
    <div id="app-title">Chinchilla</div>
    <div id="app-wordmark-divider"></div>
    <div id="app-subtitle-word">Model</div>
    <div id="app-meta">73.5M params · CPU inference</div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def create_ui():
    with gr.Blocks(css=CSS, title="Chinchilla-1-73M") as demo:
        gr.HTML(LOGO_HTML)

        with gr.Row(elem_id="main-row", equal_height=False):
            # ── Chat column ──────────────────────────────
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    height=560,
                    show_label=False,
                    render_markdown=True,
                    type="messages",
                    placeholder=(
                        "<div style='text-align:center;color:#2A2A2A;"
                        "font-family:Poppins,sans-serif;font-size:0.75rem;"
                        "letter-spacing:0.06em;padding:40px 0'>"
                        "SEND A MESSAGE TO BEGIN"
                        "</div>"
                    ),
                )

                gen_stats = gr.HTML(
                    value="",
                    elem_id="gen-stats",
                )

                with gr.Row(elem_id="input-row"):
                    msg_input = gr.Textbox(
                        placeholder="Send a message…",
                        show_label=False,
                        elem_id="chat-input",
                        lines=1,
                        max_lines=5,
                        scale=5,
                        container=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button(
                        "Send",
                        elem_id="send-btn",
                        scale=1,
                        min_width=76,
                        variant="secondary",
                    )

                with gr.Row(elem_id="action-row"):
                    clear_btn = gr.Button(
                        "Clear history",
                        size="sm",
                        elem_id="clear-btn",
                        variant="secondary",
                    )

            # ── Sidebar ──────────────────────────────────
            with gr.Column(scale=1, min_width=210, elem_id="sidebar"):
                gr.HTML('<span class="s-label">Generation</span>')

                temperature = gr.Slider(
                    0.0,
                    2.0,
                    value=0.7,
                    step=0.05,
                    label="Temperature",
                    info="Deterministic ←→ Random",
                    elem_classes=["slider-wrap"],
                )
                max_tokens = gr.Slider(
                    16,
                    512,
                    value=64,
                    step=16,
                    label="Max tokens",
                    info="Keep ≤ 64 to limit rambling",
                    elem_classes=["slider-wrap"],
                )
                top_k = gr.Slider(
                    1,
                    100,
                    value=50,
                    step=1,
                    label="Top-K",
                    info="Sampling vocabulary size",
                    elem_classes=["slider-wrap"],
                )

                gr.HTML('<span class="s-label">Architecture</span>')
                gr.Markdown(MODEL_INFO_MD, elem_id="model-info")

                gr.HTML('<span class="s-label">Limitations</span>')
                gr.Markdown(LIMITATIONS_MD, elem_id="limitations")

        # ── Event wiring ────────────────────────────────

        def user_turn(message, history):
            """Immediately append user message; bot slot starts empty."""
            return "", history + [{"role": "user", "content": message}]

        def bot_turn(history, temp, max_tok, topk):
            """Stream bot response token by token, yield stats too."""
            user_msg = history[-1]["content"]
            prev_history = history[:-1]
            history.append({"role": "assistant", "content": ""})
            start = time.time()
            final_count = 0
            for partial, nt in generate_stream(
                user_msg, prev_history, temp, max_tok, topk
            ):
                history[-1]["content"] = partial
                final_count = nt
                elapsed = time.time() - start
                tps = final_count / elapsed if elapsed > 0 else 0
                stats = f"{final_count} tok · {tps:.1f} tok/s"
                yield history, stats

        msg_input.submit(
            user_turn, [msg_input, chatbot], [msg_input, chatbot], queue=False
        ).then(
            bot_turn, [chatbot, temperature, max_tokens, top_k], [chatbot, gen_stats]
        )

        send_btn.click(
            user_turn, [msg_input, chatbot], [msg_input, chatbot], queue=False
        ).then(
            bot_turn, [chatbot, temperature, max_tokens, top_k], [chatbot, gen_stats]
        )

        clear_btn.click(lambda: ([], "", ""), None, [chatbot, msg_input, gen_stats])

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_model()

    print("Pre-loading model...")
    t0 = time.time()
    try:
        load_model()
        print(f"✓ Model loaded in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"✗ Failed to load model: {e}", file=sys.stderr)
        raise

    demo = create_ui()
    demo.launch()
