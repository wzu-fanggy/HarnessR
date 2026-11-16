from __future__ import annotations



import json

import os





def estimate_tokens(text: str, model_name: str = "deepseek-v4-pro") -> int:

    """Estimate the number of tokens consumed by ``text`` for ``model_name``.



    Uses the model's tokenizer when available; otherwise falls back to a

    conservative character heuristic (``len(text) // 3``) for CJK-heavy text.

    """

    if "deepseek" in model_name or "gpt" in model_name:

        try:

            import tiktoken



            enc = tiktoken.encoding_for_model(model_name)

            return len(enc.encode(text))

        except Exception:

            pass

    return len(text) // 3





def get_context_limit() -> int:

    """Return the model context-window limit from ``orion/model/config.json``.



    Falls back to ``128000`` if the config file is missing or unreadable.

    """

    config_path = os.path.join(

        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),

        "model",

        "config.json",

    )

    try:

        with open(config_path, "r", encoding="utf-8") as f:

            return json.load(f).get("context_windows_limit", 128000)

    except Exception:

        return 128000

