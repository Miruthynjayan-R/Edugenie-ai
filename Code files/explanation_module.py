"""
explanation_module.py - Concept Explanation module

Uses a lightweight, locally-run instruction-tuned model
(MBZUAI/LaMini-Flan-T5-783M) to explain concepts in simple language.

The model is loaded lazily (on first request) rather than at import
time, so the server still starts even if the model hasn't been
downloaded yet or there's no internet connection at boot. The first
call to /explain/ will take longer while the model downloads
(~3 GB, one-time) and loads into memory.
"""

from functools import lru_cache

MODEL_NAME = "MBZUAI/LaMini-Flan-T5-783M"


@lru_cache(maxsize=1)
def _load_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def explain_topic(topic: str) -> str:
    """Generate a simple, student-friendly explanation of `topic`."""
    try:
        tokenizer, model = _load_model()

        input_text = (
            f"Explain the concept of '{topic}' in a simple and clear way "
            f"for a school student."
        )
        inputs = tokenizer(input_text, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )

        explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation
    except Exception as e:
        return (
            f"⚠️ Error in Explanation module: {e}. "
            "(First run downloads the model — ~3GB — and needs internet "
            "access plus a few GB of free disk space and RAM.)"
        )