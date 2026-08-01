"""
groq_client.py
Groq plays the role of "teacher" here: it takes the tiny model's raw output
(which may be pure gibberish early on, since its weights are random) and
returns a corrected/improved version. That corrected version is then used
as the training target for real backpropagation.
"""

import os
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


def _call(messages, temperature=0.5, max_tokens=200):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def correct_reply(persona_name, other_name, conversation_so_far, raw_reply, topic_hint=""):
    """
    Returns a corrected/improved version of the tiny model's reply, to be
    used as a training target. Groq acts as a language teacher here, not
    as a stand-in speaker for the tiny model.
    """
    system = (
        "You are a language teacher helping a tiny model that is training "
        "from scratch improve. You'll be given a raw reply (which may be "
        "garbled or incomplete since the model is still learning). "
        "Your job: write a short (one or two sentences, under 25 words), "
        "natural, understandable version that follows the same spirit or "
        "direction the model was trying to express, so it can learn to "
        "imitate it. Reply with the corrected text only, no preamble or "
        "explanation."
    )
    user = (
        f"Topic/context: {topic_hint}\n"
        f"Conversation so far:\n{conversation_so_far}\n\n"
        f"{persona_name} tried to say:\n\"{raw_reply}\"\n\n"
        f"Write a corrected, understandable version of {persona_name}'s line "
        f"(replying to {other_name})."
    )
    return _call([{"role": "system", "content": system}, {"role": "user", "content": user}])


def suggest_opening_topic():
    """A simple free topic to open the dialogue each time, to vary the training data."""
    system = "Suggest a short, simple discussion topic (one sentence) for a free-form dialogue between two characters. Reply with just the sentence."
    return _call([{"role": "system", "content": system}], temperature=1.0, max_tokens=40)
