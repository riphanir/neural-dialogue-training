"""
run_cycle.py
One full cycle: the two tiny models talk to each other (using their current
weights), Groq corrects each reply, and each model is then actually trained
(backprop) on its corrected version. Weights are saved at the end of the cycle.
"""

import os
import json
import datetime
import torch

from tiny_transformer import load_or_init_model, save_model, count_parameters
from trainer import train_on_text
import groq_client

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
CONV_DIR = os.path.join(ROOT, "conversations")
STATE_DIR = os.path.join(ROOT, "state")

N_EXCHANGES = int(os.environ.get("N_EXCHANGES", "6"))  # total replies in the dialogue
TRAIN_EPOCHS_PER_TURN = int(os.environ.get("TRAIN_EPOCHS_PER_TURN", "3"))


def weights_path(model_key):
    return os.path.join(MODELS_DIR, model_key, "weights.pt")


def stats_path(model_key):
    return os.path.join(MODELS_DIR, model_key, "stats.json")


def load_stats(model_key):
    p = stats_path(model_key)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "cycle_count": 0,
        "total_training_steps": 0,
        "loss_history": [],  # last N values (loss at end of each cycle) to track progress
        "last_updated": None,
    }


def save_stats(model_key, stats):
    with open(stats_path(model_key), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def run():
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not found in environment variables")

    os.makedirs(CONV_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)

    # load both models (or create them with random weights on first run)
    model_a, a_is_new = load_or_init_model(weights_path("model_a"))
    model_b, b_is_new = load_or_init_model(weights_path("model_b"))

    if a_is_new:
        print(f"New random-weight init for model_a ({count_parameters(model_a):,} params)")
    if b_is_new:
        print(f"New random-weight init for model_b ({count_parameters(model_b):,} params)")

    stats_a = load_stats("model_a")
    stats_b = load_stats("model_b")

    topic = groq_client.suggest_opening_topic()

    transcript = []           # raw output from each tiny model (pre-correction)
    corrected_transcript = []  # corrected versions (used as conversation context)

    speakers = [("Aria", model_a, "model_a", stats_a), ("Nox", model_b, "model_b", stats_b)]
    conversation_context = f"Topic: {topic}\n"

    for i in range(N_EXCHANGES):
        name, model, key, stats = speakers[i % 2]
        other_name = speakers[(i + 1) % 2][0]

        # 1) the tiny model generates a reply from its current weights
        #    (may be gibberish early on)
        raw_reply = model.generate(conversation_context, max_new_tokens=80)
        transcript.append((name, raw_reply, "raw"))

        # 2) Groq corrects/improves the reply (this becomes the training target)
        corrected = groq_client.correct_reply(
            persona_name=name,
            other_name=other_name,
            conversation_so_far=conversation_context,
            raw_reply=raw_reply,
            topic_hint=topic,
        )
        corrected_transcript.append((name, corrected))

        # 3) real training: backprop on the corrected version
        first_loss, last_loss = train_on_text(model, corrected, epochs=TRAIN_EPOCHS_PER_TURN)
        if first_loss is not None:
            stats["total_training_steps"] += TRAIN_EPOCHS_PER_TURN
            stats["loss_history"].append(round(last_loss, 4))
            stats["loss_history"] = stats["loss_history"][-100:]  # keep only last 100 values
            print(f"{name}: loss {first_loss:.4f} -> {last_loss:.4f}")

        # update context with the corrected version (keeps the dialogue coherent)
        conversation_context += f"{name}: {corrected}\n"

    # save the updated weights (they actually changed via training)
    save_model(model_a, weights_path("model_a"))
    save_model(model_b, weights_path("model_b"))

    for key, stats in [("model_a", stats_a), ("model_b", stats_b)]:
        stats["cycle_count"] += 1
        stats["last_updated"] = datetime.datetime.utcnow().isoformat() + "Z"
        save_stats(key, stats)

    # save the dialogue log (raw + corrected) for the archive
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    md_path = os.path.join(CONV_DIR, f"{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Cycle {ts}\n\n**Topic:** {topic}\n\n")
        f.write("## Raw model output (before Groq correction)\n\n")
        for name, text, _ in transcript:
            f.write(f"**{name} (raw):** {text}\n\n")
        f.write("## Corrected version (used as training target)\n\n")
        for name, text in corrected_transcript:
            f.write(f"**{name}:** {text}\n\n")

    print(f"Saved log to {md_path}")
    print(f"Aria: cycle #{stats_a['cycle_count']}, total training steps: {stats_a['total_training_steps']}")
    print(f"Nox: cycle #{stats_b['cycle_count']}, total training steps: {stats_b['total_training_steps']}")


if __name__ == "__main__":
    run()
