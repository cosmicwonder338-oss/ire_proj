import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# --------------------------
# DEVICE
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# LOAD MODEL (FIXED)
# --------------------------
MODEL_PATH = "saved_model"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)

model.eval()

labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


# --------------------------
# SAFE TOKENIZATION
# --------------------------
def encode(claim, evidence):
    return tokenizer(
        claim,
        evidence,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256  # ⚡ matched with training
    ).to(device)


# --------------------------
# SINGLE PREDICTION
# --------------------------
def predict(claim, evidence):

    if not evidence or not evidence.strip():
        evidence = "no evidence"

    inputs = encode(claim, evidence)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=1)[0]

    scores = {
        "SUPPORTS": float(probs[0] * 100),
        "REFUTES": float(probs[1] * 100),
        "NOT ENOUGH INFO": float(probs[2] * 100),
    }

    label = labels[torch.argmax(probs).item()]

    return {
        "label": label,
        "scores": scores,
        "confidence": float(torch.max(probs).item() * 100),
        "logits": logits[0]
    }


# --------------------------
# MULTI-EVIDENCE AGGREGATION
# --------------------------
def predict_with_evidence_list(claim, evidence_list):

    if not evidence_list:
        r = predict(claim, "no evidence")
        r["used_evidence"] = "no evidence"
        return r

    best_result = None
    best_conf = -1

    sum_scores = {
        "SUPPORTS": 0.0,
        "REFUTES": 0.0,
        "NOT ENOUGH INFO": 0.0
    }

    valid_evidence_count = 0

    for ev in evidence_list:

        if not ev or len(ev.split()) < 3:
            continue

        result = predict(claim, ev)
        scores = result["scores"]
        conf = result["confidence"]

        # accumulate for UI
        for k in sum_scores:
            sum_scores[k] += scores[k]

        valid_evidence_count += 1

        # 🔥 best evidence selection
        if conf > best_conf:
            best_conf = conf
            best_result = {
                "label": result["label"],
                "scores": scores,
                "used_evidence": ev,
                "confidence": conf
            }

    # fallback if nothing valid
    if valid_evidence_count == 0:
        return {
            "label": "NOT ENOUGH INFO",
            "scores": {
                "SUPPORTS": 0.0,
                "REFUTES": 0.0,
                "NOT ENOUGH INFO": 100.0
            },
            "used_evidence": "no valid evidence",
            "confidence": 100.0
        }

    # average scores (for UI only)
    avg_scores = {
        k: v / valid_evidence_count for k, v in sum_scores.items()
    }

    return {
        "label": best_result["label"],
        "scores": avg_scores,
        "used_evidence": best_result["used_evidence"],
        "confidence": best_result["confidence"]
    }