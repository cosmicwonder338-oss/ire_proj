import json
import random


# --------------------------
# LOAD FEVER DATA
# --------------------------
def load_fever(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                if "claim" in item:
                    data.append(item)
            except:
                continue
    return data


# --------------------------
# EXTRACT EVIDENCE TEXT
# --------------------------
def get_evidence_text(evidence, wiki):
    texts = []

    for group in evidence:
        for ev in group:
            if len(ev) < 4:
                continue

            page = ev[2]
            sent_id = ev[3]

            if page is None or sent_id is None:
                continue

            if page in wiki:
                sentence = wiki[page].get(sent_id)

                # 🔥 strong filtering
                if sentence:
                    words = sentence.split()
                    if 5 <= len(words) <= 80:
                        texts.append(sentence)

    return texts


# --------------------------
# LABEL MAP
# --------------------------
def label_map(label):
    if label == "SUPPORTS":
        return 0
    elif label == "REFUTES":
        return 1
    else:
        return 2


# --------------------------
# NEGATIVE SAMPLING
# --------------------------
def get_random_sentence(wiki):
    page = random.choice(list(wiki.keys()))
    sentences = list(wiki[page].values())

    if not sentences:
        return "random unrelated sentence"

    return random.choice(sentences)


# --------------------------
# PREPARE DATASET (IMPROVED)
# --------------------------
def prepare_data(data, wiki, limit=20000):
    dataset = []

    for item in data:
        claim = item.get("claim", "").strip()

        if not claim:
            continue

        label = label_map(item.get("label", "NOT ENOUGH INFO"))

        evidence_list = get_evidence_text(item.get("evidence", []), wiki)

        # --------------------------
        # MULTI-EVIDENCE STRATEGY
        # --------------------------
        if label in [0, 1]:  # SUPPORTS / REFUTES
            if evidence_list:
                # 🔥 use multiple evidence combined
                evidence = " ".join(evidence_list[:2])
            else:
                evidence = "no evidence"

        else:  # NOT ENOUGH INFO
            # 🔥 negative sampling (VERY IMPORTANT)
            evidence = get_random_sentence(wiki)

        dataset.append((claim, evidence, label))

        if len(dataset) >= limit:
            break

    # --------------------------
    # BALANCING (OPTIONAL BUT POWERFUL)
    # --------------------------
    supports = [x for x in dataset if x[2] == 0]
    refutes = [x for x in dataset if x[2] == 1]
    nei = [x for x in dataset if x[2] == 2]

    min_size = min(len(supports), len(refutes), len(nei))

    balanced = (
        random.sample(supports, min_size) +
        random.sample(refutes, min_size) +
        random.sample(nei, min_size)
    )

    random.shuffle(balanced)

    print(f"Final dataset size: {len(balanced)} (balanced)")

    return balanced