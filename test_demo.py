from predict import predict_with_evidence_list
from retrieval import Retriever
from wiki_loader import load_all_wiki

# --------------------------
# LOAD SYSTEM
# --------------------------
wiki = load_all_wiki("data/wiki/", max_files=200)
retriever = Retriever(wiki)

# --------------------------
# DEMO DATASET
# --------------------------
demo_dataset = [
    ("The 2004 Formula 3 Sudamericana season was the 15th Formula 3 Sudamericana season", "SUPPORTS"),
    ("The 2004 Formula 3 Sudamericana season was the 20th season", "REFUTES"),
    ("The 2004 Formula 3 Sudamericana season was a football league season", "NOT ENOUGH INFO"),

    ("The 2014 FIBA 3x3 World Championships was an international basketball event", "SUPPORTS"),
    ("The 2014 FIBA 3x3 World Championships was a tennis tournament", "REFUTES"),
    ("The 2014 FIBA 3x3 World Championships was a cricket event in India", "NOT ENOUGH INFO"),
]

# --------------------------
# TEST LOOP
# --------------------------
correct = 0

for claim, true_label in demo_dataset:

    evidence = retriever.retrieve(claim, k=5)
    evidence_texts = [e["text"] for e in evidence]

    result = predict_with_evidence_list(claim, evidence_texts)

    pred = result["label"]

    print("\n-----------------------------")
    print("Claim:", claim)
    print("Expected:", true_label)
    print("Predicted:", pred)
    print("Confidence:", result.get("confidence"))

    if pred == true_label:
        correct += 1

# --------------------------
# FINAL ACCURACY
# --------------------------
accuracy = correct / len(demo_dataset) * 100
print("\n🔥 Accuracy:", accuracy)