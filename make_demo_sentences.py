from wiki_loader import load_all_wiki
import random

# --------------------------
# CONFIG
# --------------------------
MAX_FILES = 200
NUM_SAMPLES = 20
MIN_WORDS = 6
MAX_WORDS = 30

# --------------------------
# LOAD WIKI
# --------------------------
print("📚 Loading wiki...")
wiki = load_all_wiki("data/wiki/", max_files=MAX_FILES)

# --------------------------
# COLLECT CLEAN SENTENCES
# --------------------------
sentences = []

for page, sents in wiki.items():
    for sent_id, text in sents.items():

        words = text.split()

        # 🔥 strong filtering
        if MIN_WORDS <= len(words) <= MAX_WORDS:
            sentences.append({
                "page": page,
                "text": text
            })

print(f"\n✅ Total clean candidate sentences: {len(sentences)}")

# --------------------------
# SAFETY CHECK
# --------------------------
if len(sentences) == 0:
    print("❌ No valid sentences found. Check wiki_loader filtering.")
    exit()

# --------------------------
# RANDOM SAMPLING
# --------------------------
num_samples = min(NUM_SAMPLES, len(sentences))
demo = random.sample(sentences, num_samples)

# --------------------------
# DISPLAY OUTPUT
# --------------------------
print("\n===== 🔍 DEMO SENTENCES =====\n")

for i, item in enumerate(demo):
    print(f"{i+1}. 📄 [{item['page']}]")
    print(f"   ➤ {item['text']}\n")

# --------------------------
# EXTRA DEBUG INFO
# --------------------------
lengths = [len(s["text"].split()) for s in sentences]

print("📊 Stats:")
print(f"   • Avg length: {sum(lengths)/len(lengths):.2f} words")
print(f"   • Min length: {min(lengths)} words")
print(f"   • Max length: {max(lengths)} words")