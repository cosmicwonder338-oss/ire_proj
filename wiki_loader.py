import json
import os


def clean_text(text):
    return text.strip()


def load_all_wiki(folder_path, max_files=None):
    wiki = {}

    files = [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]
    files = sorted(files)

    if max_files is not None:
        files = files[:max_files]

    print(f"📚 Loading {len(files)} wiki files...")

    total_sentences = 0

    for i, file in enumerate(files):
        path = os.path.join(folder_path, file)

        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except:
                        continue

                    page = item.get("id", "")
                    if not page:
                        continue

                    sentences = {}

                    raw_lines = item.get("lines", "").split("\n")

                    for l in raw_lines:
                        if not l.strip():
                            continue

                        parts = l.split("\t")

                        if len(parts) < 2:
                            continue

                        if not parts[0].isdigit():
                            continue

                        idx = int(parts[0])
                        text = clean_text(parts[1])

                        # 🔥 stronger filtering
                        word_count = len(text.split())

                        if word_count < 5 or word_count > 80:
                            continue

                        sentences[idx] = text
                        total_sentences += 1

                    if sentences:
                        wiki[page] = sentences

        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")
            continue

        # progress log
        if (i + 1) % 25 == 0:
            print(f"  Loaded {i+1}/{len(files)} files | Pages: {len(wiki)} | Sentences: {total_sentences}")

    print(f"\n✅ Done!")
    print(f"📄 Total pages: {len(wiki)}")
    print(f"🧾 Total sentences: {total_sentences}")

    return wiki