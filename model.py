from wiki_loader import load_all_wiki
from preprocess import load_fever, prepare_data
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, random_split
from collections import Counter
import random

if __name__ == "__main__":

    # --------------------------
    # LOAD DATA
    # --------------------------
    print("Loading wiki...")
    wiki = load_all_wiki("data/wiki/", max_files=300)

    print("Loading FEVER...")
    data = load_fever("data/fever/train.jsonl")

    print("Preparing dataset...")
    dataset = prepare_data(data, wiki, limit=20000)  # 🔥 increased

    random.shuffle(dataset)

    labels_only = [x[2] for x in dataset]
    print("Label distribution:", Counter(labels_only))

    # --------------------------
    # DEVICE
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------
    # MODEL
    # --------------------------
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    ).to(device)

    # ❌ REMOVED FREEZE (VERY IMPORTANT)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # --------------------------
    # CLASS WEIGHTS
    # --------------------------
    label_counts = Counter(labels_only)
    total_samples = len(labels_only)

    class_weights = torch.tensor([
        total_samples / label_counts[0],
        total_samples / label_counts[1],
        total_samples / label_counts[2]
    ]).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # --------------------------
    # COLLATE FUNCTION
    # --------------------------
    def collate_fn(batch):
        claims = [x[0] for x in batch]
        evidences = [x[1] for x in batch]
        labels = torch.tensor([x[2] for x in batch])

        inputs = tokenizer(
            claims,
            evidences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        return inputs, labels

    # --------------------------
    # SPLIT DATA
    # --------------------------
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # --------------------------
    # SCHEDULER
    # --------------------------
    total_steps = len(train_loader) * 4
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # --------------------------
    # AMP
    # --------------------------
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # --------------------------
    # TRAINING
    # --------------------------
    print("Training...")

    best_val_acc = 0
    patience = 2
    patience_counter = 0

    for epoch in range(4):  # 🔥 increased epochs

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(**inputs)
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100

        # ---------------- VALIDATION ----------------
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100

        print(
            f"Epoch {epoch+1} | Loss: {total_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        # ---------------- EARLY STOPPING ----------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            print("💾 Saving best model...")
            model.save_pretrained("saved_model")
            tokenizer.save_pretrained("saved_model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⛔ Early stopping triggered")
                break

    print("✅ Training complete!")