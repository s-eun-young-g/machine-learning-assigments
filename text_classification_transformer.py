pip install -q transformers datasets accelerate evaluate scikit-learn

from transformers import set_seed
set_seed(0)

!rm -f emails.py
!wget --quiet --no-check-certificate https://introml.mit.edu/_static/fall25/homework/hw09/emails.py

from emails import emails

set_seed(0)

data = emails()

### convert emails() output to X_train/y_train/X_test/y_test ##
if isinstance(data, (tuple, list)) and len(data) == 4:
    X_train, y_train, X_test, y_test = data
elif isinstance(data, (tuple, list)) and len(data) == 2:
    X, y = data
    X_train, X_test, y_train, y_test = train_test_split(
        list(X), list(y), test_size=0.2, random_state=0, stratify=list(y)
    )
elif isinstance(data, dict) and "train" in data and "test" in data:
    X_train = [t for (t, l) in data["train"]]
    y_train = [l for (t, l) in data["train"]]
    X_test  = [t for (t, l) in data["test"]]
    y_test  = [l for (t, l) in data["test"]]
else:
    X = [t for (t, l) in data]
    y = [l for (t, l) in data]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

print("train:", len(X_train), "test:", len(X_test))
print("example:", X_train[0][:120], "...")
print("labels:", sorted(set(y_train)))

### make label mapping ###
label_list = sorted(list(set(y_train)))
label_to_id = {lab: i for i, lab in enumerate(label_list)}
id_to_label = {i: lab for lab, i in label_to_id.items()}

y_train_ids = [label_to_id[l] for l in y_train]
y_test_ids  = [label_to_id[l] for l in y_test]

### simple train/val split from the training set ###
X_train2, X_val, y_train2, y_val = train_test_split(
    X_train, y_train_ids, test_size=0.15, random_state=0, stratify=y_train_ids
)

### tokenizer + model ###
model_name = "distilbert-base-uncased"  # easiest for classification
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(label_list)
)

### dataset ###
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        # squeeze batch dimension
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

train_ds = EmailDataset(X_train2, y_train2, tokenizer)
val_ds   = EmailDataset(X_val, y_val, tokenizer)
test_ds  = EmailDataset(X_test, y_test_ids, tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=64)
test_loader  = DataLoader(test_ds, batch_size=64)

### training setup ###
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

def get_accuracy(dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].numel()
    return correct / total if total > 0 else 0.0

### train loop ###
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch}")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 200 == 0:
            print("  step", i + 1, "avg loss", running_loss / (i + 1))

    val_acc = get_accuracy(val_loader)
    print(f"epoch {epoch} avg loss={running_loss/len(train_loader):.4f} val_acc={val_acc:.4f}")

test_acc = get_accuracy(test_loader)
print("final test acc:", test_acc)

### simple prediction helper ###
def predict_one(text):
    model.eval()
    enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        pred = int(torch.argmax(logits, dim=-1).item())
    return id_to_label[pred]

print(predict_one("Congratulations! You won a prize, click here to claim!"))
print(predict_one("Hi, can we meet tomorrow at 2pm to discuss the project?"))
