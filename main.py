import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, roc_curve
import seaborn as sns

# -------------------- CNN Block -------------------- #
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# ---------------- Transformer Block ---------------- #
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)
        return x

# ------------------ Full Model --------------------- #
class HybridCNNTransformer(nn.Module):
    def __init__(self, lob_features, sentiment_dim, cnn_channels=64, d_model=64, nhead=4, num_layers=2, num_classes=3):
        super(HybridCNNTransformer, self).__init__()
        self.cnn = CNNBlock(in_channels=lob_features, out_channels=cnn_channels, kernel_size=3)
        self.transformer = TransformerBlock(d_model=cnn_channels, nhead=nhead, num_layers=num_layers)
        self.sentiment_proj = nn.Linear(sentiment_dim, cnn_channels)
        self.classifier = nn.Sequential(
            nn.Linear(cnn_channels * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, lob, sentiment):
        x = lob.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.transformer(x)
        x = x.mean(dim=2)
        sentiment_feat = self.sentiment_proj(sentiment)
        combined = torch.cat([x, sentiment_feat], dim=1)
        out = self.classifier(combined)
        return out

# ------------------ Custom Dataset ----------------- #
class LOBSentDataset(Dataset):
    def __init__(self, lob_data, sentiment_vecs, labels):
        self.lob_data = lob_data
        self.sentiment_vecs = sentiment_vecs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.lob_data[idx], dtype=torch.float32),
            torch.tensor(self.sentiment_vecs[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# ------------------ Load Sentiment ----------------- #
def get_sentiment_scores(texts):
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    result = sentiment_pipeline(texts)
    enc = {'positive': [1, 0, 0], 'neutral': [0, 1, 0], 'negative': [0, 0, 1]}
    return np.array([enc[r['label']] for r in result])

# ------------------ Main Execution ----------------- #
def main():
    os.makedirs("plots", exist_ok=True)

    file_path = "fi2010.csv"
    df = pd.read_csv(file_path)

    num_samples = 1000
    seq_len = 100
    lob_features = 40

    X = df.iloc[:num_samples * seq_len, 3:3+lob_features].values.reshape(num_samples, seq_len, lob_features)
    y_full = df.iloc[:num_samples * seq_len]['LABEL_1TICK'].values.reshape(num_samples, seq_len)
    label_map = {-1: 0, 0: 1, 1: 2}
    y = np.vectorize(label_map.get)(y_full[:, -1])

    example_texts = ["The company performed well with increased profits." for _ in range(num_samples)]
    sentiment_vecs = get_sentiment_scores(example_texts)

    dataset = LOBSentDataset(X, sentiment_vecs, y)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = HybridCNNTransformer(lob_features=lob_features, sentiment_dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses, val_accuracies, train_accuracies = [], [], [], []

    for epoch in range(10):
        model.train()
        total_loss, correct_train, total_train = 0, 0, 0
        for lob, sent, label in train_loader:
            optimizer.zero_grad()
            output = model(lob, sent)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = output.argmax(dim=1)
            correct_train += (preds == label).sum().item()
            total_train += label.size(0)

        train_acc = correct_train / total_train
        train_losses.append(total_loss / len(train_loader))
        train_accuracies.append(train_acc)

        model.eval()
        correct, total, val_loss = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for lob, sent, label in val_loader:
                output = model(lob, sent)
                loss = criterion(output, label)
                val_loss += loss.item()
                preds = output.argmax(dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(label.tolist())
                correct += (preds == label).sum().item()
                total += label.size(0)

        val_losses.append(val_loss / len(val_loader))
        acc = correct / total
        val_accuracies.append(acc)

        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {acc:.4f}")

    # Save loss and accuracy plots
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss over Epochs")
    plt.savefig("plots/loss_curve.png")
    plt.clf()

    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")
    plt.savefig("plots/accuracy_curve.png")
    plt.clf()

    # Metrics
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc_final = accuracy_score(all_labels, all_preds)

    pd.DataFrame(cm).to_csv("plots/confusion_matrix.csv", index=False)
    pd.DataFrame({
        'Train Loss': train_losses,
        'Val Loss': val_losses,
        'Train Acc': train_accuracies,
        'Val Acc': val_accuracies
    }).to_csv("plots/metrics_log.csv", index_label='Epoch')

    with open("plots/metrics_summary.txt", "w") as f:
        f.write(f"Final F1 Score: {f1:.4f}\nFinal Accuracy: {acc_final:.4f}\n")

    if len(set(all_labels)) == 2:
        probs = F.softmax(torch.tensor(all_preds), dim=1)[:, 1].numpy()
        fpr, tpr, _ = roc_curve(all_labels, probs)
        auc_score = roc_auc_score(all_labels, all_preds)
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        plt.legend()
        plt.title("ROC Curve")
        plt.savefig("plots/roc_curve.png")

if __name__ == "__main__":
    main()
