
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from real_valued_logic import RealValuedLogic
from weighted_logic import WeightedLogic
from logical_mask import LogicalMask
from data_generation import generate_synthetic_data

# Define the Enhanced Logical Mask Function
def enhanced_logic_fn(Q, K):
    logic = torch.sigmoid(Q * K - 0.5) + torch.tanh(Q + K)
    return logic

# Transformer with Logical Masking and Complexity, and Embedding Layer
class TransformerWithLNN(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_classes, logic_fn, dropout_rate=0.3):
        super(TransformerWithLNN, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            batch_first=True,
            dropout=dropout_rate
        )
        self.fc = nn.Linear(d_model, num_classes)
        self.logic_mask = LogicalMask(logic_fn)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src, tgt):
        src_emb = self.embedding(src).unsqueeze(1)
        tgt_emb = self.embedding(tgt).unsqueeze(1)
        memory = self.transformer(src_emb, tgt_emb)
        output = self.fc(memory[:, -1])
        return self.dropout(output), src_emb

    def apply_logical_mask(self, Q, K):
        mask = self.logic_mask(Q, K)
        return mask

# Train and evaluate with cross-validation and CosineAnnealingWarmRestarts scheduler
def train_and_evaluate_with_CosineAnnealing(model, criterion, optimizer, train_loader, test_loader, num_epochs=15, T_0=5, T_mult=2):
    train_losses = []
    val_losses = []

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(inputs, inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step(epoch + (epoch / num_epochs))

        running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs, _ = model(inputs, inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = running_val_loss / len(test_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

    return train_losses, val_losses

def plot_learning_curves(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def print_embeddings(model, data_loader):
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            _, embeddings = model(inputs, inputs)
            print("Embeddings:")
            print(embeddings)
            break

if __name__ == "__main__":
    num_samples = 1000
    num_features = 64
    d_model = 128
    nhead = 4
    num_encoder_layers = 4
    num_classes = 2
    learning_rate = 0.001
    batch_size = 32

    X, y = generate_synthetic_data(num_samples, num_features)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    logic_fn = enhanced_logic_fn

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold + 1}")
        X_train, X_test = torch.tensor(X[train_idx].copy(), dtype=torch.float32), torch.tensor(X[test_idx].copy(), dtype=torch.float32)
        y_train, y_test = torch.tensor(y[train_idx].clone(), dtype=torch.long), torch.tensor(y[test_idx].clone(), dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = TransformerWithLNN(
            input_dim=num_features,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_classes=num_classes,
            logic_fn=logic_fn
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        train_losses, val_losses = train_and_evaluate_with_CosineAnnealing(
            model, criterion, optimizer, train_loader, test_loader, num_epochs=15
        )

        plot_learning_curves(train_losses, val_losses)

        model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs, _ = model(inputs, inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.numpy())
                all_predictions.extend(probabilities.numpy())
        y_true = np.array(all_labels)
        y_prob = np.array(all_predictions)
        y_pred = np.round(y_prob)

        accuracy = np.mean(y_pred == y_true)
        print(f'Fold {fold + 1} - Accuracy: {accuracy:.4f}')

        fpr, tpr, _ = roc_curve(y_true, y_prob)
