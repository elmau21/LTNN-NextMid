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

# Define Real-Valued Logic Operations
class RealValuedLogic(nn.Module):
    def __init__(self):
        super(RealValuedLogic, self).__init__()

    def forward(self, x, y):
        # Logical AND operation (element-wise minimum)
        and_op = torch.minimum(x, y)
        # Logical OR operation (element-wise maximum)
        or_op = torch.maximum(x, y)
        return and_op, or_op

# Define Weighted Logic with Activation Functions
class WeightedLogic(nn.Module):
    def __init__(self, activation_fn=nn.Sigmoid()):
        super(WeightedLogic, self).__init__()
        self.activation_fn = activation_fn

    def forward(self, x, y, weight, bias):
        # Weighted AND operation with activation function
        # Applying a weighted logical function: sigmoid(bias - weight * (1 - x) - weight * (1 - y))
        # This can be interpreted as a weighted logical AND with bias and weight adjustments
        weighted_and = self.activation_fn(bias - weight * (1 - x) - weight * (1 - y))
        return weighted_and

# Define the Logical Mask for Attention
class LogicalMask(nn.Module):
    def __init__(self, logic_fn):
        super(LogicalMask, self).__init__()
        self.logic_fn = logic_fn

    def forward(self, Q, K):
        # Apply the logical function to queries (Q) and keys (K)
        # This generates a mask based on the logical function provided
        mask = self.logic_fn(Q, K)
        return mask

# Transformer with Logical Masking and Complexity, and Embedding Layer
class LTNN(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_classes, logic_fn, dropout_rate=0.3):
        super(LTNN, self).__init__()
        # Embedding layer to project input into d_model space
        # This converts input features into a space of dimension d_model
        self.embedding = nn.Linear(input_dim, d_model)

        # Transformer layer
        self.transformer = nn.Transformer(
            d_model=d_model,           # Dimension of the model's embedding space
            nhead=nhead,               # Number of attention heads
            num_encoder_layers=num_encoder_layers,  # Number of encoder layers
            batch_first=True,          # Input tensors are expected to have batch dimension first
            dropout=dropout_rate       # Dropout rate for regularization
        )
        self.fc = nn.Linear(d_model, num_classes)  # Final fully connected layer for classification
        self.logic_mask = LogicalMask(logic_fn)    # Logical mask for attention mechanism
        self.dropout = nn.Dropout(dropout_rate)    # Dropout layer for regularization

    def forward(self, src, tgt):
        # Convert source and target inputs to embeddings
        src_emb = self.embedding(src).unsqueeze(1)  # Add sequence dimension for transformer
        tgt_emb = self.embedding(tgt).unsqueeze(1)  # Add sequence dimension for transformer

        # Pass embeddings through transformer layers
        memory = self.transformer(src_emb, tgt_emb)
        # Extract the output corresponding to the last token in the sequence
        output = self.fc(memory[:, -1])
        return self.dropout(output), src_emb  # Return logits and embeddings

    def apply_logical_mask(self, Q, K):
        # Apply the logical mask function to queries and keys
        mask = self.logic_mask(Q, K)
        return mask

# Define an Enhanced Logical Mask Function
def enhanced_logic_fn(Q, K):
    # Enhanced logical function combining sigmoid and tanh
    # The function adds complexity by combining sigmoid and tanh operations
    # sigmoid(Q * K - 0.5) introduces a non-linear transformation based on the product of Q and K
    # tanh(Q + K) introduces another non-linearity based on the sum of Q and K
    logic = torch.sigmoid(Q * K - 0.5) + torch.tanh(Q + K)
    return logic

# Generate synthetic data just for testing purposes.
def generate_synthetic_data(num_samples, num_features):
    # Generate random features
    X = np.random.rand(num_samples, num_features)
    # Add noise to create a binary classification target
    noise = np.random.normal(0, 0.1, num_samples)  # Gaussian noise
    y = (X.sum(axis=1) + noise > (num_features / 2)).astype(int)  # Binary target based on feature sum and noise
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# Train and evaluate with cross-validation and CosineAnnealingWarmRestarts scheduler
def CosineAnnealing(model, criterion, optimizer, train_loader, test_loader, num_epochs=15, T_0=5, T_mult=2):
    train_losses = []
    val_losses = []

    # Initialize the Cosine Annealing Warm Restarts scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(inputs, inputs)  # Forward pass through the model
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backward pass to compute gradients
            optimizer.step()  # Update model parameters

        # Update learning rate with cosine annealing warm restart
        scheduler.step(epoch + (epoch / num_epochs))

        # Log training loss for this epoch
        running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
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
    # Plot the training and validation loss curves
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
    # Plot the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def print_embeddings(model, data_loader):
    # Print the embeddings for the first batch of data
    model.eval()
    with torch.no_grad():
        for inputs, _ in data_loader:
            _, embeddings = model(inputs, inputs)  # Get embeddings
            print("Embeddings:")
            print(embeddings)
            break  # Only print the first batch for brevity

# Main script
if __name__ == "__main__":
    num_samples = 1000
    num_features = 64  # Increased feature complexity
    d_model = 128  # Set the model dimension for embeddings
    nhead = 4  # Number of attention heads
    num_encoder_layers = 4  # Depth of transformer layers
    num_classes = 2  # Number of output classes (binary classification)
    learning_rate = 0.001
    batch_size = 32

    X, y = generate_synthetic_data(num_samples, num_features)

    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Prepare data with K-fold Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    logic_fn = enhanced_logic_fn  # Use the enhanced logic function

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold + 1}")
        X_train, X_test = torch.tensor(X[train_idx].copy(), dtype=torch.float32), torch.tensor(X[test_idx].copy(), dtype=torch.float32)
        y_train, y_test = torch.tensor(y[train_idx].clone(), dtype=torch.long), torch.tensor(y[test_idx].clone(), dtype=torch.long)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        model = LTNN(
            input_dim=num_features,  # Embedding input size
            d_model=d_model,         # Dimension of model's embedding space
            nhead=nhead,             # Number of attention heads
            num_encoder_layers=num_encoder_layers,  # Number of encoder layers
            num_classes=num_classes,  # Number of output classes
            logic_fn=logic_fn        # Logical function for masking
        )
        criterion = nn.CrossEntropyLoss()  # Loss function for classification
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Optimizer with weight decay
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)  # Scheduler for learning rate adjustment

        # Train the model
        train_losses, val_losses = CosineAnnealing(
            model, criterion, optimizer, train_loader, test_loader, num_epochs=15
        )

        # Plot learning curves
        plot_learning_curves(train_losses, val_losses)

        # Evaluate on the test set
        model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs, _ = model(inputs, inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Get probabilities for positive class
                all_labels.extend(labels.numpy())
                all_predictions.extend(probabilities.numpy())
        y_true = np.array(all_labels)
        y_prob = np.array(all_predictions)
        y_pred = np.round(y_prob)  # Convert probabilities to binary predictions

        # Calculate Accuracy
        accuracy = np.mean(y_pred == y_true)
        print(f'Fold {fold + 1} - Accuracy: {accuracy:.4f}')

        # ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)  # Compute ROC curve
        roc_auc = auc(fpr, tpr)  # Compute ROC AUC
        print(f'ROC AUC: {roc_auc:.4f}')

        # Plot ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Fold {fold + 1} - Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()

        # Precision, Recall, F1 Score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        print(f'Fold {fold + 1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

        # Plot Confusion Matrix
        plot_confusion_matrix(y_true, y_pred)
