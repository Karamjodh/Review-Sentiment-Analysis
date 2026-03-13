import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_imdb_data():
    """Load IMDB dataset from HuggingFace"""
    print("Loading IMDB dataset...")
    dataset = load_dataset('imdb')
    
    train_texts = dataset['train']['text']
    train_labels = dataset['train']['label']
    test_texts = dataset['test']['text']
    test_labels = dataset['test']['label']
    
    # Use a subset for faster training (remove these lines for full dataset)
    train_texts = train_texts[:5000]
    train_labels = train_labels[:5000]
    test_texts = test_texts[:1000]
    test_labels = test_labels[:1000]
    
    return train_texts, train_labels, test_texts, test_labels

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy, predictions, true_labels

def test_negation_cases(model, tokenizer, device):
    """Test model on tricky negation cases"""
    test_cases = [
        ("This movie is not good", 0),  # Negative
        ("This movie is not bad", 1),   # Positive
        ("I did not like this film", 0), # Negative
        ("Not the worst movie ever", 1), # Positive
        ("This is great", 1),            # Positive
        ("This is terrible", 0),         # Negative
        ("Never seen anything better", 1), # Positive
        ("Could not be more disappointed", 0), # Negative
    ]
    
    print("\n" + "="*60)
    print("Testing on Negation Cases:")
    print("="*60)
    
    model.eval()
    correct = 0
    
    for text, true_label in test_cases:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()
        
        is_correct = pred == true_label
        correct += is_correct
        
        sentiment = "Positive" if pred == 1 else "Negative"
        true_sentiment = "Positive" if true_label == 1 else "Negative"
        status = "✓" if is_correct else "✗"
        
        print(f"{status} Text: '{text}'")
        print(f"  Predicted: {sentiment} | True: {true_sentiment}\n")
    
    accuracy = correct / len(test_cases) * 100
    print(f"Negation Test Accuracy: {accuracy:.1f}%")
    print("="*60)

def main():
    # Load data
    train_texts, train_labels, test_texts, test_labels = load_imdb_data()
    
    # Initialize tokenizer and model
    print("\nInitializing DistilBERT model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        
        test_loss, test_acc, _, _ = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"✓ New best model saved! Accuracy: {best_accuracy:.4f}")
    
    # Load best model and test on negation cases
    print("\nLoading best model for negation testing...")
    model.load_state_dict(torch.load('best_model.pt'))
    test_negation_cases(model, tokenizer, device)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best Test Accuracy: {best_accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()