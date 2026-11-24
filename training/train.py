import os
import torch
import torch.nn as nn
import torch.optim as optim
import config
from training.evaluate import evaluate_model

def train_model(
    model, 
    train_loader, 
    test_loader, 
    epochs=config.EPOCHS, 
    lr=config.LEARNING_RATE, 
    device=config.DEVICE,
    save_path=os.path.join(config.MODELS_DIR, "best_cnn_bilstm.pt")
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_f1 = 0.0
    history = {"train_loss": [], "val_accuracy": [], "val_f1": []}

    print(f"Training on {device} (Epochs: {epochs}, Batch Size: {config.BATCH_SIZE}, LR: {lr})")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(train_loader))
        history["train_loss"].append(avg_loss)
        
        metrics = evaluate_model(model, test_loader, device=device)
        val_acc = metrics["accuracy"]
        val_f1 = metrics["f1_score"]
        
        history["val_accuracy"].append(val_acc)
        history["val_f1"].append(val_f1)
        
        scheduler.step(val_f1)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Accuracy: {val_acc*100:.2f}% | F1: {val_f1*100:.2f}%")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")
            
    return history
