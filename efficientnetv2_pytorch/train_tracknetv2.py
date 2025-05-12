import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader
from efficientnetv2_pytorch.tracknetv2 import TrackNetV2
from efficientnetv2_pytorch.tracknetv2_dataset import TrackNetV2Dataset

def collate_fn(batch):
    # 只取 inputs, outputs
    inputs = [item[0] for item in batch]
    outputs = [item[1] for item in batch]
    inputs = torch.tensor(inputs, dtype=torch.float32)
    outputs = torch.tensor(outputs, dtype=torch.float32)
    # 不需要再 /255.0，dataset 已經做了
    return inputs, outputs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_set = TrackNetV2Dataset(csv_path='train.csv')  # 根據你的csv路徑調整
    val_set = TrackNetV2Dataset(csv_path='val.csv')
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = TrackNetV2().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()

    best_val_loss = float('inf')
    for epoch in range(10):
        # 訓練
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        # 驗證
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                # 簡單二值化準確率
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()
        avg_val_loss = val_loss / len(val_loader.dataset)
        acc = correct / total
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val Acc: {acc:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_tracknetv2.pth")
            print("Best model saved.")

if __name__ == '__main__':
    main()