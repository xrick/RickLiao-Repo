import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 自定義數據集
class SoundDataset(Dataset):
    def __init__(self, X, y_rain, y_traffic):
        self.X = torch.FloatTensor(X)
        self.y_rain = torch.LongTensor(y_rain)
        self.y_traffic = torch.LongTensor(y_traffic)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_rain[idx], self.y_traffic[idx]

# 定義模型
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Flatten()
        )
        
        self.rain_classifier = nn.Linear(64 * 16 * 16, 2)
        self.traffic_classifier = nn.Linear(64 * 16 * 16, 2)

    def forward(self, x):
        shared_features = self.shared_layers(x)
        rain_output = self.rain_classifier(shared_features)
        traffic_output = self.traffic_classifier(shared_features)
        return rain_output, traffic_output

# 訓練函數
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for X, y_rain, y_traffic in train_loader:
        X, y_rain, y_traffic = X.to(device), y_rain.to(device), y_traffic.to(device)
        
        optimizer.zero_grad()
        rain_output, traffic_output = model(X)
        loss_rain = criterion(rain_output, y_rain)
        loss_traffic = criterion(traffic_output, y_traffic)
        loss = loss_rain + loss_traffic
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 評估函數
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_rain = 0
    correct_traffic = 0
    with torch.no_grad():
        for X, y_rain, y_traffic in test_loader:
            X, y_rain, y_traffic = X.to(device), y_rain.to(device), y_traffic.to(device)
            rain_output, traffic_output = model(X)
            
            loss_rain = criterion(rain_output, y_rain)
            loss_traffic = criterion(traffic_output, y_traffic)
            total_loss += (loss_rain + loss_traffic).item()
            
            _, predicted_rain = torch.max(rain_output, 1)
            _, predicted_traffic = torch.max(traffic_output, 1)
            correct_rain += (predicted_rain == y_rain).sum().item()
            correct_traffic += (predicted_traffic == y_traffic).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    rain_acc = correct_rain / len(test_loader.dataset)
    traffic_acc = correct_traffic / len(test_loader.dataset)
    return avg_loss, rain_acc, traffic_acc

# 主程序
def main():
    # 設置設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成模擬數據
    X_train = np.random.rand(1000, 1, 64, 64).astype(np.float32)
    y_train_rain = np.random.randint(0, 2, (1000,))
    y_train_traffic = np.random.randint(0, 2, (1000,))
    
    X_test = np.random.rand(200, 1, 64, 64).astype(np.float32)
    y_test_rain = np.random.randint(0, 2, (200,))
    y_test_traffic = np.random.randint(0, 2, (200,))

    # 創建數據集和數據加載器
    train_dataset = SoundDataset(X_train, y_train_rain, y_train_traffic)
    test_dataset = SoundDataset(X_test, y_test_rain, y_test_traffic)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 初始化模型、損失函數和優化器
    model = MultiTaskModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練模型
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, rain_acc, traffic_acc = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        print(f'Rain Accuracy: {rain_acc:.4f}, Traffic Accuracy: {traffic_acc:.4f}')

if __name__ == "__main__":
    main()
