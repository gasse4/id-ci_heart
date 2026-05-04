import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# df = pd.read_csv("heart.csv")
df = pd.read_csv("WineQT.csv", sep=",")

# X = df.drop("target", axis=1)
# y = df["target"]
X = df.drop("quality", axis=1)
df = df.drop("Id", axis=1)
y = (df["quality"] > 5).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

#  Scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# PyTorch ANN


X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

my_columns = X.shape[1]
print  ("Number of features:", my_columns)

class ANN(nn.Module):
    def __init__(self , input_dim = my_columns):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = ANN()

# loss + optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01 )

# training
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

# evaluation
with torch.no_grad():
    preds = model(X_test_t)
    preds = (preds > 0.6).float()
    accuracy = (preds.eq(y_test_t).sum() / len(y_test_t)).item()

print("Accuracy:", accuracy)

sample = X_test_t[0].unsqueeze(0)

with torch.no_grad():
    pred = model(sample)
    pred = (pred > 0.5).float()
for i in range(10):
    sample = X_test_t[i].unsqueeze(0)
    with torch.no_grad():
        pred = model(sample)
        pred = (pred > 0.5).float()

    print(f"Sample {i} -> Pred: {int(pred.item())}, Actual: {int(y_test_t[i].item())}")