import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_single_image(model, x_test):
    x_test_copy = np.expand_dims(x_test, axis=0)
    x_test_copy = np.expand_dims(x_test_copy, axis=0)

    x_test_tensor = torch.from_numpy(x_test_copy).float()

    model.eval()
    with torch.no_grad():
        pred = model(x_test_tensor)
        _, predicted_label = torch.max(pred, 1)
        predicted_label = predicted_label.item()

        print(f"This image is likely {chr(ord('A') + predicted_label)}")
        plt.imshow(x_test, cmap='grey')
        plt.show()


def create_submission_file(model, x_test):
    indices = []
    predicted_labels = []

    model.eval()

    for index, x in enumerate(x_test):
        x_test_tensor = torch.unsqueeze(torch.from_numpy(x).float(), 0)

        with torch.no_grad():
            pred = model(x_test_tensor)
            _, predicted_label = torch.max(pred, 1)
            predicted_label = predicted_label.item()

        indices.append(index)
        predicted_labels.append(predicted_label)

    df_submission = pd.DataFrame({'index': indices, 'class': predicted_labels})
    df_submission.to_csv('submission.csv', index=False)


X_test = np.load('X_train.npy/X_train.npy')
model = torch.load('convolutional_NN.pth')
create_submission_file(model, X_test)
