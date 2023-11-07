import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from CustomDataset import CustomDataset
from ConvolutionalNetwork import CNN
from ConvolutionalNetwork import back_pass, test
import numpy as np


X_train = np.load('X_train.npy/X_train.npy')
y_train = np.load('y_train.npy/y_train.npy')
X_valid = np.load('X_val.npy/X_val.npy')
y_valid = np.load('y_val.npy')

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_valid: {X_valid.shape}")
print(f"Shape of y_valid: {y_valid.shape}")

amounts_in_category = np.sum(y_train, axis=0)
print("Amount of data in each category:")
print(amounts_in_category)

y_train = np.argmax(y_train, axis=1)
y_valid = np.argmax(y_valid, axis=1)
y_train = torch.from_numpy(y_train)
y_valid = torch.from_numpy(y_valid)

X_train = torch.unsqueeze(torch.from_numpy(X_train), 1).float()
X_valid = torch.unsqueeze(torch.from_numpy(X_valid), 1).float()

class_sample_counts = np.array([amounts_in_category[i] for i in range(len(amounts_in_category))])
weights = 1. / class_sample_counts
samples_weights = np.array([weights[t] for t in y_train.numpy()])

batch_size = 32
epochs = 65
learning_rate = 0.0001

samples_weights = torch.from_numpy(samples_weights)
samples_weights = samples_weights.double()

sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weights, len(samples_weights))

train_loader = DataLoader(dataset=CustomDataset(X_train, y_train, True), batch_size=batch_size, sampler=sampler)
test_loader = DataLoader(dataset=CustomDataset(X_valid, y_valid, True), batch_size=batch_size, shuffle=True)

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_accu, test_accu, train_loss, test_loss = [], [], [], []

for i in range(epochs):
    print(f"Epoch: {i + 1}")
    back_pass(model, train_loader, loss_fn, optimizer)
    test_accuracy, test_l = test(model, test_loader, loss_fn)
    train_accuracy, train_l = test(model, train_loader, loss_fn)
    print(f"Accuracy on train data: {train_accuracy}%")
    print(f"Accuracy on test data: {test_accuracy}%")
    train_accu.append(train_accuracy)
    test_accu.append(test_accuracy)
    test_loss.append(test_l)
    train_loss.append(train_l)

test_accuracy, test_l, y_pred_test, y_real_test = test(model, test_loader, loss_fn, True)
train_accuracy, train_l, y_pred_train, y_real_train = test(model, train_loader, loss_fn, True)

### commented out after finished training

# torch.save(model, 'convolutional_NN.pth')
#
# np.save('test_accuracy.npy', test_accuracy)
# np.save('test_loss.npy', test_l)
# np.save('y_pred_test.npy', y_pred_test)
# np.save('y_real_test.npy', y_real_test)
# np.save('train_accuracy.npy', train_accuracy)
# np.save('train_loss.npy', train_l)
# np.save('y_pred_train.npy', y_pred_train)
# np.save('y_real_train.npy', y_real_train)
# torch.save(model.state_dict(), 'convolutional_NN_dict')