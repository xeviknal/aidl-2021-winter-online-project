import torch

from dataset import MyDataset
from model import MyLeNet
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from utils import accuracy, save_model

device = torch.device("cuda") if False else torch.device("cpu")


def correct_predictions(predicted_batch, label_batch):
    pred = predicted_batch.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    acum = pred.eq(label_batch.view_as(pred)).sum().item()
    return acum


def train_single_epoch(epoch, train_loader, model):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.NLLLoss()
    avg_loss = None
    avg_weight = 0.1
    for batch_idx, (img, target) in enumerate(train_loader):
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, target)
        loss.backward()
        if avg_loss:
            avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
        else:
            avg_loss = loss.item()
        optimizer.step()
        if batch_idx % config['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return avg_loss


def eval_single_epoch(epoch, val_loader, model):
    model.eval()
    criterion = nn.NLLLoss(reduction='sum')
    test_loss = 0
    acc = 0
    with torch.no_grad():
        for img, target in val_loader:
            img, target = img.to(device), target.to(device)
            output = model(img)
            test_loss += criterion(output, target).item()
            acc += correct_predictions(output, target)
    test_loss = test_loss / len(val_loader.dataset)
    test_acc = (acc / len(val_loader.dataset)) * 100.
    print('\nTest set: {} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, acc, len(val_loader.dataset), test_acc,
    ))


def train_model(train_loader, val_loader):
    my_model = MyLeNet(
        config['model_conv1_feat_size'], config['model_conv2_feat_size'], config['model_conv_kernel'],
        config['model_conv_pooling'],
        config['model_fc_hidden1'], config['model_fc_hidden2'], config['model_num_classes']
    ).to(device)
    for epoch in range(config["num_epochs"]):
        train_single_epoch(epoch, train_loader, my_model)
        eval_single_epoch(epoch, val_loader, my_model)

    return my_model


def test_model(something):
    pass


if __name__ == "__main__":
    config = {
        ## MODEL PARAMS
        'model_conv1_feat_size': 6,
        'model_conv2_feat_size': 16,
        'model_conv_kernel': 5,
        'model_conv_pooling': 2,
        'model_fc_hidden1': 120,
        'model_fc_hidden2': 84,
        'model_num_classes': 15,

        ## TRAINING PARAMS
        'batch_size': 64,
        'num_epochs': 10,
        'test_batch_size': 64,
        'learning_rate': 1e-3,
        'log_interval': 25,
    }

    my_dataset = MyDataset('./data/data', './data/chinese_mnist.csv', transform=transforms.ToTensor())

    # Splitting the data sets for the whole experiment: 70-30 (train - test), 70-30 (train - val)
    dataset_size = len(my_dataset)
    train_size = int(np.floor(dataset_size * 0.7))
    train_set, test_set = torch.utils.data.random_split(my_dataset, [train_size, dataset_size - train_size])

    dataset_size = train_size
    train_size = int(np.floor(dataset_size * .7))
    train_set, val_set = torch.utils.data.random_split(train_set, [train_size, dataset_size - train_size])

    tr_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True)

    v_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config['batch_size'],
        shuffle=True)

    tt_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=True)

    train_model(tr_loader, v_loader)

    # print(test_model(...))
