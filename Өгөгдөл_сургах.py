import torchvision
import torch
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import matplotlib.pyplot as plt
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from sklearn.metrics import accuracy_score

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(),])
dataset=torchvision.datasets.ImageFolder("/content/data", transform=transform)
trainloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)


model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=9)

class PrintLogEvery10Epochs(Callback):
    def on_epoch_end(self, net, **kwargs):
        if net.history[-1, 'epoch'] % 10 == 0:
            print(f'Epoch: {net.history[-1, "epoch"]}: | Loss - {net.history[-1, "train_loss"]:.4f}')

net = NeuralNetClassifier(
    model,
    max_epochs=120,
    criterion=nn.CrossEntropyLoss(),
    lr=0.01,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True,
    train_split=None,
    device='cuda',
    verbose=0,
    callbacks=[PrintLogEvery10Epochs()]
  
)
net.fit(dataset, y=None)
torch.save(net.module_, 'my_model_10.pth') #загвар татаж авах

