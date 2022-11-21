import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import MaxPool2d

from dataload import MusicDistortionPair
from show_array import show_array
import matplotlib.pyplot as plt


device = "cpu"

dataset = MusicDistortionPair('music_data/*.npy', 'dmusic_data/*.npy')

batch_size = 5

# Create data loaders.

training_data, test_data = torch.utils.data.random_split(dataset, [dataset.__len__() - 40, 40])
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        # initialize first set of CONV => RELU => POOL layers
        self.net = nn.Sequential(
        nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3)), 
        nn.ReLU(),
        nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3,3), padding = 2), 
        nn.ReLU(),
        nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = (3,3), padding = 1), 
        )
        # self.net = nn.Sequential(
        # nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3,3)), 
        # nn.ReLU(),
        # # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # )

    def forward(self, x):
        
        # for layer in self.net:
        #     x = layer(x)
        #     print(x.size())
        
        logits = self.net(x)
        return logits
       


model = NeuralNetwork().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)



# x = torch.randn(1,128, 128)
# print("shape")
# print(x.shape) 
# output = model(x)
# INPUT_PATH = 'music_data/*.npy'
# OUTPUT_PATH = 'dmusic_data/*.npy'



def train(dataloader, model, loss_fn, optimizer):
    
    size = len(dataloader)
    model.train()
    
    loss_list = []
    for batch, (X, y) in enumerate(dataloader):
        
       
        X, y = X.to(device), y.to(device)
        
        
        # Compute prediction error
        pred = model(X)
        
        
        loss = loss_fn(pred, y)
        

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        MAX_NORM = 30.0
        torch.nn.utils.clip_grad_norm(model.parameters(), MAX_NORM)
        
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")    
            loss_list.append(loss)

    # plt.plot(loss_list)
    # plt.ylabel('loss')
    # plt.show()



train(dataset, model, loss_fn, optimizer)
torch.save(model.state_dict(), "trained/trained.pt")



# def test(dataset, model, loss_fn):
#     size = len(dataset)
#     num_batches = len(dataset)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataset:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             loss = loss_fn(pred, y)
#             print(f"{loss}")
            
            
    
    
  
# test(dataset, model, loss_fn)

# print(output)
# print(model)