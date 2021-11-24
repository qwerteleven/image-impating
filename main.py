import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from torchvision.utils import save_image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
          
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, stride=2, padding=1),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            torch.nn.BatchNorm2d(8), 
            torch.nn.ReLU(True),
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=1),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16), 
            torch.nn.ReLU(True),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=0),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            torch.nn.ReLU(True),
        )

        ### Flatten layer
        self.flatten = torch.nn.Flatten(start_dim=1)

        ### Linear section
        self.encoder_lin = torch.nn.Sequential(
            torch.nn.Linear(2048, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 128),

            torch.nn.Linear(128, 256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, 2048),
            torch.nn.ReLU(True)
        )
        
        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(32, 8, 8))

          
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(8, 3, 19, stride=2, padding=1, output_padding=1),
        )
  
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)

        return x

def init_model():

    # Model Initialization
    model = AE()
    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()
    
    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = 1e-3,
                                weight_decay = 1e-8)
    
    return optimizer, model, loss_function


def train_model(arrg, loader):

    epochs = 10000
    outputs = []
    losses = []
    optimizer, model, loss_function = arrg
    model.to(device)


    for epoch in range(epochs):
        losses = []
        for (image, labels) in loader:

            reconstructed = model(image)
                
            # Calculating the loss function
            loss = loss_function(reconstructed, labels)
                
            # The gradients are set to zero,
            # the the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            # Storing the losses in a list for plotting
            losses.append(loss.detach())
        
        print("loss: ", sum(losses) / len(losses))

        # outputs.append((epochs, image, reconstructed.cpu()))

        if (epoch % 200 == 0):

            save_image(image.detach(), ".\\results\\GT" + str(epoch) + "_" + str(len(losses)) + ".jpg")
            save_image(reconstructed.detach(), ".\\results\\predict" + str(epoch) + "_" + str(len(losses)) + ".jpg")

            #cv2.imwrite( ".\\results\\GT" + str(epoch) + ".jpg", A)
            #cv2.imwrite( ".\\results\\predict" + str(epoch) + ".jpg", B)

            torch.save(model, ".\\weights\\weights_" + str(epoch) + ".pth")

        print("epoch: ", epoch)


class NumbersDataset(Dataset):
    def __init__(self, tensor_transform, test_transform, device):
        self.samples = datasets.ImageFolder(root='./dataset/train',  transform=tensor_transform)
        self.labels = datasets.ImageFolder(root='./dataset/train', transform=test_transform)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples.__getitem__(idx)[0].to(device), self.labels.__getitem__(idx)[0].to(device)



def main():

    
    tensor_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=1),
    ])

    test_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Download the MNIST Dataset

    dataset = NumbersDataset(tensor_transform, test_transform, device)
    

    loader = torch.utils.data.DataLoader(dataset = dataset,
                                        batch_size = 64,
                                        shuffle = True)


    train_model(init_model(), loader)
    print('Done')


if __name__ == '__main__':
    main()