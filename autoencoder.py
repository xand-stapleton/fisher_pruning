# Simple autoencoder (deliberately degenerate) to demonstrate FIM pruning
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm.autonotebook import tqdm

# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=latent_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=input_size),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class MNISTModel():
    def __init__(self, latent_dim=256, batch_size=1000):
        # Set a Douglas Adams random seed for reproducibility
        torch.manual_seed(42)

        # Hyperparameters
        self.input_size = 784  # Size of the input data (e.g., MNIST images flattened)
        self.latent_size = latent_dim

        # Initialise training variables
        self.lr = None
        self.batch_size = batch_size
        self.num_epochs = None

        # Load MNIST dataset

        # Define a transform to convert image to tensor
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, transform=transform, download=True
        )

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=batch_size, shuffle=True
        )

        # Instantiate the autoencoder
        self.autoencoder = None
        return None


    def train(self, lr=0.001, num_epochs=10):
        self.lr = lr
        self.num_epochs = num_epochs


        # Initialize the autoencoder
        self.autoencoder = Autoencoder(input_size=self.input_size,
                                       latent_size=self.latent_size)

        # Loss function and optimiser
        loss_fn = nn.MSELoss()
        optimiser = optim.Adam(self.autoencoder.parameters(), lr=lr)

        # Training loop
        for epoch in tqdm(range(num_epochs), desc='Epochs: ',
                          leave=True, position=1):

            for data in tqdm(self.train_loader, desc='Data: '):
                inputs, _ = data
                inputs = inputs.view(-1, self.input_size)
                
                # Zero the gradients
                optimiser.zero_grad()
                
                # Forward pass
                outputs = self.autoencoder(inputs)
                
                # Compute the loss
                loss = loss_fn(outputs, inputs)
                
                # Backpropagation
                loss.backward()
                
                # Update the weights
                optimiser.step()
            
            # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        print()
        print('Completed training successfully...')


    # Save the trained model
    # torch.save(autoencoder.state_dict(), 'autoencoder.pth')

if __name__=='__main__':
    model = MNISTModel()
    model.train()
