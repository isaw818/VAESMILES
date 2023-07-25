import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from rdkit import Chem
from rdkit.Chem import Draw

# Set the device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset class for molecular data
class MolDataset(Dataset):
    def __init__(self, data):
        self.data = torch.as_tensor(data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index].float()

# Function to preprocess SMILES strings from a file
def preprocess_smiles(smi_file_path, maxlen=6):
    # Helper function to read SMILES strings from a file
    def filepath(file_path):
        with open(file_path) as file:
            lines = file.readlines()
        return [line.split()[0] for line in lines]
    
    # Read SMILES strings from the file
    mol = filepath(smi_file_path)
    smilelist = []
    for element in mol:
        # Pad each SMILES string with 'E' and '!' to make them fixed length
        element = "!" + element
        if len(element) <= maxlen:
            c = maxlen - len(element)
            for i in range(c):
                element = element + 'E'
            smilelist.append(element)
    return smilelist

# Function to create a one-hot encoder for the unique characters in SMILES strings
def create_one_hot_encoder(smilelist):
    # Collect unique characters from SMILES strings
    dictionary = set()
    for element in smilelist:
        for a in range(0, len(element)):
            b = element[a]
            dictionary.add(b)
    dictionary = list(dictionary)
    dictionary_in_list = [list(element) for element in dictionary]

    # Create and fit the one-hot encoder
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(dictionary_in_list)
    encoded_data_array = encoded_data.toarray()
    token = dict(zip(dictionary, encoded_data_array))
    
    return token, encoder, dictionary

# Function to generate a SMILES string using the trained VAE model
def generate_string(token, encoder, ltoken, VAE, maxlen, latent_space):
    b = torch.randn(1, latent_space).to(device)
    a = VAE.decoder(b)
    index = torch.argmax(a, 2).cpu().numpy()[0]
    list1 = []
    for i in range(maxlen):
        character = [0 for j in range(ltoken)]
        character[index[i]] = 1
        character = np.asarray(character).reshape(1, -1)
        character = encoder.inverse_transform(character).reshape(-1)
        list1.append(character[0])
    string1 = ''.join(list1[1:])
    string1 = string1.replace('E', '')
    return string1

# Function to create an RDKit molecule from a SMILES string and save it as an image
def create_mol(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        img = Draw.MolToImage(mol)
        img.save("mol/" + smile + ".png")
        return True
    return False

# VAE loss function
def vae_loss(x_decoded_mean, x, z_mean, z_logvar):
    xent_loss = F.binary_cross_entropy(x_decoded_mean, x, size_average=False)
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    return xent_loss + kl_loss

# Model Class: Molecule Variational Autoencoder (VAE)
class MoleculeVAE(nn.Module):
    def __init__(self, ltoken, maxlen, latent_space):
        super().__init__()
        self.ltoken = ltoken
        self.maxlen = maxlen
        self.latent_space = latent_space
        
        # Encoder layers
        self.dense1 = nn.Linear(ltoken * maxlen, 64)
        self.aver = nn.Linear(64, latent_space)
        self.var = nn.Linear(64, latent_space)
        
        # Decoder layers
        self.dense = nn.Linear(latent_space, 64)
        self.dense2 = nn.Linear(64, ltoken * maxlen)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=2)
        
    def encoder(self, x):
        x = x.view(-1, self.ltoken * self.maxlen)
        x = self.relu(self.dense1(x))
        mu = self.aver(x)
        sig = self.var(x)
        return mu, sig
    
    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean
    
    def decoder(self, x):
        x = self.relu(self.dense(x))
        x = self.dense2(x)
        x = x.view(-1, self.maxlen, self.ltoken)
        return self.softmax(x)
        
    def forward(self, x):
        mu, var = self.encoder(x)
        z = self.sampling(mu, var)
        newx = self.decoder(z)
        return newx, mu, var

# Function to train the VAE model
def train(model, dataloader, optimizer, latent_space):
    model.train()
    train_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        prediction, mu, var = model(batch)
        loss = vae_loss(prediction, batch, mu, var)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss

def main():
    # File path of the SMILES data
    smi_file_path = 'mole.smi'

    # Define VAE model architecture and other configurations
    maxlen = 10
    latent_space = 200
    
    # Preprocess the SMILES strings and pad them
    smilelist = preprocess_smiles(smi_file_path, maxlen)
    text = smilelist

    # Create a one-hot encoder for unique characters in SMILES strings
    token, encoder, dictionary = create_one_hot_encoder(text)
    ltoken = len(token)

    converttext = []
    for element in text:
        element_of_token = [token[elements] for elements in element]
        converttext.append(element_of_token)
    converttext = np.asarray(converttext)

    # Create a custom Dataset and DataLoader for the encoded data
    dataset = MolDataset(converttext)
    dataset_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    VAE = MoleculeVAE(ltoken, maxlen, latent_space).float().to(device)
    optimizer = optim.Adam(VAE.parameters())

    # Train the VAE for a number of epochs
    for i in range(1000):
        train_loss = train(VAE, dataset_loader, optimizer, latent_space)
        print(f"Epoch {i}, Training Loss: {train_loss:.4f}")

    # Save the trained VAE model
    torch.save(VAE, "VAE25")

    # Generate molecules using the trained VAE model
    for l in range(10):
        while True:
            string1 = generate_string(token, encoder, ltoken, VAE, maxlen, latent_space)
            t = create_mol(string1)
            if t:
                break

if __name__ == "__main__":
    main()

