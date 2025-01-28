import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 256)
        self.batch_norm_1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm_2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, self.output_size)
        self.batch_norm_3 = nn.BatchNorm1d(self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.batch_norm_3(x)

        return F.softmax(x)


class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, 128)
        self.batch_norm_1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.batch_norm_2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, self.output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x


class Autoencoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = Encoder(self.input_size, self.hidden_size)
        self.decoder = Decoder(self.hidden_size, self.input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


if __name__ == "__main__":
    state_size = 1028
    num_tensors = 10000
    representation_size = 64
    autoencoder = Autoencoder(state_size, representation_size)
    random_state = torch.randn((num_tensors, state_size))

    train_epochs = 1000
    optimizer = Adam(params=autoencoder.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for epoch in range(train_epochs):
        optimizer.zero_grad()
        outputs = autoencoder(random_state)
        loss = loss_fn(outputs, random_state)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{train_epochs}], Loss: {loss.item():.4f}")

    print(autoencoder(random_state).shape)
