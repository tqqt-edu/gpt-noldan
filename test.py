import os

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(100, 200)
        self.fc = nn.Linear(200, 1)

    def forward(self, x):
        return self.fc(self.embedding(x))
torch.random.manual_seed(42)
# Initialize model, loss function, and optimizer
input_tensor = torch.randint(100, size=(1, 5, 100)).to(torch.long)

if not os.path.exists('model.pth'):
    model = MyModel()

    # Save the model state dict
    torch.save(model.state_dict(), 'model.pth')
    # Perform prediction with the loaded model
    prediction = model(input_tensor)
    print(prediction[0, 0, 0])
else:
    # Load the model state dict
    loaded_model = MyModel()
    loaded_model.load_state_dict(torch.load('model.pth'))
    loaded_model.eval()

    # Perform prediction with the loaded model
    prediction = loaded_model(input_tensor)
    print(prediction[0, 0, 0])
