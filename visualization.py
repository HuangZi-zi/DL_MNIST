import torch
from torchvision import transforms
from PIL import Image, ImageOps
import torch.nn as nn
import matplotlib.pyplot as plt
import tkinter as tk


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()  # Create an instance of your model architecture
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Set the model to evaluation mode

# Define your 10 images
image_paths = [
    "./data/MyNumbers/0.png",
    "./data/MyNumbers/1.png",
    "./data/MyNumbers/2.png",
    "./data/MyNumbers/3.png",
    "./data/MyNumbers/4.png",
    "./data/MyNumbers/5.png",
    "./data/MyNumbers/6.png",
    "./data/MyNumbers/7.png",
    "./data/MyNumbers/8.png",
    "./data/MyNumbers/9.png"
]

# Preprocess and load images
images = []
for path in image_paths:
    image = Image.open(path).convert('L')  # Convert to grayscale
    # Invert the colors
    inverted_image = ImageOps.invert(image)
    images.append(inverted_image)

# Recognize the digits and store the results in an array
recognized_numbers = []
for image in images:
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_image)

    predicted_class = torch.argmax(output, dim=1).item()
    recognized_numbers.append(predicted_class)

fig = plt.figure()
for i in range(10):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(images[i], cmap='gray', interpolation='none')
    plt.title("Predicted: {}".format(recognized_numbers[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()
