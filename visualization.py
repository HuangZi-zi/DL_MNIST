import matplotlib.pyplot as plt

model = Net()  # Create an instance of your model architecture
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Set the model to evaluation mode