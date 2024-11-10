import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a neural network model better suited for MNIST
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load MNIST dataset and split among clients
num_clients = 5
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# Split dataset among clients
data_per_client = len(mnist_train) // num_clients
client_data = [
    data.DataLoader(data.Subset(mnist_train, range(i * data_per_client, (i + 1) * data_per_client)), batch_size=64, shuffle=True)
    for i in range(num_clients)
]

# Federated learning parameters
global_model = Net()
epochs = 3
secure_aggregation_key = 42  # A simple key to simulate secure aggregation

# Function to securely aggregate models
# Here we use simple random noise as a placeholder for secure aggregation
# (Note: This is just a conceptual illustration)
def secure_aggregate(models, key):
    aggregated_model = models[0]
    # noise = random.Random(key).uniform(-0.01, 0.01)
    noise = 0
    for name, param in aggregated_model.state_dict().items():
        for other_model in models[1:]:
            param += other_model.state_dict()[name]
        param /= len(models)
        # Add noise for "secure" aggregation    
        param += noise
    return aggregated_model

# Training loop
for epoch in range(epochs):
    local_models = []

    # Clients train on their local data
    for client_idx, client_loader in enumerate(client_data):
        local_model = Net()
        local_model.load_state_dict(global_model.state_dict())

        optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        # Train on client data
        local_model.train()
        for x_train, y_train in client_loader:
            for _ in range(1):  # 1 local epoch per client
                y_pred = local_model(x_train)
                loss = loss_fn(y_pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        local_models.append(local_model)

    # Secure aggregation of local models
    global_model = secure_aggregate(local_models, secure_aggregation_key)
    print(f"Epoch {epoch+1}: Aggregation Complete")

# Save the global model
torch.save(global_model.state_dict(), "global_model.pth")

# Comprehensive Evaluation
test_loader = data.DataLoader(datasets.MNIST(root="./data", train=False, download=True, transform=transform), batch_size=1000, shuffle=False)

# Evaluate the global model on the full test set
global_model.load_state_dict(torch.load("global_model.pth"))  # Load the saved model
global_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        y_pred = global_model(x_test)
        _, predicted = torch.max(y_pred.data, 1)
        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()

accuracy = 100 * correct / total
print(f"Global Model Accuracy: {accuracy:.2f}%")

# Visualizer to demonstrate model capabilities
num_samples = 10
x_samples, _ = next(iter(test_loader))
x_samples = x_samples[:num_samples]

# Generate model predictions
with torch.no_grad():
    predictions = global_model(x_samples.to(device))
    _, predicted_labels = torch.max(predictions, 1)

# Initial setup for visualization
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

# Display the first sample initially
original_img = x_samples[0].squeeze().cpu().numpy()
predicted_label = predicted_labels[0].item()

original_display = ax[0].imshow(original_img, cmap='gray')
ax[0].set_title(f'Original Image')
label_display = ax[1].text(0.5, 0.5, f'Predicted Label: {predicted_label}', fontsize=15, ha='center')
ax[1].set_axis_off()

# Adding a slider for sample index
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_slider, 'Sample', 0, num_samples - 1, valinit=0, valstep=1)

# Update function for the slider
def update(val):
    idx = int(slider.val)
    original_img = x_samples[idx].squeeze().cpu().numpy()
    predicted_label = predicted_labels[idx].item()
    original_display.set_data(original_img)
    label_display.set_text(f'Predicted Label: {predicted_label}')
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()
