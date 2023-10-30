import matplotlib.pyplot as plt
import json

# Load the logs from the JSON file
with open('training_logs.json', 'r') as json_file:
    training_logs = json.load(json_file)
# with open('training_logs_xs.json', 'r') as json_file:
#     training_logs_xs = json.load(json_file)
# Extract data for plotting
epochs = [entry['epoch'] for entry in training_logs]
train_loss = [entry['train_loss'] for entry in training_logs]
val_loss = [entry['val_loss'] for entry in training_logs]
train_accuracy = [entry['train_accuracy'] for entry in training_logs]
val_accuracy = [entry['val_accuracy'] for entry in training_logs]
learning_rate = [entry['learning_rate'] for entry in training_logs]


# Plotting with style
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, linestyle='-', color='b', label='train_loss', linewidth=1)
plt.plot(epochs, val_loss, linestyle='-', color='r', label='val_loss', linewidth=1)

# plt.plot(epochs, loss2, linestyle='-', color='r', label='Training loss 2', linewidth=1)

# Adding labels and title
plt.title('Training Loss Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xlim(0, max(epochs) + 1)
plt.ylim(0, 0.3)  # Adding some padding for better visualization
# Adding grid
plt.grid(True, linestyle='--', alpha=0.7)

# Adding legend
plt.legend(loc='upper right', fontsize=12)

# Save the plot as an image file
plt.savefig('beautiful_loss_plot.png', dpi=300)

