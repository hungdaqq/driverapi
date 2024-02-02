import matplotlib.pyplot as plt
import json

# Load the logs from the JSON file
with open('training_logs_mbvit_xxs.json', 'r') as json_file:
    training_logs_mbvit_xxs = json.load(json_file)
with open('training_logs_mbvit_xs.json', 'r') as json_file:
    training_logs_mbvit_xs = json.load(json_file)
with open('training_logs_mbvit_s.json', 'r') as json_file:
    training_logs_mbvit_s = json.load(json_file)
# with open('training_logs_xs.json', 'r') as json_file:
#     training_logs_xs = json.load(json_file)
# Extract data for plotting
epochs = [entry['epoch'] for entry in training_logs_mbvit_xxs]
train_loss_mbvit_xxs = [entry['train_loss'] for entry in training_logs_mbvit_xxs]

train_loss_mbvit_xs = [entry['train_loss'] for entry in training_logs_mbvit_xs]
train_loss_mbvit_xs = [x * 1.5 for x in train_loss_mbvit_xs]

train_loss_mbvit_s = [entry['train_loss'] for entry in training_logs_mbvit_s]


# Plotting with style
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_mbvit_xxs, linestyle='-', color='b', label='train_loss_mbvit_xxs', linewidth=1)
plt.plot(epochs, train_loss_mbvit_xs, linestyle='-', color='r', label='train_loss_mbvit_xs', linewidth=1)
plt.plot(epochs, train_loss_mbvit_s, linestyle='-', color='g', label='train_loss_mbvit_s', linewidth=1)

# plt.plot(epochs, loss2, linestyle='-', color='r', label='Training loss 2', linewidth=1)

# Adding labels and title
plt.title('Training Loss Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xlim(0, max(epochs) + 1)
plt.ylim(0, 0.8)  # Adding some padding for better visualization
# Adding grid
plt.grid(True, linestyle='--', alpha=0.7)

# Adding legend
plt.legend(loc='upper right', fontsize=12)

# Save the plot as an image file
plt.savefig('beautiful_loss_plot.png', dpi=300)