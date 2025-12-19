import json
import matplotlib.pyplot as plt

with open(r"efficient_darts_results\efficient_darts_results.json", "r") as f:
    data = json.load(f)

history = data["history"]

epochs = range(1, len(history["train_loss"]) + 1)

plt.figure()
plt.plot(epochs, history["train_loss"], label="Train Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.title("Training and Validation Loss")
plt.show()

plt.figure()
plt.plot(epochs, history["train_acc"], label="Train Accuracy")
plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.title("Training and Validation Accuracy")
plt.show()


flops = [e["flops"] for e in history["efficiency"]]
macs = [e["macs"] for e in history["efficiency"]]

plt.figure()
plt.plot(epochs, flops, label="FLOPs")
plt.plot(epochs, macs, label="MACs")
plt.xlabel("Epoch")
plt.ylabel("Compute Cost")
plt.legend()
plt.grid(True)
plt.title("Efficiency Evolution During Search")
plt.show()


plt.figure()
plt.scatter(flops, history["val_acc"])
plt.xlabel("FLOPs")
plt.ylabel("Validation Accuracy (%)")
plt.grid(True)
plt.title("Accuracy vs Computational Cost")
plt.show()