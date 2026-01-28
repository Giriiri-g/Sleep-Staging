import re
import matplotlib.pyplot as plt
from collections import OrderedDict
import math
# =========================
# 1. PASTE YOUR LOG HERE
# =========================
LOG_TEXT = r"""
Epoch [1/50] (206.3s)
  Shared  - Loss: 0.6191, Acc: 69.0566
  Val     - Loss: 0.6312, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 7.2694, Reward (AUC): 0.5456
  Best Reward (AUC): 0.5456

Epoch [2/50] (221.6s)
  Shared  - Loss: 0.6043, Acc: 71.0692
  Val     - Loss: 0.6433, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 8.6031, Reward (AUC): 0.5651
  Best Reward (AUC): 0.5651

Epoch [3/50] (137.0s)
  Shared  - Loss: 0.6187, Acc: 68.4277
  Val     - Loss: 0.6198, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 8.7506, Reward (AUC): 0.5700
  Best Reward (AUC): 0.5700

Epoch [4/50] (100.9s)
  Shared  - Loss: 0.6260, Acc: 70.1887
  Val     - Loss: 0.6173, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.5479, Reward (AUC): 0.5833
  Best Reward (AUC): 0.5833

Epoch [5/50] (98.9s)
  Shared  - Loss: 0.6136, Acc: 70.0629
  Val     - Loss: 0.6108, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.9855, Reward (AUC): 0.5923
  Best Reward (AUC): 0.5923

Epoch [6/50] (115.1s)
  Shared  - Loss: 0.6214, Acc: 69.8113
  Val     - Loss: 0.6070, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 10.6503, Reward (AUC): 0.6044
  Best Reward (AUC): 0.6044

Epoch [7/50] (136.3s)
  Shared  - Loss: 0.6172, Acc: 70.1887
  Val     - Loss: 0.5994, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.7357, Reward (AUC): 0.5968
  Best Reward (AUC): 0.6044

Epoch [8/50] (127.4s)
  Shared  - Loss: 0.6170, Acc: 70.4403
  Val     - Loss: 0.6264, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 11.0850, Reward (AUC): 0.6174
  Best Reward (AUC): 0.6174

Epoch [9/50] (122.9s)
  Shared  - Loss: 0.6092, Acc: 68.9308
  Val     - Loss: 0.5972, Acc: 0.6848, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.9726, Reward (AUC): 0.6075
  Best Reward (AUC): 0.6174

Epoch [10/50] (120.2s)
  Shared  - Loss: 0.6236, Acc: 68.9308
  Val     - Loss: 0.6075, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.4267, Reward (AUC): 0.6044
  Best Reward (AUC): 0.6174

Epoch [11/50] (119.2s)
  Shared  - Loss: 0.6239, Acc: 69.1824
  Val     - Loss: 0.6031, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 10.6068, Reward (AUC): 0.6227
  Best Reward (AUC): 0.6227

Epoch [12/50] (121.8s)
  Shared  - Loss: 0.6138, Acc: 70.6918
  Val     - Loss: 0.6065, Acc: 0.6957, F1: 0.0667
  Val     - Prec: 0.5000, Rec: 0.0357
  Controller - Loss: 10.6134, Reward (AUC): 0.6267
  Best Reward (AUC): 0.6267

Epoch [13/50] (123.8s)
  Shared  - Loss: 0.6102, Acc: 70.5660
  Val     - Loss: 0.5956, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.6788, Reward (AUC): 0.6189
  Best Reward (AUC): 0.6267

Epoch [14/50] (122.8s)
  Shared  - Loss: 0.6178, Acc: 68.6792
  Val     - Loss: 0.6000, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.8866, Reward (AUC): 0.6252
  Best Reward (AUC): 0.6267

Epoch [15/50] (125.3s)
  Shared  - Loss: 0.6174, Acc: 69.6855
  Val     - Loss: 0.5945, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 10.2477, Reward (AUC): 0.6332
  Best Reward (AUC): 0.6332

Epoch [16/50] (123.5s)
  Shared  - Loss: 0.6061, Acc: 69.8113
  Val     - Loss: 0.6074, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.7344, Reward (AUC): 0.6306
  Best Reward (AUC): 0.6332

Epoch [17/50] (121.2s)
  Shared  - Loss: 0.6057, Acc: 70.9434
  Val     - Loss: 0.6047, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 10.2950, Reward (AUC): 0.6413
  Best Reward (AUC): 0.6413

Epoch [18/50] (123.9s)
  Shared  - Loss: 0.6032, Acc: 70.0629
  Val     - Loss: 0.5906, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.8871, Reward (AUC): 0.6400
  Best Reward (AUC): 0.6413

Epoch [19/50] (124.3s)
  Shared  - Loss: 0.6262, Acc: 70.0629
  Val     - Loss: 0.6081, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.9468, Reward (AUC): 0.6443
  Best Reward (AUC): 0.6443

Epoch [20/50] (125.2s)
  Shared  - Loss: 0.6082, Acc: 69.6855
  Val     - Loss: 0.5952, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.4950, Reward (AUC): 0.6425
  Best Reward (AUC): 0.6443

Epoch [21/50] (123.5s)
  Shared  - Loss: 0.6089, Acc: 68.9308
  Val     - Loss: 0.5940, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 10.0036, Reward (AUC): 0.6524
  Best Reward (AUC): 0.6524

Epoch [22/50] (121.7s)
  Shared  - Loss: 0.6142, Acc: 68.1761
  Val     - Loss: 0.5913, Acc: 0.7065, F1: 0.1818
  Val     - Prec: 0.6000, Rec: 0.1071
  Controller - Loss: 9.7867, Reward (AUC): 0.6533
  Best Reward (AUC): 0.6533

Epoch [23/50] (123.8s)
  Shared  - Loss: 0.6188, Acc: 67.6730
  Val     - Loss: 0.6204, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.6329, Reward (AUC): 0.6550
  Best Reward (AUC): 0.6550

Epoch [24/50] (119.6s)
  Shared  - Loss: 0.6163, Acc: 68.4277
  Val     - Loss: 0.6300, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.1839, Reward (AUC): 0.6530
  Best Reward (AUC): 0.6550

Epoch [25/50] (126.1s)
  Shared  - Loss: 0.6022, Acc: 69.5597
  Val     - Loss: 0.6008, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.5272, Reward (AUC): 0.6607
  Best Reward (AUC): 0.6607

Epoch [26/50] (122.0s)
  Shared  - Loss: 0.6015, Acc: 69.4340
  Val     - Loss: 0.6550, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 9.0535, Reward (AUC): 0.6582
  Best Reward (AUC): 0.6607

Epoch [27/50] (123.6s)
  Shared  - Loss: 0.6064, Acc: 70.0629
  Val     - Loss: 0.5883, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 8.8428, Reward (AUC): 0.6590
  Best Reward (AUC): 0.6607

Epoch [28/50] (123.6s)
  Shared  - Loss: 0.6102, Acc: 69.6855
  Val     - Loss: 0.6830, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 8.8637, Reward (AUC): 0.6625
  Best Reward (AUC): 0.6625

Epoch [29/50] (124.2s)
  Shared  - Loss: 0.6068, Acc: 69.4340
  Val     - Loss: 0.5752, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 8.2775, Reward (AUC): 0.6585
  Best Reward (AUC): 0.6625

Epoch [30/50] (116.9s)
  Shared  - Loss: 0.6072, Acc: 68.0503
  Val     - Loss: 0.5742, Acc: 0.7065, F1: 0.1818
  Val     - Prec: 0.6000, Rec: 0.1071
  Controller - Loss: 8.2334, Reward (AUC): 0.6610
  Best Reward (AUC): 0.6625

Epoch [31/50] (152.6s)
  Shared  - Loss: 0.5966, Acc: 69.5597
  Val     - Loss: 0.5710, Acc: 0.7065, F1: 0.2286
  Val     - Prec: 0.5714, Rec: 0.1429
  Controller - Loss: 7.8876, Reward (AUC): 0.6597
  Best Reward (AUC): 0.6625

Epoch [32/50] (155.0s)
  Shared  - Loss: 0.5902, Acc: 69.5597
  Val     - Loss: 0.5808, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 7.4806, Reward (AUC): 0.6576
  Best Reward (AUC): 0.6625

Epoch [33/50] (101.0s)
  Shared  - Loss: 0.6145, Acc: 69.9371
  Val     - Loss: 0.6954, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 7.4970, Reward (AUC): 0.6605
  Best Reward (AUC): 0.6625

Epoch [34/50] (100.2s)
  Shared  - Loss: 0.6148, Acc: 69.4340
  Val     - Loss: 0.5781, Acc: 0.7174, F1: 0.2353
  Val     - Prec: 0.6667, Rec: 0.1429
  Controller - Loss: 7.3566, Reward (AUC): 0.6616
  Best Reward (AUC): 0.6625

Epoch [35/50] (144.4s)
  Shared  - Loss: 0.5930, Acc: 71.0692
  Val     - Loss: 0.6430, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 7.3336, Reward (AUC): 0.6640
  Best Reward (AUC): 0.6640

Epoch [36/50] (96.4s)
  Shared  - Loss: 0.5750, Acc: 71.5723
  Val     - Loss: 0.5864, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 7.0638, Reward (AUC): 0.6633
  Best Reward (AUC): 0.6640

Epoch [37/50] (99.4s)
  Shared  - Loss: 0.5947, Acc: 69.8113
  Val     - Loss: 0.6282, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 7.1893, Reward (AUC): 0.6675
  Best Reward (AUC): 0.6675

Epoch [38/50] (116.0s)
  Shared  - Loss: 0.5967, Acc: 68.9308
  Val     - Loss: 0.5752, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 7.2787, Reward (AUC): 0.6713
  Best Reward (AUC): 0.6713

Epoch [39/50] (124.8s)
  Shared  - Loss: 0.5970, Acc: 70.1887
  Val     - Loss: 0.5678, Acc: 0.7174, F1: 0.1333
  Val     - Prec: 1.0000, Rec: 0.0714
  Controller - Loss: 6.7569, Reward (AUC): 0.6675
  Best Reward (AUC): 0.6713

Epoch [40/50] (122.7s)
  Shared  - Loss: 0.5924, Acc: 71.1950
  Val     - Loss: 0.6260, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 6.6920, Reward (AUC): 0.6692
  Best Reward (AUC): 0.6713

Epoch [41/50] (121.7s)
  Shared  - Loss: 0.5934, Acc: 69.6855
  Val     - Loss: 0.5868, Acc: 0.7174, F1: 0.1875
  Val     - Prec: 0.7500, Rec: 0.1071
  Controller - Loss: 6.4672, Reward (AUC): 0.6689
  Best Reward (AUC): 0.6713

Epoch [42/50] (123.7s)
  Shared  - Loss: 0.6058, Acc: 70.5660
  Val     - Loss: 0.5615, Acc: 0.7174, F1: 0.3158
  Val     - Prec: 0.6000, Rec: 0.2143
  Controller - Loss: 6.1985, Reward (AUC): 0.6679
  Best Reward (AUC): 0.6713

Epoch [43/50] (122.7s)
  Shared  - Loss: 0.6136, Acc: 67.9245
  Val     - Loss: 0.5735, Acc: 0.7174, F1: 0.3158
  Val     - Prec: 0.6000, Rec: 0.2143
  Controller - Loss: 6.1154, Reward (AUC): 0.6691
  Best Reward (AUC): 0.6713

Epoch [44/50] (123.9s)
  Shared  - Loss: 0.5976, Acc: 69.4340
  Val     - Loss: 0.5758, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 5.9601, Reward (AUC): 0.6694
  Best Reward (AUC): 0.6713

Epoch [45/50] (125.0s)
  Shared  - Loss: 0.5700, Acc: 72.7044
  Val     - Loss: 0.5987, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 6.1001, Reward (AUC): 0.6734
  Best Reward (AUC): 0.6734

Epoch [46/50] (121.9s)
  Shared  - Loss: 0.6108, Acc: 71.1950
  Val     - Loss: 0.6318, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 5.7550, Reward (AUC): 0.6713
  Best Reward (AUC): 0.6734

Epoch [47/50] (122.2s)
  Shared  - Loss: 0.5771, Acc: 69.1824
  Val     - Loss: 0.5840, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 5.6609, Reward (AUC): 0.6723
  Best Reward (AUC): 0.6734

Epoch [48/50] (123.0s)
  Shared  - Loss: 0.5782, Acc: 70.8176
  Val     - Loss: 0.6021, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 5.3173, Reward (AUC): 0.6701
  Best Reward (AUC): 0.6734

Epoch [49/50] (123.7s)
  Shared  - Loss: 0.5988, Acc: 69.5597
  Val     - Loss: 0.5806, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 5.2463, Reward (AUC): 0.6712
  Best Reward (AUC): 0.6734

Epoch [50/50] (124.0s)
  Shared  - Loss: 0.5811, Acc: 69.6855
  Val     - Loss: 0.6026, Acc: 0.6957, F1: 0.0000
  Val     - Prec: 0.0000, Rec: 0.0000
  Controller - Loss: 5.0058, Reward (AUC): 0.6701
  Best Reward (AUC): 0.6734"""

epoch_re = re.compile(r"Epoch\s*\[(\d+)/\d+\]")

shared_re = re.compile(
    r"Shared\s*-\s*Loss:\s*([\d.]+),\s*Acc:\s*([\d.]+)"
)

val_re = re.compile(
    r"Val\s*-\s*Loss:\s*([\d.]+),\s*Acc:\s*([\d.]+),\s*F1:\s*([\d.]+)"
)

prec_re = re.compile(
    r"Val\s*-\s*Prec:\s*([\d.]+),\s*Rec:\s*([\d.]+)"
)

reward_re = re.compile(
    r"Reward\s*\(AUC\):\s*([\d.]+)"
)

best_reward_re = re.compile(
    r"Best\s+Reward\s*\(AUC\):\s*([\d.]+)"
)

# =========================
# 3. PARSE LOG
# =========================
data = OrderedDict()
current_epoch = None

for line in LOG_TEXT.splitlines():

    ep = epoch_re.search(line)
    if ep:
        current_epoch = int(ep.group(1))
        data.setdefault(current_epoch, {
            "shared_loss": None,
            "shared_acc": None,
            "val_loss": None,
            "val_acc": None,
            "val_f1": None,
            "prec": None,
            "rec": None,
            "reward": None,
            "best_reward": None
        })
        continue

    if current_epoch is None:
        continue

    if m := shared_re.search(line):
        data[current_epoch]["shared_loss"] = float(m.group(1))
        data[current_epoch]["shared_acc"] = float(m.group(2))

    if m := val_re.search(line):
        data[current_epoch]["val_loss"] = float(m.group(1))
        data[current_epoch]["val_acc"] = float(m.group(2))
        data[current_epoch]["val_f1"] = float(m.group(3))

    if m := prec_re.search(line):
        data[current_epoch]["prec"] = float(m.group(1))
        data[current_epoch]["rec"] = float(m.group(2))

    if m := reward_re.search(line):
        data[current_epoch]["reward"] = float(m.group(1))

    if m := best_reward_re.search(line):
        data[current_epoch]["best_reward"] = float(m.group(1))

# =========================
# 4. UTILITIES
# =========================
epochs = sorted(data.keys())

def collect(key):
    return [
        data[e][key] if data[e][key] is not None else math.nan
        for e in epochs
    ]

shared_loss = collect("shared_loss")
shared_acc  = collect("shared_acc")
val_loss    = collect("val_loss")
val_acc     = collect("val_acc")
val_f1      = collect("val_f1")
prec        = collect("prec")
rec         = collect("rec")
reward      = collect("reward")
best_reward = collect("best_reward")

# =========================
# 5. PLOTTING
# =========================
plt.figure(figsize=(16, 10))

plt.subplot(3, 2, 1)
plt.plot(epochs, shared_loss, marker="o")
plt.title("Shared Model Loss")
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(epochs, shared_acc, marker="o")
plt.title("Shared Model Accuracy")
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(epochs, val_loss, marker="o")
plt.title("Validation Loss")
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(epochs, val_acc, marker="o")
plt.title("Validation Accuracy")
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(epochs, val_f1, marker="o", label="F1")
plt.plot(epochs, prec, marker="o", label="Precision")
plt.plot(epochs, rec, marker="o", label="Recall")
plt.title("Validation F1 / Precision / Recall")
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(epochs, reward, marker="o", label="Reward (AUC)")
plt.plot(epochs, best_reward, linestyle="--", label="Best Reward")
plt.title("Controller Reward (AUC)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()