import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from models.lightning_model import Classifier


# ----------- Config -----------

data_dir = "data/animals10"
checkpoint_path = "lightning_logs/version_0/checkpoints/"  # Update if needed
batch_size = 32
num_classes = 10

# ----------- Load model -----------

# Find the latest checkpoint
ckpt_file = os.listdir(checkpoint_path)[0]
ckpt_path = os.path.join(checkpoint_path, ckpt_file)
model = Classifier.load_from_checkpoint(ckpt_path, num_classes=num_classes)
model.eval()

# ----------- Load validation data -----------

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ----------- Run Predictions -----------

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        images, labels = batch
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

# ----------- Metrics -----------

print("\n✅ Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=val_dataset.classes))

# ----------- Confusion Matrix -----------

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=val_dataset.classes, yticklabels=val_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png") 
