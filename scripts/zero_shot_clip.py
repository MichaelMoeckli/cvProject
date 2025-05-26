from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import classification_report

# ----- Config -----
data_dir = os.getenv("DATA_DIR", "data/animals10/val")  # Configurable via environment variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
top_k = 1  # Use top-1 accuracy

# ----- Load model & processor -----
try:
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
except Exception as e:
    print(f"Error loading CLIP model or processor: {e}")
    exit(1)

# ----- Load class names -----
try:
    labels = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )  # Ensure only directories are considered
except FileNotFoundError:
    print(f"Error: Data directory '{data_dir}' not found.")
    exit(1)

# ----- Image transform (for PIL) -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# ----- Load dataset using PIL -----
try:
    dataset = ImageFolder(root=data_dir, transform=transform)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Load original PIL images to feed into CLIP
images = [Image.open(path).convert("RGB") for path, _ in dataset.samples]
true_labels = [dataset.classes[label] for _, label in dataset.samples]


# ----- Run zero-shot predictions -----
print("Running zero-shot classification...")
predicted_labels = []

for image in tqdm(images):
    try:
        inputs = processor(images=image, text=labels, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)
            pred_idx = probs.argmax().item()
            predicted_labels.append(labels[pred_idx])
    except Exception as e:
        print(f"Error processing image: {e}")
        predicted_labels.append("unknown")

# ----- Evaluate -----
print("\n✅ Zero-Shot Classification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=labels))