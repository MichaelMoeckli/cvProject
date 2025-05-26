import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.data_module import ImageDataModule
from models.lightning_model import Classifier
from pytorch_lightning import Trainer

def main():
    data_dir = "data/animals10"

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset not found at: {data_dir}")

    datamodule = ImageDataModule(data_dir=data_dir, batch_size=32)
    model = Classifier(num_classes=10, lr=1e-3)

    trainer = Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
