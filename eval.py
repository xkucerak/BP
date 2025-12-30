from torch.utils.data import DataLoader, Dataset
import torchmetrics
import torch
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from softtools.eval import create_val_loader
import onnxruntime as ort
import numpy as np
from tqdm import tqdm


val_loader = create_val_loader()


def bench_model(onnx_path):

    session = ort.InferenceSession(
        f"{onnx_path}",
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    inp_name = session.get_inputs()[0].name
    inp_type = session.get_inputs()[0].type

    top1 = torchmetrics.Accuracy(top_k=1, task="multiclass", num_classes=1000)
    top5 = torchmetrics.Accuracy(top_k=5, task="multiclass", num_classes=1000)

    for inputs, labels in val_loader:

        if inp_type == "tensor(float16)":
            inputs = inputs.to(torch.float16)

        outputs = session.run(None, {inp_name: inputs.numpy()})[0]

        outputs = torch.tensor(outputs)

        top1.update(outputs, labels)
        top5.update(outputs, labels)

    return (top1.compute(), top5.compute())


from pathlib import Path
import csv
import time

if __name__ == "__main__":

    path = Path.cwd() / "models/"

    file_path = "raw_onnx.csv"

    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["local_path", "model", "model_type", "chunk_id", "top1", "top5"]
        )

        for file in tqdm(list(path.rglob("*.onnx"))):
            if file.suffix == ".onnx":
                # top1, top5 =
                top1, top5 = bench_model(str(file))
                print(top1, top5)
                writer.writerow(
                    [
                        str(file),
                        file.parent.parent.name,
                        file.parent.name,
                        file.stem,
                        top1.item(),
                        top5.item(),
                    ]
                )
                csvfile.flush()
