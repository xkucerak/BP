from torch.utils.data import DataLoader
import torch
from torchvision import datasets
from torchvision.models import ResNet50_Weights
import onnxruntime as ort
import onnx
import torchmetrics


def create_val_loader():
    weights = ResNet50_Weights.DEFAULT
    dataset = datasets.ImageFolder(root="../archive", transform=weights.transforms())

    SIZE = 2**12

    BATCH = 2**8
    subset, _ = torch.utils.data.random_split(
        dataset,
        [SIZE, len(dataset) - SIZE],
        generator=torch.Generator().manual_seed(21),
    )

    val_loader = DataLoader(
        dataset=subset,
        batch_size=BATCH,
        # shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    return val_loader


def bench_model(model: onnx.ModelProto, val_loader):
    so = ort.SessionOptions()
    so.log_severity_level = 3

    session = ort.InferenceSession(
        model.SerializeToString(),
        sess_options=so,
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
