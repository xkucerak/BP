from torch.utils.data import DataLoader, Dataset
import torchmetrics
import torch
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import onnxruntime as ort
import onnx
import numpy as np
from tqdm import tqdm
from softtools.bit_op import BitPack
from enum import StrEnum
import gc
from pathlib import Path
import csv
from softtools.eval import create_val_loader, bench_model
from softtools.model_manipulator import ModelManipulate


class Type(StrEnum):
    fp32 = "fp32"
    fp16 = "fp16"
    int8 = "int8"
    int4 = "int4"


if __name__ == "__main__":

    # currently onnx does have memory leak in its raw_data and whole model needs to be crated
    iter_count = 0

    path = Path.cwd() / "model_chunks_corr/"

    val_loader = create_val_loader()

    cvs_path = "model_chunks_corr.csv"

    with open(cvs_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "model",
                "model_type",
                "chunk_id",
                "top1",
                "top5",
            ]
        )

        last_model_name = None

        for file in tqdm(sorted(list(path.rglob("*.bin")))):
            actual_model_name = file.parent.stem

            if last_model_name != actual_model_name:
                print(actual_model_name)
                iter_count = 0
                weights_offset = 0
                weights_offset_len = 0
                last_model_name = actual_model_name

                TYPE = Type(str(file.parent.stem).split("_")[-1])

                onnx_path = (
                    Path.cwd()
                    / "models"
                    / file.parent.parent.stem
                    / str(file.parent.stem + ".onnx")
                )

                model, target_nodes, weights_dict, all_weights = (
                    ModelManipulate.prep_model(onnx_path)
                )

            # memory leak prevention from onnx package
            if iter_count > 10:
                iter_count = 0
                model, target_nodes, weights_dict, all_weights = (
                    ModelManipulate.prep_model(onnx_path)
                )

            all = all_weights.copy()

            new_data = file.read_bytes()
            if TYPE == Type.int4:
                new_data = BitPack.unpack_int4(new_data)

            # apply changed chunk to bytes
            weights_offset_len = weights_offset + len(new_data)
            all[weights_offset:weights_offset_len] = bytearray(new_data)
            weights_offset = weights_offset_len

            all = memoryview(all)

            start = 0
            for (
                name
            ) in target_nodes:  # apply changed weights to name of node they are in
                stop = start + len(weights_dict[name].raw_data)
                weights_dict[name].raw_data = bytes(all[start:stop])
                start = stop

            iter_count += 1

            if True:
                top1, top5 = bench_model(model, val_loader)
                print(top1, top5)
                writer.writerow(
                    [
                        file.parent.parent.name,
                        file.parent.name,
                        file.stem,
                        top1.item(),
                        top5.item(),
                    ]
                )
                csvfile.flush()
