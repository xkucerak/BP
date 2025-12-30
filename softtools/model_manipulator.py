import onnx
import os
import pathlib
from .bit_op import BitPack
from enum import StrEnum
from onnx import ModelProto


class Type(StrEnum):
    fp32 = "fp32"
    fp16 = "fp16"
    int8 = "int8"
    int4 = "int4"


class ModelManipulate:

    _SEARCH = [
        (lambda i: i.op_type == "Clip" and "weight_quant" in i.name, 0),
        (lambda i: i.op_type == "DequantizeLinear" and "weight_quant" in i.name, 0),
        (lambda i: i.op_type == "Conv" or i.op_type == "Gemm", 1),
    ]

    # finds in conv and fc nodes and names of weights inputs
    @staticmethod
    def find_nodes(model: ModelProto):
        found = False
        target_nodes = {}

        for op, index in ModelManipulate._SEARCH:
            for i in model.graph.node:
                if op(i):
                    found = True
                    target_nodes[i.input[index]] = i.name
            if found == True:
                break

        return target_nodes

    # gets all data needed for chunk replacements
    @staticmethod
    def prep_model(onnx_path):
        model = onnx.load(onnx_path)
        target_nodes = ModelManipulate.find_nodes(model)
        weights_dict = {
            i.name: i for i in model.graph.initializer
        }  # extract names of weights

        all_weights = bytearray()
        for name in target_nodes:
            all_weights.extend(weights_dict[name].raw_data)

        return model, target_nodes, weights_dict, all_weights

    # dump weights of model
    @staticmethod
    def dump_model(source_path: pathlib.Path, destination_path: pathlib.Path):
        NAME, TYPE = source_path.stem.split("_")
        TYPE = Type(TYPE.lower())
        model = onnx.load(str(source_path))

        target_nodes = ModelManipulate.find_nodes(model)
        weights_dict = {
            i.name: i for i in model.graph.initializer
        }  # extract names of weights

        to_dump = []

        for name in target_nodes:  # connects weights to name of node they are in
            if name in weights_dict:
                to_dump.append([target_nodes[name], weights_dict[name].raw_data])

        save_path = destination_path / NAME / (NAME + "_" + TYPE)
        save_path.mkdir(parents=True, exist_ok=True)

        print("Dumping:", source_path, "to:", save_path)

        for i, (name, data) in enumerate(to_dump):
            file_name = str(i).zfill(3) + name.replace("/", "-") + ".bin"

            with open(save_path / file_name, "wb") as file:
                if TYPE == Type.int4:
                    data = BitPack.pack_int4(data)
                file.write(data)

    @staticmethod
    def load_full_model(
        source_path: pathlib.Path,
        destination_path: pathlib.Path,
        model_path: pathlib.Path | None = None,
        by_layer=False,
    ):
        # Loads weights to model and saves it.
        NAME, TYPE = source_path.parent.stem.split("_")
        TYPE = Type(TYPE.lower())

        if model_path is None:
            model_path = (
                pathlib.Path("models").resolve()
                / source_path.parent.parent.stem
                / (str(source_path.parent.stem) + ".onnx")
            )

        model = onnx.load(str(model_path))

        target_nodes = ModelManipulate.find_nodes(model)
        weights_dict = {i.name: i for i in model.graph.initializer}

        save_path = destination_path / NAME / (NAME + "_" + TYPE)

        if by_layer:
            save_path = save_path / source_path.stem

        save_path.mkdir(parents=True, exist_ok=True)

        i = 0
        for name in target_nodes:
            if name in weights_dict:
                file_name = (
                    str(i).zfill(3) + target_nodes[name].replace("/", "-") + ".bin"
                )
                target_file = source_path / file_name

                if by_layer:
                    backup = weights_dict[name].raw_data

                with open(target_file, "rb") as file:
                    data = file.read()

                if TYPE == Type.int4:
                    data = bytearray(data)
                    data = BitPack.unpack_int4(data)
                    data = bytes(data)

                weights_dict[name].raw_data = data

                if by_layer:
                    onnx.save(model, (save_path / (str(i).zfill(3) + ".onnx")))
                    weights_dict[name].raw_data = backup

                i += 1

        if by_layer is False:
            onnx.save(model, (save_path / (str(source_path.stem) + ".onnx")))
