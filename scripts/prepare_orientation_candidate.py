"""Prepare the pinned upstream orientation model for PhoSca evaluation."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


SOURCE_SHA256 = "cffe911c1dff47fbfbbd90110aaab9c07134645c460d35b3ae8832079bea91ba"
UPSTREAM_SIZE = 384
INSET = 16
RUNTIME_SIZE = UPSTREAM_SIZE + 2 * INSET
MEAN = np.float32([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
STD = np.float32([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


def prepare_model(source: bytes) -> bytes:
    if sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the upstream v2 ONNX artifact with the pinned SHA-256")

    model = onnx.load_model_from_string(source)
    graph = model.graph
    model_input = graph.input[0]
    upstream_input = model_input.name
    model_input.name = "phosca_rgb"
    for dimension in model_input.type.tensor_type.shape.dim[2:]:
        dimension.dim_value = RUNTIME_SIZE

    for name, values in (
        ("phosca_crop_start", np.int64([INSET, INSET])),
        ("phosca_crop_end", np.int64([INSET + UPSTREAM_SIZE] * 2)),
        ("phosca_crop_axes", np.int64([2, 3])),
        ("phosca_mean", MEAN),
        ("phosca_std", STD),
    ):
        graph.initializer.append(numpy_helper.from_array(values, name))

    # PhoSca already resizes to the ONNX input size and supplies RGB in [0, 1].
    preprocessing = [
        helper.make_node(
            "Slice",
            ["phosca_rgb", "phosca_crop_start", "phosca_crop_end", "phosca_crop_axes"],
            ["phosca_center"],
        ),
        helper.make_node("Sub", ["phosca_center", "phosca_mean"], ["phosca_centered"]),
        helper.make_node("Div", ["phosca_centered", "phosca_std"], [upstream_input]),
    ]
    for node in reversed(preprocessing):
        graph.node.insert(0, node)
    model.metadata_props.add(key="phosca.source_sha256", value=SOURCE_SHA256)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def verify_logits(source: bytes, prepared: bytes) -> None:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    reference = ort.InferenceSession(source, options, providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(prepared, options, providers=["CPUExecutionProvider"])
    rgb = np.random.default_rng(412).random((2, 3, RUNTIME_SIZE, RUNTIME_SIZE), dtype=np.float32)
    center = rgb[:, :, INSET:-INSET, INSET:-INSET]
    expected = reference.run(None, {reference.get_inputs()[0].name: (center - MEAN) / STD})[0]
    actual = candidate.run(None, {candidate.get_inputs()[0].name: rgb})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Upstream orientation_model_v2_0.9882.onnx")
    parser.add_argument("--output", type=Path, required=True, help="Destination for the evaluation model")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Use a different path for the prepared model")
    try:
        source = args.source.read_bytes()
        prepared = prepare_model(source)
        verify_logits(source, prepared)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as output:
            output.write(prepared)
    except (OSError, ValueError, AssertionError) as error:
        parser.error(str(error))
    print(f"{sha256(prepared).hexdigest()}  {args.output}")


if __name__ == "__main__":
    main()
