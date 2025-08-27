# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import Optional, Tuple

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, external_data_helper, helper, numpy_helper


class OnnxTransform:
    """
    OnnxTransform is the base class for graph modifications on exported onnx.
    """

    def __init__(self):
        raise TypeError("Transform classes are not to be instantiated. Directly use the `apply` method.")

    @classmethod
    def apply(cls, model: ModelProto, **kwargs) -> Tuple[ModelProto, bool]:
        """
        Override this class to apply a transformation.
        :param model: The model's ONNX graph to transform
        :param kwargs: Parameters needed for specific transforms. All transforms should take **kwargs to ignore unneeded kwargs.

        :returns: ONNX graph after applying the transform
        :returns: Boolean indicating whether transform was applied
        """
        raise NotImplementedError("Use subclasses for ONNX transform")


class FP16ClipTransform(OnnxTransform):
    """
    Clips the tensor values to be in FP16 range, but preserves -inf values.
    """

    @classmethod
    def apply(cls, model: ModelProto, *, onnx_base_dir: Optional[str] = None, **kwargs) -> Tuple[ModelProto, bool]:
        """
        :param onnx_base_dir: Base directory to load tensors
        """
        finfo = np.finfo(np.float16)
        fp16_max = finfo.max
        fp16_min = finfo.min
        transformed = False

        for tensor in external_data_helper._get_all_tensors(model):
            nptensor = numpy_helper.to_array(tensor, onnx_base_dir)
            if nptensor.dtype == np.float32 and (np.any(nptensor > fp16_max) or np.any(nptensor < fp16_min)):
                neg_inf_mask = np.isinf(nptensor) & (nptensor < 0)
                clipped_tensor = np.clip(nptensor, fp16_min, fp16_max)

                # Restore -inf values
                if neg_inf_mask.any():
                    clipped_tensor = np.where(neg_inf_mask, np.float32("-inf"), clipped_tensor)

                new_tensor = numpy_helper.from_array(clipped_tensor, tensor.name)
                tensor.CopyFrom(new_tensor)
                transformed = True

        return model, transformed


class SplitTensorsTransform(OnnxTransform):
    """
    Split external tensors file
    """

    @classmethod
    def apply(
        cls,
        model: ModelProto,
        *,
        model_name: str,
        onnx_base_dir: Optional[str] = None,
        file_chunk_size: int = 10 * 2**30,  # 10 GiB
        size_threshold: int = 1024,
        **kwargs,
    ) -> Tuple[ModelProto, bool]:
        """
        :param model_name: Used for naming external files. i.e. {model_name}_0.onnx.data
        :param onnx_base_dir: Base directory to load tensors (if not already loaded).
        :param file_chunk_size: Chunk size to split external files into.
        :param size_threshold: Only tensors greater than this threshold (in bytes) will be saved externally.
        """
        file_num = 0
        current_file_size = 0
        transformed = False
        external_data_helper.load_external_data_for_model(model, onnx_base_dir)
        for tensor in external_data_helper._get_all_tensors(model):
            if tensor.HasField("raw_data") and ((tsize := len(tensor.raw_data)) > size_threshold):
                transformed = True
                current_file_size += tsize
                if current_file_size > file_chunk_size:
                    file_num += 1
                    current_file_size = tsize
                external_data_helper.set_external_data(tensor, f"{model_name}_{file_num}.onnx.data")
        return model, transformed


class AttachProbeOutput(OnnxTransform):
    """
    Append a benign FP32 zeros tensor as an extra graph output 'probe_chunk'
    with concrete dims [1, S_cap, H]. This is gated from _export via env var
    and does NOT require any runtime buffer handling.
    """

    @staticmethod
    def apply(model, **kwargs) -> Tuple["onnx.ModelProto", bool]:
        g = model.graph

        # Read compile-time export hints; fallback to safe defaults.
        S = int(kwargs.get("seq_len", 32))
        H = int(kwargs.get("hidden_size", 2048))
        try:
            print(
                f"[transform] {type(AttachProbeOutput).__name__} kwargs: seq_len={S}, hidden_size={H}",
                flush=True,
            )
        except Exception:
            pass

        # Build INT64 shape [1, S, H] as an initializer
        shape_name = "probe_shape_i64"
        shape_init = helper.make_tensor(
            name=shape_name, data_type=TensorProto.INT64, dims=[3], vals=[1, S, H]
        )
        g.initializer.append(shape_init)

        # Make Constant node that yields the shape tensor
        shape_const = helper.make_node(
            "Constant", [], ["probe_shape"], name="probe_shape_const", value=shape_init
        )
        g.node.extend([shape_const])

        # ConstantOfShape → FP32 zeros of [1,S,H]
        # NOTE: For opset-13, ConstantOfShape.value must be a 1-D tensor (not scalar).
        zero_f32 = helper.make_tensor(
            name="probe_zero_f32",
            data_type=TensorProto.FLOAT,
            dims=[1],  # 1-D vector, length 1
            vals=[0.0],
        )
        zeros = helper.make_node(
            "ConstantOfShape",
            ["probe_shape"],
            ["probe_chunk_f32"],
            name="probe_zeros",
            value=zero_f32,
        )
        # Register both the node and its attribute tensor; QAIC resolves the attribute via the initializer table.
        g.node.extend([zeros])
        # avoid duplicate initializer by name if re-running
        if not any(init.name == "probe_zero_f32" for init in g.initializer):
            g.initializer.append(zero_f32)

        # Register graph output with concrete dims, FP32
        if not any(o.name == "probe_chunk_f32" for o in g.output):
            g.output.extend(
                [helper.make_tensor_value_info("probe_chunk_f32", TensorProto.FLOAT, [1, S, H])]
            )

        return model, True


