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


class AttachSpecPrefillScoring(OnnxTransform):
    @staticmethod
    def apply(
        model: "onnx.ModelProto",
        *,
        seq_len_const: int = 128,
        hidden_size: int = 2048,
        **kwargs,
    ):
        g = model.graph

        # If already present, do nothing
        if any(o.name == "importance_chunk" for o in g.output):
            return model, False

        # Expect an upstream FP32 tensor to cast from. If your scoring head already
        # produces "importance_fp32", reuse it. Otherwise, STOP and add the compute first.
        source_name = None
        # Try to find a 1D FP32 tensor in value_infos (best-effort)
        for vi in list(g.value_info) + list(g.output):
            if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
                source_name = vi.name
                break
        if source_name is None:
            print(
                "[transform] AttachSpecPrefillScoring: no FP32 source tensor found; not adding output"
            )
            return model, False

        # Cast → [S_cap] FP16
        imp_fp16_1d = "importance_fp16_1d"
        g.node.extend(
            [
                helper.make_node(
                    "Cast",
                    [source_name],
                    [imp_fp16_1d],
                    name="importance_cast_fp16",
                    to=TensorProto.FLOAT16,
                )
            ]
        )

        # Opset-13: Unsqueeze uses axes as input tensors
        axes_add_batch = helper.make_tensor(
            "importance_axes_add_batch", TensorProto.INT64, [1], [0]
        )  # [S_cap]→[1,S_cap]
        axes_add_hidden = helper.make_tensor(
            "importance_axes_add_hidden", TensorProto.INT64, [1], [2]
        )  # [1,S_cap]→[1,S_cap,1]
        g.initializer.extend([axes_add_batch, axes_add_hidden])

        # From [S_cap] → [1,S_cap] → [1,S_cap,1]
        imp_1S = "importance_1S"
        g.node.extend(
            [
                helper.make_node(
                    "Unsqueeze",
                    [imp_fp16_1d, "importance_axes_add_batch"],
                    [imp_1S],
                    name="importance_unsqueeze_batch",
                )
            ]
        )
        imp_1S1 = "importance_1S1"
        g.node.extend(
            [
                helper.make_node(
                    "Unsqueeze",
                    [imp_1S, "importance_axes_add_hidden"],
                    [imp_1S1],
                    name="importance_unsqueeze_hidden",
                )
            ]
        )

        # Make zeros [1,1,H] via ConstantOfShape (FP32) → Cast FP16
        hidden = int(kwargs.get("hidden_size", hidden_size))
        shape_1x1xH = helper.make_tensor(
            "importance_shape_1x1xH", TensorProto.INT64, [3], [1, 1, hidden]
        )
        g.initializer.append(shape_1x1xH)
        zeros_fp32 = "importance_zeros_fp32"
        g.node.extend(
            [
                helper.make_node(
                    "ConstantOfShape",
                    ["importance_shape_1x1xH"],
                    [zeros_fp32],
                    name="importance_zeros_fp32",
                )
            ]
        )
        zeros_fp16 = "importance_zeros_fp16"
        g.node.extend(
            [
                helper.make_node(
                    "Cast",
                    [zeros_fp32],
                    [zeros_fp16],
                    name="importance_zeros_cast",
                    to=TensorProto.FLOAT16,
                )
            ]
        )

        # Broadcast ADD: [1,S_cap,1] + [1,1,H] → [1,S_cap,H]
        out_1SH = "importance_1SH"
        g.node.extend(
            [
                helper.make_node(
                    "Add",
                    [imp_1S1, zeros_fp16],
                    [out_1SH],
                    name="importance_broadcast_add",
                )
            ]
        )

        # Final graph output (FP16, [1, seq_len, hidden])
        out_name = "importance_chunk"
        if not any(o.name == out_name for o in g.output):
            g.output.extend(
                [
                    helper.make_tensor_value_info(
                        out_name,
                        TensorProto.FLOAT16,
                        ["batch", "seq_len", "hidden"],
                    )
                ]
            )
        g.node.extend(
            [
                helper.make_node(
                    "Identity",
                    [out_1SH],
                    [out_name],
                    name="importance_out_bind",
                )
            ]
        )
        print(
            "[transform] AttachSpecPrefillScoring: appended graph output 'importance_chunk' (FLOAT16 [1,S_cap,hidden])"
        )
        return model, True
