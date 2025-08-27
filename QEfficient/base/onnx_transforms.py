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

        # Expand to [S_cap, hidden] then add batch → [1, S_cap, hidden]
        # --- Opset 13-compliant constants ---
        axes1 = helper.make_tensor("importance_axes1", TensorProto.INT64, [1], [1])
        g.initializer.append(axes1)
        axes0 = helper.make_tensor("importance_axes0", TensorProto.INT64, [1], [0])
        g.initializer.append(axes0)
        shape_const = helper.make_tensor(
            name="importance_expand_shape",
            data_type=TensorProto.INT64,
            dims=[2],
            vals=[seq_len_const, hidden_size],
        )
        g.initializer.append(shape_const)

        # Unsqueeze to [S_cap,1] then Expand to [S_cap, hidden]
        imp_unsq = "importance_unsq"
        g.node.extend(
            [
                helper.make_node(
                    "Unsqueeze",
                    [imp_fp16_1d, "importance_axes1"],
                    [imp_unsq],
                    name="importance_unsqueeze",
                )
            ]
        )
        imp_exp = "importance_expanded"
        g.node.extend(
            [
                helper.make_node(
                    "Expand",
                    [imp_unsq, "importance_expand_shape"],
                    [imp_exp],
                    name="importance_expand",
                )
            ]
        )

        # Add batch dim → [1, S_cap, hidden]
        imp_unsq_batch = "importance_unsq_batch"
        g.node.extend(
            [
                helper.make_node(
                    "Unsqueeze",
                    [imp_exp, "importance_axes0"],
                    [imp_unsq_batch],
                    name="importance_add_batch",
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
                    [imp_unsq_batch],
                    [out_name],
                    name="importance_out_bind",
                )
            ]
        )
        print(
            "[transform] AttachSpecPrefillScoring: appended graph output 'importance_chunk' (FLOAT16 [1,S_cap,hidden])"
        )
        return model, True
