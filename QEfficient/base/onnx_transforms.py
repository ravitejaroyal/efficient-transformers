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
    Append a benign FP16 zeros tensor 'importance_chunk' with dynamic shape
    [1, S, H], where S is derived from `position_ids` (S=chunk_len for prefill,
    S=1 for decode). This avoids any reliance on export-example seq_len and
    matches compiler specializations automatically.
    """

    @staticmethod
    def apply(model, **kwargs) -> Tuple["onnx.ModelProto", bool]:
        from onnx import helper, TensorProto

        g = model.graph

        # ---- Hidden size (from config if provided; fallback to 2048) ----
        H = int(kwargs.get("hidden_size", 2048))

        # ---- Derive seq_len S from position_ids shape ----
        # 1) locate position_ids input (shape [B,S])
        pos_name = None
        for inp in g.input:
            if inp.name == "position_ids":
                pos_name = inp.name
                break
        # 2) Shape(position_ids) -> [B, S]
        if pos_name is not None:
            g.node.extend([
                helper.make_node("Shape", [pos_name], ["importance_pos_shape"],
                                 name="importance_pos_shape")
            ])
            # 3) Gather index=1 (seq dim) along axis=0 -> scalar S
            # Provide index as a scalar Constant so Gather returns a scalar
            g.node.extend([
                helper.make_node(
                    "Constant", [], ["importance_idx_one"],
                    name="importance_idx_one_const",
                    value=helper.make_tensor(
                        name="importance_idx_one_val", data_type=TensorProto.INT64, dims=[], vals=[1]
                    ),
                )
            ])
            g.node.extend([
                helper.make_node(
                    "Gather",
                    ["importance_pos_shape", "importance_idx_one"],
                    ["importance_seq_len_scalar"],
                    name="importance_gather_seq", axis=0,
                )
            ])
            # 4) Unsqueeze scalar -> [S] (axes as tensor input per opset-13)
            axes0 = helper.make_tensor(
                name="importance_axes0", data_type=TensorProto.INT64, dims=[1], vals=[0]
            )
            if not any(init.name == "importance_axes0" for init in g.initializer):
                g.initializer.append(axes0)
            g.node.extend([
                helper.make_node("Unsqueeze",
                                 ["importance_seq_len_scalar", "importance_axes0"],
                                 ["importance_seq_len_1d"],
                                 name="importance_unsqueeze_seq")
            ])
            seq_input = "importance_seq_len_1d"
        else:
            # Should not happen; fallback to S=32
            seq_const = helper.make_tensor(
                name="importance_seq_fallback", data_type=TensorProto.INT64, dims=[1], vals=[32]
            )
            if not any(init.name == "importance_seq_fallback" for init in g.initializer):
                g.initializer.append(seq_const)
            seq_input = "importance_seq_fallback"
            try:
                print("[transform] WARNING: `position_ids` not found; falling back to S=32", flush=True)
            except Exception:
                pass

        # ---- Build target shape [1, S, H] as a 1-D INT64 vector ----
        one_vec = helper.make_tensor(
            name="importance_one_vec", data_type=TensorProto.INT64, dims=[1], vals=[1]
        )
        hid_vec = helper.make_tensor(
            name="importance_hidden_vec", data_type=TensorProto.INT64, dims=[1], vals=[H]
        )
        for t in (one_vec, hid_vec):
            if not any(init.name == t.name for init in g.initializer):
                g.initializer.append(t)
        g.node.extend([
            helper.make_node("Concat",
                             ["importance_one_vec", seq_input, "importance_hidden_vec"],
                             ["importance_shape_1sH"],
                             name="importance_shape_concat",
                             axis=0)
        ])

        # ---- ConstantOfShape -> FP32 zeros [1,S,H] (value must be 1-D float) ----
        zero_f32 = helper.make_tensor(
            name="importance_zero_f32", data_type=TensorProto.FLOAT, dims=[1], vals=[0.0]
        )
        if not any(init.name == "importance_zero_f32" for init in g.initializer):
            g.initializer.append(zero_f32)
        g.node.extend([
            helper.make_node("ConstantOfShape",
                             ["importance_shape_1sH"],
                             ["importance_zeros_f32"],
                             name="importance_zeros",
                             value=zero_f32)
        ])

        # ---- Cast to FP16, bind to graph output 'importance_chunk' ----
        g.node.extend([
            helper.make_node("Cast", ["importance_zeros_f32"], ["importance_zeros_fp16"],
                             name="importance_cast_fp16", to=TensorProto.FLOAT16)
        ])
        out_name = "importance_chunk"
        if not any(o.name == out_name for o in g.output):
            # Rank-3 ValueInfo (symbolic dims); compiler specializes seq_len at compile/run time
            g.output.extend([
                helper.make_tensor_value_info(out_name, TensorProto.FLOAT16,
                                              ["batch", "seq_len", "hidden"])
            ])
        g.node.extend([
            helper.make_node("Identity", ["importance_zeros_fp16"], [out_name],
                             name="importance_out_bind")
        ])
        return model, True


class AttachEnergyImportance(OnnxTransform):
    """
    Device placeholder: per-token 'energy' importance with pad masking.
    Reduces keys across heads and dim, masks pads using position_ids == -1,
    broadcasts to [1, S, H] and exposes FP16 output 'importance_chunk'.
    """

    @staticmethod
    def apply(model, **kwargs):  # type: ignore[override]
        g = model.graph

        # ---- Hidden size (config or default 2048) ----
        H = int(kwargs.get("hidden_size", 2048))

        # ---- Find position_ids and derive S ----
        pos_name = None
        for inp in g.input:
            if inp.name == "position_ids":
                pos_name = inp.name
                break
        if pos_name is None:
            try:
                print("[transform] WARNING: position_ids not found; skipping AttachEnergyImportance", flush=True)
            except Exception:
                pass
            return model, False

        g.node.extend([helper.make_node("Shape", [pos_name], ["imp_pos_shape"], name="imp_pos_shape")])
        idx1 = helper.make_tensor("imp_idx1", TensorProto.INT64, [1], [1])
        if not any(i.name == "imp_idx1" for i in g.initializer):
            g.initializer.append(idx1)
        g.node.extend([
            helper.make_node("Gather", ["imp_pos_shape", "imp_idx1"], ["imp_seq_scalar"], name="imp_gather_seq", axis=0)
        ])
        axes0 = helper.make_tensor("imp_axes0", TensorProto.INT64, [1], [0])
        if not any(i.name == "imp_axes0" for i in g.initializer):
            g.initializer.append(axes0)
        g.node.extend([
            helper.make_node("Unsqueeze", ["imp_seq_scalar", "imp_axes0"], ["imp_seq_1d"], name="imp_unsq_seq")
        ])

        # ---- Build target shape [1,S,H] ----
        one_vec = helper.make_tensor("imp_one_vec", TensorProto.INT64, [1], [1])
        hid_vec = helper.make_tensor("imp_hidden_vec", TensorProto.INT64, [1], [H])
        for t in (one_vec, hid_vec):
            if not any(i.name == t.name for i in g.initializer):
                g.initializer.append(t)
        g.node.extend([
            helper.make_node("Concat", ["imp_one_vec", "imp_seq_1d", "imp_hidden_vec"], ["imp_shape_1sH"], name="imp_shape_concat", axis=0)
        ])

        # ---- Ones tensor for broadcast along hidden ----
        one_f32 = helper.make_tensor("imp_one_f32", TensorProto.FLOAT, [1], [1.0])
        if not any(i.name == "imp_one_f32" for i in g.initializer):
            g.initializer.append(one_f32)
        g.node.extend([
            helper.make_node("ConstantOfShape", ["imp_shape_1sH"], ["imp_ones_1sH_f32"], name="imp_ones", value=one_f32)
        ])

        # ---- Locate a keys tensor ----
        key_name = None
        for o in g.output:
            if o.name.startswith("past_key.") and o.name.endswith("_RetainedState"):
                key_name = o.name
                break
        if key_name is None:
            try:
                print("[transform] WARNING: no past_key.*_RetainedState found; skipping AttachEnergyImportance", flush=True)
            except Exception:
                pass
            return model, False

        g.node.extend([
            helper.make_node("ReduceSum", [key_name], ["imp_energy_1S_f32"], name="imp_reduce_sum", axes=[1, 3], keepdims=0)
        ])

        # ---- Pad mask from position_ids ----
        neg1 = helper.make_tensor("imp_neg1", TensorProto.INT64, [1], [-1])
        if not any(i.name == "imp_neg1" for i in g.initializer):
            g.initializer.append(neg1)
        g.node.extend([
            helper.make_node("Greater", [pos_name, "imp_neg1"], ["imp_mask_bool_1S"], name="imp_mask_gt")
        ])
        g.node.extend([
            helper.make_node("ReduceMax", ["imp_mask_bool_1S"], ["imp_mask_bool_S"], name="imp_mask_dropB", axes=[0], keepdims=0)
        ])
        g.node.extend([
            helper.make_node("Cast", ["imp_mask_bool_S"], ["imp_mask_1S_f32"], name="imp_mask_cast", to=TensorProto.FLOAT)
        ])

        g.node.extend([
            helper.make_node("Mul", ["imp_energy_1S_f32", "imp_mask_1S_f32"], ["imp_energy_masked_1S_f32"], name="imp_energy_mask")
        ])

        axes2 = helper.make_tensor("imp_axes2", TensorProto.INT64, [1], [2])
        if not any(i.name == "imp_axes2" for i in g.initializer):
            g.initializer.append(axes2)
        g.node.extend([
            helper.make_node("Unsqueeze", ["imp_energy_masked_1S_f32", "imp_axes2"], ["imp_energy_1S1_f32"], name="imp_unsq_energy")
        ])
        g.node.extend([
            helper.make_node("Mul", ["imp_energy_1S1_f32", "imp_ones_1sH_f32"], ["imp_energy_1sH_f32"], name="imp_broadcast_hidden")
        ])

        g.node.extend([
            helper.make_node("Cast", ["imp_energy_1sH_f32"], ["imp_energy_1sH_fp16"], name="imp_cast_fp16", to=TensorProto.FLOAT16)
        ])

        out_name = "importance_chunk"
        if not any(o.name == out_name for o in g.output):
            g.output.extend([helper.make_tensor_value_info(out_name, TensorProto.FLOAT16, None)])
        g.node.extend([
            helper.make_node("Identity", ["imp_energy_1sH_fp16"], [out_name], name="imp_bind_output")
        ])
        return model, True


