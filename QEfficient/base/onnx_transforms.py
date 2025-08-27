# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

from typing import List, Optional, Tuple

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
    """Inject per-chunk token-importance scoring into the speculator graph."""

    def apply(self, model: onnx.ModelProto, **kwargs) -> Tuple[onnx.ModelProto, bool]:
        g = model.graph
        changed = False

        q_names: List[str] = []
        layer_idx = 0
        while True:
            name = f"prefill_query_{layer_idx}"
            if any(v.name == name for v in list(g.value_info) + list(g.output)):
                q_names.append(name)
                layer_idx += 1
            else:
                break
        if not q_names:
            for v in g.output:
                if v.name.startswith("prefill_query_"):
                    q_names.append(v.name)
            q_names = sorted(q_names, key=lambda s: int(s.rsplit("_", 1)[-1]))
        L = len(q_names)
        if L == 0:
            return model, changed

        past_key_names: List[str] = []
        i = 0
        while True:
            pk = f"past_key.{i}_RetainedState"
            if any(o.name == pk for o in g.output):
                past_key_names.append(pk)
                i += 1
            else:
                break
        if len(past_key_names) != L:
            return model, changed

        def const(name: str, arr: np.ndarray) -> onnx.NodeProto:
            t = numpy_helper.from_array(arr.astype(np.float32), name=name)
            g.initializer.extend([t])
            vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, list(arr.shape))
            g.value_info.extend([vi])
            return helper.make_node("Identity", [name], [name], name=name + "_id")

        layer_scores: List[str] = []
        for li in range(L):
            q_name = q_names[li]
            pk_out = past_key_names[li]

            sq_q = f"{q_name}_squeeze"
            g.node.extend([helper.make_node("Squeeze", [q_name], [sq_q], name=sq_q, axes=[0])])

            pk_squeeze = f"{pk_out}_squeeze"
            g.node.extend([helper.make_node("Squeeze", [pk_out], [pk_squeeze], name=pk_squeeze, axes=[0])])
            pk_perm = f"{pk_squeeze}_perm"
            g.node.extend([helper.make_node("Transpose", [pk_squeeze], [pk_perm], name=pk_perm, perm=[0, 2, 1])])

            kt = f"{pk_perm}_kt"
            g.node.extend([helper.make_node("Transpose", [pk_perm], [kt], name=kt, perm=[0, 2, 1])])

            mm = f"qk_mm_{li}"
            g.node.extend([helper.make_node("MatMul", [sq_q, kt], [mm], name=mm)])

            scale_name = f"{mm}_scale"
            scale_const = f"{mm}_scale_c"
            const_node = const(scale_const, np.array(1.0, dtype=np.float32))
            g.node.extend([const_node, helper.make_node("Mul", [mm, scale_const], [scale_name], name=scale_name)])

            pos_vi = next((inp for inp in g.input if inp.name == "position_ids"), None)
            if pos_vi is None:
                masked = scale_name
            else:
                ge = f"mask_ge_{li}"
                cast = f"mask_f_{li}"
                g.node.extend([
                    helper.make_node(
                        "GreaterOrEqual", ["position_ids", "zero_i64"], [ge], name=ge
                    )
                ])
                if not any(ini.name == "zero_i64" for ini in g.initializer):
                    g.initializer.extend([numpy_helper.from_array(np.array(0, dtype=np.int64), name="zero_i64")])
                g.node.extend([helper.make_node("Cast", [ge], [cast], name=cast, to=TensorProto.FLOAT)])
                one = f"one_f_{li}"
                if not any(ini.name == one for ini in g.initializer):
                    g.initializer.extend([numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=one)])
                inv = f"inv_mask_{li}"
                g.node.extend([helper.make_node("Sub", [one, cast], [inv], name=inv)])
                neg = f"neg_bias_{li}"
                if not any(ini.name == neg for ini in g.initializer):
                    g.initializer.extend([numpy_helper.from_array(np.array(-1e9, dtype=np.float32), name=neg)])
                bias = f"pad_bias_{li}"
                g.node.extend([helper.make_node("Mul", [inv, neg], [bias], name=bias)])
                add = f"{scale_name}_bias"
                g.node.extend([helper.make_node("Add", [scale_name, bias], [add], name=add)])
                masked = add

            sm = f"softmax_{li}"
            g.node.extend([helper.make_node("Softmax", [masked], [sm], name=sm, axis=-1)])

            rmax = f"rmax_heads_{li}"
            g.node.extend([helper.make_node("ReduceMax", [sm], [rmax], name=rmax, axes=[0], keepdims=0)])
            layer_scores.append(rmax)

        cat = "importance_cat_layers"
        g.node.extend([helper.make_node("Concat", layer_scores, [cat], name=cat, axis=0)])
        out_name = "importance_chunk"
        rmean = "importance_rmean"
        g.node.extend([helper.make_node("ReduceMean", [cat], [out_name], name=rmean, axes=[0], keepdims=0)])

        if not any(o.name == out_name for o in g.output):
            g.output.extend([helper.make_tensor_value_info(out_name, TensorProto.FLOAT, None)])
        changed = True
        return model, changed

