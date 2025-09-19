# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# ----------------------------------------------------------------------------

import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, external_data_helper, helper, numpy_helper, shape_inference


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


class ScoringHeadTransform(OnnxTransform):
    """Insert device-side scoring head taps for selected transformer layers."""

    @classmethod
    def apply(cls, model: ModelProto, **cfg) -> Tuple[ModelProto, bool]:
        debug_enabled = os.getenv("QEFF_SCORING_DEBUG", "0") == "1"
        probe_enabled = os.getenv("QEFF_SCORING_PROBE", "0") == "1"

        def dprint(*args):
            if debug_enabled:
                print("[ScoringHeadTransform]", *args)

        def pprint_shape(name: str, shapes: Dict[str, List[Optional[int]]]) -> str:
            dims = shapes.get(name, None)
            if not dims:
                return "[]"
            return "[" + ",".join(str(x) for x in dims) + "]"

        layers_for_scoring = str(cfg.get("layers_for_scoring", "all"))
        emit_single_output = bool(cfg.get("emit_single_output", False))
        importance_dtype = str(cfg.get("importance_dtype", "float16")).lower()
        smoothing_window = int(cfg.get("smoothing_window", 3))
        layer_name_regex = cfg.get("layer_name_regex", r"model\.layers\.(\d+)\.self_attn")
        q_last_suffix = cfg.get("q_last_suffix", "Gather_2_output_0")
        k_transpose_suffix = cfg.get("k_transpose_suffix", "Transpose_1_output_0")

        dtype_map = {
            "float16": TensorProto.FLOAT16,
            "fp16": TensorProto.FLOAT16,
            "float32": TensorProto.FLOAT,
            "fp32": TensorProto.FLOAT,
        }
        dtype_enum = dtype_map.get(importance_dtype, TensorProto.FLOAT16)
        np_dtype = np.float16 if dtype_enum == TensorProto.FLOAT16 else np.float32

        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:  # pragma: no cover - best effort
            dprint(f"initial shape inference skipped: {exc}")

        if probe_enabled:
            try:
                # IR / opset imports summary
                import onnx

                print("[ScoringHeadTransform][PROBE] IR:", getattr(model, "ir_version", None))
                print(
                    "[ScoringHeadTransform][PROBE] opset_import:",
                    [(imp.domain or "", imp.version) for imp in model.opset_import],
                )
            except Exception as _:
                pass

        graph = model.graph
        node_by_output = {}
        for node in graph.node:
            for output in node.output:
                node_by_output[output] = node

        layer_pattern = re.compile(layer_name_regex)

        def collect_shapes() -> Dict[str, List[Optional[int]]]:
            shapes: Dict[str, List[Optional[int]]] = {}

            def update_from_vi(vi):
                if not vi or not vi.type.HasField("tensor_type"):
                    return
                dims: List[Optional[int]] = []
                for dim in vi.type.tensor_type.shape.dim:
                    if dim.HasField("dim_value"):
                        dims.append(dim.dim_value)
                    elif dim.HasField("dim_param"):
                        dims.append(dim.dim_param)
                    else:
                        dims.append(None)
                shapes[vi.name] = dims

            for vi in list(graph.value_info) + list(graph.output) + list(graph.input):
                update_from_vi(vi)

            for initializer in graph.initializer:
                shapes[initializer.name] = list(initializer.dims)

            return shapes

        shape_dict = collect_shapes()

        def get_dim_value(name: str, index: int) -> Optional[int]:
            dims = shape_dict.get(name)
            if not dims or index >= len(dims):
                return None
            value = dims[index]
            return value if isinstance(value, int) else None

        all_layers: List[int] = []
        for node in graph.node:
            for output in node.output:
                if not output.endswith(k_transpose_suffix):
                    continue
                match = layer_pattern.search(output)
                if match:
                    try:
                        all_layers.append(int(match.group(1)))
                    except ValueError:
                        continue

        all_layers = sorted(set(all_layers))
        if not all_layers:
            dprint("no eligible layers located")
            return model, False

        def resolve_layers(spec: str, layer_count: int) -> List[int]:
            spec = spec.strip()
            spec_lower = spec.lower()
            if spec_lower == "all":
                return list(range(layer_count))
            if spec_lower.startswith("last"):
                suffix = spec[4:]
                try:
                    count = int(suffix)
                except ValueError:
                    count = 4
                start = max(0, layer_count - count)
                return list(range(start, layer_count))
            if spec_lower.startswith("indices:"):
                body = spec.split(":", 1)[1]
                indices: List[int] = []
                for item in body.split(","):
                    item = item.strip()
                    if item:
                        try:
                            indices.append(int(item))
                        except ValueError:
                            continue
                return indices
            return list(range(layer_count))

        resolved_layers = resolve_layers(layers_for_scoring, max(all_layers) + 1)
        resolved_layer_set = set(resolved_layers)
        available_layers = [idx for idx in all_layers if idx in resolved_layer_set]
        if not available_layers:
            dprint("no layers selected after filtering")
            return model, False

        if probe_enabled:
            # Probe: print candidates for first few available layers
            max_probe = int(os.getenv("QEFF_SCORING_PROBE_LAYERS", "4"))
            print(
                f"[ScoringHeadTransform][PROBE] examining up to {max_probe} layers out of {len(available_layers)}"
            )
            # Build a map from output name -> node for quick lookup
            node_by_output = {}
            for n in graph.node:
                for o in n.output:
                    node_by_output[o] = n
            for li in available_layers[:max_probe]:
                prefix = f"model/layers.{li}/self_attn/"
                # Gather all outputs under this layer's self_attn scope
                cand_transpose = []
                cand_gather = []
                cand_softmax = []
                for n in graph.node:
                    for o in n.output:
                        if o.startswith(prefix):
                            if n.op_type == "Transpose":
                                cand_transpose.append(o)
                            elif n.op_type == "Gather":
                                cand_gather.append(o)
                            elif n.op_type == "Softmax":
                                cand_softmax.append(o)
                print(f"[PROBE][layer {li}] TRANSPOSE outputs:")
                for o in cand_transpose:
                    print("   ", o, pprint_shape(o, shape_dict))
                print(f"[PROBE][layer {li}] GATHER outputs:")
                for o in cand_gather:
                    print("   ", o, pprint_shape(o, shape_dict))
                # Mask/Where path that feeds Softmax
                # Find Softmax in this layer and track its input
                for so in cand_softmax:
                    sm_node = node_by_output.get(so, None)
                    src = sm_node.input[0] if sm_node and sm_node.input else None
                    wn = node_by_output.get(src, None)
                    print(
                        f"[PROBE][layer {li}] Softmax: {so}  in: {src}  producer:",
                        (wn.op_type if wn else None),
                    )
                    if wn and wn.op_type == "Where":
                        print(f"[PROBE][layer {li}] Where inputs: {wn.input}")
                        # print shapes for where inputs
                        for inp in wn.input:
                            print("    ", inp, pprint_shape(inp, shape_dict))

        existing_initializers = {init.name for init in graph.initializer}
        existing_outputs = {out.name for out in graph.output}

        def ensure_initializer(name: str, values, dtype) -> str:
            if name in existing_initializers:
                return name
            array = np.array(values, dtype=dtype)
            tensor = numpy_helper.from_array(array, name)
            graph.initializer.append(tensor)
            existing_initializers.add(name)
            return name

        layer_mask_bias: Dict[int, Tuple[str, str]] = {}
        for node in graph.node:
            if node.op_type != "Softmax" or not node.output:
                continue
            output_name = node.output[0]
            match = layer_pattern.search(output_name)
            if not match:
                continue
            try:
                layer_idx = int(match.group(1))
            except ValueError:
                continue
            source_name = node.input[0]
            where_node = node_by_output.get(source_name)
            if where_node and where_node.op_type == "Where" and len(where_node.input) == 3:
                mask_name, bias_name, _ = where_node.input
                layer_mask_bias[layer_idx] = (mask_name, bias_name)

        applied_layers: List[int] = []
        per_layer_results: List[Tuple[int, str]] = []

        for layer_idx in available_layers:
            q_name = f"model/layers.{layer_idx}/self_attn/{q_last_suffix}"
            k_name = f"model/layers.{layer_idx}/self_attn/{k_transpose_suffix}"

            if q_name not in node_by_output or k_name not in node_by_output:
                dprint(f"layer {layer_idx}: missing Q({q_name in node_by_output}) or K({k_name in node_by_output}) tap")
                continue

            applied_layers.append(layer_idx)

            q_squeeze = f"model/layers.{layer_idx}/importance/Q_squeeze"
            k_squeeze = f"model/layers.{layer_idx}/importance/K_squeeze"
            graph.node.append(
                helper.make_node("Squeeze", [q_name], [q_squeeze], name=q_squeeze, axes=[0])
            )
            graph.node.append(
                helper.make_node("Squeeze", [k_name], [k_squeeze], name=k_squeeze, axes=[0])
            )

            k_transposed = f"model/layers.{layer_idx}/importance/K_transpose"
            graph.node.append(
                helper.make_node("Transpose", [k_squeeze], [k_transposed], name=k_transposed, perm=[0, 2, 1])
            )

            q_unsqueezed = f"model/layers.{layer_idx}/importance/Q_unsqueeze"
            graph.node.append(
                helper.make_node("Unsqueeze", [q_squeeze], [q_unsqueezed], name=q_unsqueezed, axes=[1])
            )

            q_heads = get_dim_value(q_name, 1)   # Q: [1,H,D]
            q_dim   = get_dim_value(q_name, 2)   # D from Q if shape info present
            k_heads = get_dim_value(k_name, 1)   # K: [1,H_kv,S,D]
            k_dim   = get_dim_value(k_name, 3)   # D from K if Q dim missing
            factor = 1
            if q_heads is not None and k_heads is not None:
                if q_heads != k_heads:
                    if q_heads % k_heads != 0:
                        raise ValueError(
                            f"ScoringHeadTransform: layer {layer_idx} incompatible head dims q={q_heads}, k={k_heads}"
                        )
                    factor = q_heads // k_heads
            if factor > 1:
                # replicate K along head dim using distinct inputs for Concat
                concat_inputs = []
                for t in range(factor):
                    k_copy = f"model/layers.{layer_idx}/importance/K_copy{t}"
                    graph.node.append(helper.make_node("Identity", [k_transposed], [k_copy], name=k_copy))
                    concat_inputs.append(k_copy)
                k_tiled = f"model/layers.{layer_idx}/importance/K_tiled"
                graph.node.append(
                    helper.make_node("Concat", concat_inputs, [k_tiled], name=k_tiled, axis=0)
                )
                matmul_k_input = k_tiled
            else:
                matmul_k_input = k_transposed

            matmul = f"model/layers.{layer_idx}/importance/MatMul"
            matmul_out = f"model/layers.{layer_idx}/importance/MatMul_out"
            graph.node.append(helper.make_node("MatMul", [q_unsqueezed, matmul_k_input], [matmul_out], name=matmul))

            logits = f"model/layers.{layer_idx}/importance/MatMul_squeezed"
            graph.node.append(
                helper.make_node("Squeeze", [matmul_out], [logits], name=logits, axes=[1])
            )

            # derive D from Q first; if missing, fall back to K's last dim
            D = None
            if q_dim is not None and q_dim > 0:
                D = int(q_dim)
            elif k_dim is not None and k_dim > 0:
                D = int(k_dim)
            # If D is still unknown, keep a safe assert; we prefer failing fast to wrong scaling
            if D is None or D <= 0:
                raise ValueError(f"ScoringHeadTransform: cannot derive head dim D for layer {layer_idx}")
            scale_value = 1.0 / math.sqrt(float(D))
            scale_const_name = f"model/layers.{layer_idx}/importance/scale_const"
            ensure_initializer(scale_const_name, [scale_value], np_dtype)

            scaled_logits = f"model/layers.{layer_idx}/importance/scaled"
            graph.node.append(
                helper.make_node("Mul", [logits, scale_const_name], [scaled_logits], name=scaled_logits)
            )

            masked_input = scaled_logits
            if layer_idx in layer_mask_bias:
                mask_name, bias_name = layer_mask_bias[layer_idx]
                # reshape mask/bias to [1,S] so they broadcast to [H,S]
                shape_1S = f"model/layers.{layer_idx}/importance/shape_1S"
                ensure_initializer(shape_1S, [1, -1], np.int64)
                mask_1S = f"model/layers.{layer_idx}/importance/mask_1S"
                bias_1S = f"model/layers.{layer_idx}/importance/bias_1S"
                graph.node.append(helper.make_node("Reshape", [mask_name, shape_1S], [mask_1S], name=mask_1S))
                graph.node.append(helper.make_node("Reshape", [bias_name, shape_1S], [bias_1S], name=bias_1S))
                masked = f"model/layers.{layer_idx}/importance/masked"
                graph.node.append(helper.make_node("Where", [mask_1S, bias_1S, scaled_logits], [masked], name=masked))
                masked_input = masked

            softmax = f"model/layers.{layer_idx}/importance/softmax"
            graph.node.append(
                helper.make_node("Softmax", [masked_input], [softmax], name=softmax, axis=1)
            )

            reduce_max = f"model/layers.{layer_idx}/importance/reduce_max"
            reduce_output = f"model/layers.{layer_idx}/importance/heads_reduced"
            graph.node.append(
                helper.make_node("ReduceMax", [softmax], [reduce_output], name=reduce_max, axes=[0], keepdims=1)
            )

            final_tensor = reduce_output

            if smoothing_window > 1:
                reshape_in = f"model/layers.{layer_idx}/importance/reshape_in"
                reshape_in_shape = f"model/layers.{layer_idx}/importance/reshape_in_shape"
                ensure_initializer(reshape_in_shape, [1, 1, -1], np.int64)
                graph.node.append(
                    helper.make_node("Reshape", [final_tensor, reshape_in_shape], [reshape_in], name=reshape_in)
                )

                avg_pool = f"model/layers.{layer_idx}/importance/avg_pool"
                pad = smoothing_window // 2   # pads=[pad_left, pad_right] for 1D
                graph.node.append(
                    helper.make_node(
                        "AveragePool",
                        [reshape_in],
                        [avg_pool],
                        name=avg_pool,
                        kernel_shape=[smoothing_window],
                        strides=[1],
                        pads=[pad, pad],
                    )
                )

                reshape_out = f"model/layers.{layer_idx}/importance/reshape_out"
                reshape_out_shape = f"model/layers.{layer_idx}/importance/reshape_out_shape"
                ensure_initializer(reshape_out_shape, [1, -1], np.int64)
                graph.node.append(
                    helper.make_node("Reshape", [avg_pool, reshape_out_shape], [reshape_out], name=reshape_out)
                )
                final_tensor = reshape_out

            per_layer_results.append((layer_idx, final_tensor))

            if not emit_single_output:
                output_name = f"importance.layer{layer_idx}"
                if output_name not in existing_outputs:
                    vi = helper.make_tensor_value_info(output_name, dtype_enum, [1, "seq_len"])
                    graph.output.append(vi)
                    existing_outputs.add(output_name)
                identity = f"model/layers.{layer_idx}/importance/output"
                graph.node.append(
                    helper.make_node("Identity", [final_tensor], [output_name], name=identity)
                )

        if not per_layer_results:
            dprint("no layers processed successfully")
            return model, False

        if emit_single_output:
            unsqueezed_tensors: List[str] = []
            for layer_idx, tensor in per_layer_results:
                unsq = f"model/layers.{layer_idx}/importance/unsqueeze_final"
                graph.node.append(
                    helper.make_node("Unsqueeze", [tensor], [unsq], name=unsq, axes=[1])
                )
                unsqueezed_tensors.append(unsq)

            if len(unsqueezed_tensors) == 1:
                concat_output = unsqueezed_tensors[0]
            else:
                concat = "model/importance/concat_layers"
                concat_output = "model/importance/concat_layers_output"
                graph.node.append(
                    helper.make_node("Concat", unsqueezed_tensors, [concat_output], name=concat, axis=1)
                )

            reduce = "model/importance/reduce_layers"
            importance_output = "importance"
            graph.node.append(
                helper.make_node("ReduceMax", [concat_output], [importance_output], name=reduce, axes=[1], keepdims=0)
            )
            if importance_output not in existing_outputs:
                vi = helper.make_tensor_value_info(importance_output, dtype_enum, [1, "seq_len"])
                graph.output.append(vi)
                existing_outputs.add(importance_output)

        try:
            model = shape_inference.infer_shapes(model)
        except Exception as exc:  # pragma: no cover - best effort
            dprint(f"final shape inference skipped: {exc}")

        dprint(
            f"applied to layers {applied_layers}; emit_single_output={emit_single_output}; smoothing={smoothing_window}"
        )

        return model, True

