# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List, Optional, Union
from warnings import warn

import numpy as np

try:
    import qaicrt
except ImportError:
    import platform
    import sys

    sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
    import qaicrt

try:
    import QAicApi_pb2 as aicapi
except ImportError:
    import sys

    sys.path.append("/opt/qti-aic/dev/python")
    import QAicApi_pb2 as aicapi

aic_to_np_dtype_mapping = {
    aicapi.FLOAT_TYPE: np.dtype(np.float32),
    aicapi.FLOAT_16_TYPE: np.dtype(np.float16),
    aicapi.INT8_Q_TYPE: np.dtype(np.int8),
    aicapi.UINT8_Q_TYPE: np.dtype(np.uint8),
    aicapi.INT16_Q_TYPE: np.dtype(np.int16),
    aicapi.INT32_Q_TYPE: np.dtype(np.int32),
    aicapi.INT32_I_TYPE: np.dtype(np.int32),
    aicapi.INT64_I_TYPE: np.dtype(np.int64),
    aicapi.INT8_TYPE: np.dtype(np.int8),
}


class QAICInferenceSession:
    def __init__(
        self,
        qpc_path: Union[Path, str],
        device_ids: Optional[List[int]] = None,
        activate: bool = True,
        enable_debug_logs: bool = False,
    ):
        """
        Initialise for QAIC inference Session
        ---------

        :qpc_path: str. Path to the save generated binary file after compilation.
        :device_ids: List[int]. Device Ids to be used for compilation. if devices > 1, it enables multiple card setup.
        :activate: bool. If false, activation will be disabled. Default=True.
        :enable_debug_logs: bool. If True, It will enable debug logs. Default=False.
        """
        # Load QPC
        if device_ids is not None:
            devices = qaicrt.QIDList(device_ids)
            self.context = qaicrt.Context(devices)
            self.queue = qaicrt.Queue(self.context, device_ids[0])
        else:
            self.context = qaicrt.Context()
            self.queue = qaicrt.Queue(self.context, 0)  # Async API
        if enable_debug_logs:
            if (
                self.context.setLogLevel(qaicrt.QLogLevel.QL_DEBUG)
                != qaicrt.QStatus.QS_SUCCESS
            ):
                raise RuntimeError("Failed to setLogLevel")
        qpc = qaicrt.Qpc(str(qpc_path))
        # Load IO Descriptor
        iodesc = aicapi.IoDesc()
        status, iodesc_data = qpc.getIoDescriptor()
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to getIoDescriptor")
        iodesc.ParseFromString(bytes(iodesc_data))
        self.allowed_shapes = [
            [
                (aic_to_np_dtype_mapping[x.type].itemsize, list(x.dims))
                for x in allowed_shape.shapes
            ]
            for allowed_shape in iodesc.allowed_shapes
        ]
        self.bindings = iodesc.selected_set.bindings
        self.binding_index_map = {binding.name: binding.index for binding in self.bindings}
        # Cache compiled (selected-set) dims by name for later reshape logic
        self._compiled_dims_by_name = {b.name: list(b.dims) for b in self.bindings}
        # Optional: log compiled vs allowed for importance_chunk (diagnostic)
        imp_idx = self.binding_index_map.get("importance_chunk")
        if imp_idx is not None:
            try:
                print(
                    "[qaic:init] importance_chunk compiled dims =",
                    self._compiled_dims_by_name["importance_chunk"],
                )
                print(
                    "[qaic:init] importance_chunk allowed      =",
                    [x[imp_idx][1] for x in self.allowed_shapes],
                )
            except Exception:
                pass
        # Create and load Program
        prog_properties = qaicrt.QAicProgramProperties()
        prog_properties.SubmitRetryTimeoutMs = 60_000
        if device_ids and len(device_ids) > 1:
            prog_properties.devMapping = ":".join(map(str, device_ids))
        self.program = qaicrt.Program(self.context, None, qpc, prog_properties)
        if self.program.load() != qaicrt.QStatus.QS_SUCCESS:
            raise RuntimeError("Failed to load program")
        if activate:
            self.activate()
        # Create input qbuffers and buf_dims
        self.qbuffers = [
            qaicrt.QBuffer(bytes(binding.size)) for binding in self.bindings
        ]
        self.buf_dims = qaicrt.BufferDimensionsVecRef(
            [
                (aic_to_np_dtype_mapping[binding.type].itemsize, list(binding.dims))
                for binding in self.bindings
            ]
        )

    @property
    def input_names(self) -> List[str]:
        return [
            binding.name
            for binding in self.bindings
            if binding.dir == aicapi.BUFFER_IO_TYPE_INPUT
        ]

    @property
    def output_names(self) -> List[str]:
        return [
            binding.name
            for binding in self.bindings
            if binding.dir == aicapi.BUFFER_IO_TYPE_OUTPUT
        ]

    def activate(self):
        """Activate qpc"""

        self.program.activate()
        self.execObj = qaicrt.ExecObj(self.context, self.program)

    def deactivate(self):
        """Deactivate qpc"""

        del self.execObj
        self.program.deactivate()

    def set_buffers(self, buffers: Dict[str, np.ndarray]):
        """
        Provide buffer mapping for input and output

        Args:
            :buffer (Dict[str, np.ndarray]): Parameter for buffer mapping.
        """

        for buffer_name, buffer in buffers.items():
            if buffer_name not in self.binding_index_map:
                warn(f'Buffer: "{buffer_name}" not found')
                continue
            buffer_index = self.binding_index_map[buffer_name]
            self.qbuffers[buffer_index] = qaicrt.QBuffer(buffer.tobytes())
            self.buf_dims[buffer_index] = (
                buffer.itemsize,
                buffer.shape if len(buffer.shape) > 0 else (1,),
            )

    def skip_buffers(self, skipped_buffer_names: List[str]):
        """
        skip buffer mapping for given list of buffer names

        Args:
            :skipped_buffer_name: List[str]. List of buffer name to be skipped.
        """

        self.set_buffers({k: np.array([]) for k in skipped_buffer_names})

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Execute on cloud AI 100

        Args:
            :inputs (Dict[str, np.ndarray]): Processed numpy inputs for the model.

        Return:
            :Dict[str, np.ndarray]:
        """
        # Set inputs
        self.set_buffers(inputs)
        # ------------------------------------------------------------------
        # Dynamic dims fix: outputs like 'importance_chunk' have sequence-
        # dependent shapes (prefill S=prompt_len, decode S=1). The selected
        # set may carry a flattened core-tensor for this binding; update only
        # the dims tuple in self.buf_dims to a specialization that matches
        # the current S (derived from inputs).
        # ------------------------------------------------------------------
        try:
            # 1) Determine S from inputs (prefer position_ids over input_ids)
            S = None
            if (
                "position_ids" in inputs
                and getattr(inputs["position_ids"], "ndim", 0) >= 2
            ):
                S = int(inputs["position_ids"].shape[1])
            elif "input_ids" in inputs and getattr(inputs["input_ids"], "ndim", 0) >= 2:
                S = int(inputs["input_ids"].shape[1])
            if S is None:
                S = 1  # conservative fallback (decode step)

            # 2) Specialize dims for outputs with sequence-dependent shapes
            def _set_dyn_dims(binding_name: str) -> None:
                if binding_name not in self.binding_index_map:
                    return
                idx = self.binding_index_map[binding_name]
                # allowed_shapes is a list of shape-sets; pick dims for this idx
                if not self.allowed_shapes:
                    return
                allowed_for_idx = []
                for shape_set in self.allowed_shapes:
                    # shape_set[idx] is a tuple: (elem_size, dims_tuple)
                    try:
                        allowed_for_idx.append(shape_set[idx][1])
                    except Exception:
                        pass
                # Prefer dims whose last two slots match (S, hidden), but also
                # accept rank-3 [1,S,H]. This covers MDP/TS 5D and non-MDP 3D.
                target = None
                for d in allowed_for_idx:
                    if isinstance(d, (list, tuple)) and len(d) >= 3:
                        if d[-2] == S:  # match seq length
                            target = tuple(d)
                            break
                if target is not None:
                    # Keep element size; only swap dims tuple
                    self.buf_dims[idx] = (self.buf_dims[idx][0], target)

            # Apply to our dynamic output(s)
            _set_dyn_dims("importance_chunk")
        except Exception:
            # Fail-closed: leave buf_dims as-is if anything goes wrong here
            pass
        if (
            self.execObj.setData(self.qbuffers, self.buf_dims)
            != qaicrt.QStatus.QS_SUCCESS
        ):
            raise MemoryError("Failed to setData")
        # # Run with sync API
        # if self.execObj.run(self.qbuffers) != qaicrt.QStatus.QS_SUCCESS:
        # Run with async API
        if self.queue.enqueue(self.execObj) != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to enqueue")
        if self.execObj.waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            error_message = "Failed to run"
            # Print additional error messages for unmatched dimension error
            if self.allowed_shapes:
                error_message += "\n\n"
                error_message += (
                    '(Only if "No matching dimension found" error is present above)'
                )
                error_message += "\nAllowed shapes:"
                for i, allowed_shape in enumerate(self.allowed_shapes):
                    error_message += f"\n{i}\n"
                    for binding, (elemsize, shape), (_, passed_shape) in zip(
                        self.bindings, allowed_shape, self.buf_dims
                    ):
                        if passed_shape == [0]:
                            if not binding.is_partial_buf_allowed:
                                warn(f"Partial buffer not allowed for: {binding.name}")
                            continue
                        error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
                error_message += "\n\nPassed shapes:\n"
                for binding, (elemsize, shape) in zip(self.bindings, self.buf_dims):
                    if shape == [0]:
                        continue
                    error_message += f"{binding.name}:\t{elemsize}\t{shape}\n"
            raise ValueError(error_message)
        # Get output buffers
        status, output_qbuffers = self.execObj.getData()
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to getData")
        # Build output
        outputs = {}
        for output_name in self.output_names:
            buffer_index = self.binding_index_map[output_name]
            if self.qbuffers[buffer_index].size == 0:
                continue
            arr = np.frombuffer(
                bytes(output_qbuffers[buffer_index]),
                aic_to_np_dtype_mapping[self.bindings[buffer_index].type],
            )
            # Robust reshape for dynamic output compiled as flat core-tensor
            if output_name == "importance_chunk":
                # target dims chosen earlier for validator (buf_dims); may be 3D or 5D
                target_dims = tuple(self.buf_dims[buffer_index][1])
                need_elems = int(np.prod(target_dims))
                if arr.size != need_elems:
                    # Pick an allowed dims entry whose product matches buffer size
                    try:
                        idx = buffer_index
                        cand = None
                        for shape_set in self.allowed_shapes:
                            dims = shape_set[idx][1]
                            if isinstance(dims, (list, tuple)) and int(np.prod(dims)) == arr.size:
                                cand = tuple(dims)
                                break
                        if cand is not None:
                            target_dims = cand
                    except Exception:
                        pass
                arr = arr.reshape(target_dims)
                # Optional: expose the "logical" S (e.g., decode step S=1) by slicing
                try:
                    s_logical = None
                    if "position_ids" in inputs and getattr(inputs["position_ids"], "ndim", 0) >= 2:
                        s_logical = int(inputs["position_ids"].shape[1])
                    elif "input_ids" in inputs and getattr(inputs["input_ids"], "ndim", 0) >= 2:
                        s_logical = int(inputs["input_ids"].shape[1])
                    # slice only if device returned a larger S than we logically want
                    if s_logical is not None and target_dims[-2] > s_logical:
                        # handle both 3D [1,S,H] and 5D [...,S,H]
                        if len(target_dims) == 3:
                            arr = arr[:, :s_logical, :]
                        elif len(target_dims) == 5:
                            arr = arr[:, :, :, :s_logical, :]
                except Exception:
                    pass
                outputs[output_name] = arr
            else:
                outputs[output_name] = arr.reshape(tuple(self.buf_dims[buffer_index][1]))
        return outputs
