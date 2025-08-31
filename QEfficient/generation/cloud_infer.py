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

    def enable_outputs(self, names: List[str]):
        """
        Re-enable previously skipped outputs by recreating QBuffers and restoring
        ``buf_dims`` from the selected bindings (element size & dims).
        """
        for name in names:
            if name not in self.binding_index_map:
                continue
            idx = self.binding_index_map[name]
            binding = self.bindings[idx]  # selected_set binding
            # Recreate buffer to the selected size
            self.qbuffers[idx] = qaicrt.QBuffer(bytes(binding.size))
            # Restore buf_dims elem size & dims from the binding
            elem_size = aic_to_np_dtype_mapping[binding.type].itemsize
            dims = list(binding.dims)
            self.buf_dims[idx] = (elem_size, dims)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Execute on cloud AI 100

        Args:
            :inputs (Dict[str, np.ndarray]): Processed numpy inputs for the model.

        Return:
            :Dict[str, np.ndarray]:
        """
        import os
        import time

        t0 = time.perf_counter()
        self.set_buffers(inputs)
        t_set = time.perf_counter()
        if (
            self.execObj.setData(self.qbuffers, self.buf_dims)
            != qaicrt.QStatus.QS_SUCCESS
        ):
            raise MemoryError("Failed to setData")
        t_setdata = time.perf_counter()
        if self.queue.enqueue(self.execObj) != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to enqueue")
        t_enqueue = time.perf_counter()
        if self.execObj.waitForCompletion() != qaicrt.QStatus.QS_SUCCESS:
            error_message = "Failed to run"
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
        t_wait = time.perf_counter()
        status, output_qbuffers = self.execObj.getData()
        if status != qaicrt.QStatus.QS_SUCCESS:
            raise MemoryError("Failed to getData")
        t_get = time.perf_counter()
        outputs = {}
        total_bytes = 0
        for output_name in self.output_names:
            buffer_index = self.binding_index_map[output_name]
            if self.qbuffers[buffer_index].size == 0:
                continue
            arr = np.frombuffer(
                bytes(output_qbuffers[buffer_index]),
                aic_to_np_dtype_mapping[self.bindings[buffer_index].type],
            )
            shaped = arr.reshape(tuple(self.buf_dims[buffer_index][1]))
            outputs[output_name] = shaped
            try:
                total_bytes += int(arr.size) * arr.itemsize
            except Exception:
                pass
        t_unmarshal = time.perf_counter()

        if os.getenv("QEFF_SPEC_DEBUG", ""):
            try:
                print(
                    "[qaic:run] set=%.4f setData=%.4f enqueue=%.4f wait=%.4f get=%.4f unmarshal=%.4f bytes=%.2fMB"
                    % (
                        (t_set - t0),
                        (t_setdata - t_set),
                        (t_enqueue - t_setdata),
                        (t_wait - t_enqueue),
                        (t_get - t_wait),
                        (t_unmarshal - t_get),
                        total_bytes / 1e6,
                    ),
                    flush=True,
                )
            except Exception:
                pass
        return outputs
