# SPDX-License-Identifier: Apache-2.0

import pickle
from collections.abc import Sequence
from inspect import isclass
from types import FunctionType
from typing import Any, Optional, Union

import cloudpickle
import msgspec
import numpy as np
import torch
import zmq
from msgspec import msgpack

from vllm.multimodal import MultiModalKwargs, NestedTensors

CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2

bytestr = Union[bytes, bytearray, memoryview, zmq.Frame]


class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization."""

    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        self.aux_buffers: Optional[list[bytestr]] = None

    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            self.aux_buffers = bufs = [b'']
            bufs[0] = self.encoder.encode(obj)
            return bufs
        finally:
            self.aux_buffers = None

    def encode_into(self, obj: Any, buf: bytearray) -> Sequence[bytestr]:
        try:
            self.aux_buffers = [buf]
            bufs = self.aux_buffers
            self.encoder.encode_into(obj, buf)
            return bufs
        finally:
            self.aux_buffers = None

    def enc_hook(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return self._encode_ndarray(obj.numpy())

        # Fall back to pickle for object or void kind ndarrays.
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ('O', 'V'):
            return self._encode_ndarray(obj)

        if isinstance(obj, MultiModalKwargs):
            return {k: NestedTensorsStruct.new(nt) for k, nt in obj.items()}

        if isinstance(obj, FunctionType):
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        return msgpack.Ext(CUSTOM_TYPE_PICKLE,
                           pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_ndarray(self, obj: np.ndarray) -> Any:
        assert self.aux_buffers is not None
        # Must get shape before calling ascontiguousarray since it will
        # convert scalars to arrays.
        shape = obj.shape
        obj = np.ascontiguousarray(obj)
        index = len(self.aux_buffers)
        self.aux_buffers.append(obj.data)
        return obj.dtype.str, shape, index


class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization."""

    def __init__(self, t: Optional[Any] = None):
        args = () if t is None else (t, )
        self.decoder = msgpack.Decoder(*args,
                                       ext_hook=self.ext_hook,
                                       dec_hook=self.dec_hook)
        self.aux_buffers: Sequence[bytestr] = ()

    def decode(self, bufs: Union[bytestr, Sequence[bytestr]]) -> Any:
        if isinstance(bufs, (bytes, bytearray, memoryview, zmq.Frame)):
            return self.decoder.decode(bufs)

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])
        finally:
            self.aux_buffers = ()

    def dec_hook(self, t: type, obj: Any) -> Any:
        if isclass(t):
            if issubclass(t, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(t, torch.Tensor):
                return torch.from_numpy(self._decode_ndarray(obj))

        if t is MultiModalKwargs:
            converted = msgspec.convert(obj,
                                        dict[str, NestedTensorsStruct],
                                        dec_hook=self.dec_hook)
            return MultiModalKwargs({
                k: nts.to_nested_tensors()
                for k, nts in converted.items()
            })

        return obj

    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        dtype, shape, index = arr
        return np.ndarray(buffer=self.aux_buffers[index],
                          dtype=np.dtype(dtype),
                          shape=shape)

    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_PICKLE:
            return pickle.loads(data)
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)

        raise NotImplementedError(
            f"Extension type code {code} is not supported")


class NestedTensorsStruct(msgspec.Struct):
    tensors: list[torch.Tensor]
    layout: Union[int, list]

    @staticmethod
    def new(nt: NestedTensors):
        tensors: list[torch.Tensor] = []
        return NestedTensorsStruct(tensors, _nt_to_layout(nt, tensors))

    def to_nested_tensors(self):
        return _layout_to_nt(self.layout, self.tensors)


def _nt_to_layout(nt: NestedTensors,
                  tensors: list[torch.Tensor]) -> Union[int, list]:
    if isinstance(nt, torch.Tensor):
        tensors.append(nt)
        return len(tensors) - 1
    assert isinstance(nt, (list, tuple))
    return [_nt_to_layout(sub, tensors) for sub in nt]


def _layout_to_nt(layout: Union[int, list],
                  tensors: list[torch.Tensor]) -> NestedTensors:
    return tensors[layout] if isinstance(layout, int) \
        else [_layout_to_nt(sub, tensors) for sub in layout]
