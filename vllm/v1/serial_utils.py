# SPDX-License-Identifier: Apache-2.0

import pickle
from collections.abc import Sequence
from dataclasses import asdict
from inspect import isclass
from itertools import chain
from types import FunctionType
from typing import Any, Optional, Union

import cloudpickle
import numpy as np
import torch
import zmq
from msgspec import msgpack

from vllm.multimodal.inputs import (MultiModalFieldElem, MultiModalKwargs,
                                    MultiModalKwargsItem, NestedTensors)

CUSTOM_TYPE_PICKLE = 1
CUSTOM_TYPE_CLOUDPICKLE = 2

# TODO calibrate this size
INLINE_BUF_SIZE_THRESHOLD = 256

bytestr = Union[bytes, bytearray, memoryview, zmq.Frame]


class MsgpackEncoder:
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    def __init__(self):
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        # This is used as a local stash of buffers that we can then access from
        # our custom `msgspec` hook, `enc_hook`. We don't have a way to
        # pass custom data to the hook otherwise.
        self.aux_buffers: Optional[list[bytestr]] = None

    def encode(self, obj: Any) -> Sequence[bytestr]:
        try:
            self.aux_buffers = bufs = [b'']
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
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
            mm: MultiModalKwargs = obj
            if mm.modalities:
                # ignore the main dict, it will be re-indexed.
                # pass a list of MultiModalKwargsItem, then see below
                # Any tensors *not* indexed by modality will be ignored.
                return [mm.get_items(m) for m in mm.modalities]
            # just return the main dict if there are no modalities
            return {k: v for k, v in obj.items()}

        if isinstance(obj, MultiModalKwargsItem):
            # Encode as plain dictionary + special handling for '.field'
            rd = {}
            for k, v in obj.items():
                vv = asdict(v)
                vv['field'] = pickle.dumps(v.field,
                                           protocol=pickle.HIGHEST_PROTOCOL)
                rd[k] = vv
            return rd

        if isinstance(obj, FunctionType):
            # `pickle` is generally faster than cloudpickle, but can have
            # problems serializing methods.
            return msgpack.Ext(CUSTOM_TYPE_CLOUDPICKLE, cloudpickle.dumps(obj))

        return msgpack.Ext(CUSTOM_TYPE_PICKLE,
                           pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def _encode_ndarray(
        self, obj: np.ndarray
    ) -> tuple[str, tuple[int, ...], Union[int, memoryview]]:
        assert self.aux_buffers is not None
        if not obj.shape or obj.nbytes < INLINE_BUF_SIZE_THRESHOLD:
            # Encode small arrays and scalars inline.
            data = obj.data
        else:
            # Otherwise encode index of backing buffer.
            obj = np.ascontiguousarray(obj)
            data = len(self.aux_buffers)
            self.aux_buffers.append(obj.data)
        # We serialize the ndarray as a tuple of native types.
        # The data is either inlined if small, or an index into a list of
        # backing buffers that we've stashed in `aux_buffers`.
        return obj.dtype.str, obj.shape, data


class MsgpackDecoder:
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    def __init__(self, t: Optional[Any] = None):
        args = () if t is None else (t, )
        self.decoder = msgpack.Decoder(*args,
                                       ext_hook=self.ext_hook,
                                       dec_hook=self.dec_hook)
        self.aux_buffers: Sequence[bytestr] = ()

    def decode(self, bufs: Union[bytestr, Sequence[bytestr]]) -> Any:
        if isinstance(bufs, (bytes, bytearray, memoryview, zmq.Frame)):
            # TODO - This check can become `isinstance(bufs, bytestr)`
            # as of Python 3.10.
            return self.decoder.decode(bufs)

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])
        finally:
            self.aux_buffers = ()

    def dec_hook(self, t: type, obj: Any) -> Any:
        # Given native types in `obj`, convert to type `t`.
        if isclass(t):
            if issubclass(t, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(t, MultiModalKwargs) and isinstance(obj, dict):
                return MultiModalKwargs(
                    {k: self._decode_nested(obj[k])
                     for k in obj})
            if issubclass(t, MultiModalKwargs) and isinstance(obj, list):
                return MultiModalKwargs.from_items(self._decode_items(obj))
            if issubclass(t, torch.Tensor):
                return torch.from_numpy(self._decode_ndarray(obj))
        return obj

    def _decode_ndarray(self, arr: Any) -> np.ndarray:
        dtype, shape, data = arr
        buffer = self.aux_buffers[data] if isinstance(data, int) else data
        return np.ndarray(buffer=buffer, dtype=np.dtype(dtype), shape=shape)

    def _decode_items(self, obj: list) -> list[MultiModalKwargsItem]:
        all = []
        for item in chain.from_iterable(obj):
            elems = []
            for v in item.values():
                v['data'] = self._decode_nested(v['data'])
                v['field'] = pickle.loads(v['field'])
                elems.append(MultiModalFieldElem(**v))
            all.append(MultiModalKwargsItem.from_elems(elems))
        return all

    def _decode_nested(self, obj: Any) -> NestedTensors:
        if isinstance(obj, list) and isinstance(obj[0], str):
            return torch.from_numpy(self._decode_ndarray(obj))
        if isinstance(obj, list):
            return [self._decode_nested(x) for x in obj]
        raise TypeError(f"Unexpected NestedArray contents: {obj}")

    def ext_hook(self, code: int, data: memoryview) -> Any:
        if code == CUSTOM_TYPE_PICKLE:
            return pickle.loads(data)
        if code == CUSTOM_TYPE_CLOUDPICKLE:
            return cloudpickle.loads(data)

        raise NotImplementedError(
            f"Extension type code {code} is not supported")
