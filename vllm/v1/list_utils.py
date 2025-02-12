# SPDX-License-Identifier: Apache-2.0

from typing import Generic, List, Optional, Sequence, TypeVar, Union, overload

import numpy as np

T = TypeVar("T")


class NumpyList(Generic[T], Sequence):
    """A Numpy-backed list."""

    def __init__(
        self,
        dtype: np.dtype,
        x: Optional[List[T]] = None,
        initial_capacity: Optional[int] = None,
    ):
        self.dtype = dtype
        if not x:
            self.size = 0
            if initial_capacity is None:
                initial_capacity = 16
            self.capacity = initial_capacity
            self.data = np.empty(self.capacity, dtype=self.dtype)
        else:
            self.size = len(x)
            if initial_capacity is not None:
                assert initial_capacity >= self.size
            else:
                # If no initial capacity is provided, double the size.
                initial_capacity = 2 * self.size
            self.capacity = initial_capacity
            self.data = np.empty(self.capacity, dtype=self.dtype)
            self.data[:self.size] = x

    def _resize(self, new_capacity: int) -> None:
        new_data = np.empty(new_capacity, dtype=self.dtype)
        new_data[:self.size] = self.data[:self.size]
        self.data = new_data
        self.capacity = new_capacity

    def append(self, value) -> None:
        if self.size >= self.capacity:
            # Double the capacity.
            self._resize(2 * self.capacity)
        self.data[self.size] = value
        self.size += 1

    def extend(self, x: List[T]) -> None:
        num_new_items = len(x)

        # Increase the capacity if necessary.
        new_capacity = self.capacity
        while new_capacity < self.size + num_new_items:
            new_capacity = new_capacity * 2
        if new_capacity > self.capacity:
            self._resize(new_capacity)

        self.data[self.size:self.size + num_new_items] = x
        self.size += num_new_items

    def index(
        self,
        value: T,
        start: int = 0,
        stop: Optional[int] = None,
    ) -> int:
        """Find the index of the first occurrence of the value. If not found,
        raise a ValueError."""
        data = self.data[:self.size][start:stop]
        indices = np.where(data == value)[0]
        if len(indices) == 0:
            raise ValueError(f"Value {value} not found in the list.")
        return int(indices[0])

    def get_numpy_array(self) -> np.ndarray:
        return self.data[:self.size]

    @overload
    def __getitem__(self, i: int) -> T:
        ...

    @overload
    def __getitem__(self, s: slice, /) -> List[T]:
        ...

    def __getitem__(self, i: Union[int, slice]) -> Union[T, List[T]]:
        """If i is a slice, return a list. Otherwise, return the item."""
        # Truncate the array before indexing, to handle negative indices.
        return self.data[:self.size][i].tolist()

    @overload
    def __setitem__(self, i: int, value: T):
        ...

    @overload
    def __setitem__(self, s: slice, value: Union[T, List[T]]):
        ...

    def __setitem__(self, i: Union[int, slice], value: Union[T, List[T]]):
        # Truncate the array before indexing, to handle negative indices.
        self.data[:self.size][i] = value

    def __len__(self):
        return self.size

    def __iter__(self):
        for i in range(self.size):
            yield self.data[i]

    def __repr__(self):
        return f"NumpyList({self.data[:self.size]!r})"
