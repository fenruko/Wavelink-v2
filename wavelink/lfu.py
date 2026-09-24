"""
MIT License

Copyright (c) 2019-Current PythonistaGuild, EvieePy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .exceptions import WavelinkException


class CapacityZero(WavelinkException): ...


class _MissingSentinel:
    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        return False

    def __bool__(self) -> bool:
        return False

    def __hash__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "..."


class _NotFoundSentinel(_MissingSentinel):
    def __repr__(self) -> str:
        return "NotFound"


MISSING: Any = _MissingSentinel()
NotFound: Any = _NotFoundSentinel()


class DLLNode:
    __slots__ = ("value", "previous", "later")

    def __init__(self, value: Any | None = None, previous: DLLNode | None = None, later: DLLNode | None = None) -> None:
        self.value = value
        self.previous = previous
        self.later = later


@dataclass(slots=True)
class DataNode:
    key: Any
    value: Any
    frequency: int
    node: DLLNode


class LFUCache:
    def __init__(self, *, capacity: int) -> None:
        self._capacity = capacity
        self._cache: dict[Any, DataNode] = {}

        # Plain dict so empty frequency buckets can be dropped. A defaultdict
        # retained every frequency a hot key ever reached.
        self._freq_map: dict[int, DLL] = {}
        self._min: int = 1
        self._used: int = 0

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, key: Any) -> Any:
        if key not in self._cache:
            raise KeyError(f'"{key}" could not be found in LFU.')

        return self.get(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        return self.put(key, value)

    @property
    def capacity(self) -> int:
        return self._capacity

    def _bucket(self, frequency: int) -> DLL:
        bucket = self._freq_map.get(frequency)
        if bucket is None:
            bucket = DLL()
            self._freq_map[frequency] = bucket
        return bucket

    def get(self, key: Any, default: Any = MISSING) -> Any:
        data: DataNode | None = self._cache.get(key)
        if data is None:
            return default if default is not MISSING else NotFound

        freq = data.frequency
        bucket = self._freq_map[freq]
        bucket.remove(data.node)
        if not bucket:
            del self._freq_map[freq]

        # Reuse the node. The previous implementation allocated a DataNode on every hit.
        data.frequency = freq + 1
        self._bucket(freq + 1).append(data.node)

        if self._min == freq and freq not in self._freq_map:
            self._min = freq + 1

        return data.value

    def put(self, key: Any, value: Any) -> None:
        if self._capacity <= 0:
            raise CapacityZero("Unable to place item in LFU as capacity has been set to 0 or below.")

        existing = self._cache.get(key)
        if existing is not None:
            existing.value = value
            self.get(key)
            return

        if self._used == self._capacity:
            least_freq = self._freq_map.get(self._min)
            evicted = least_freq.popleft() if least_freq is not None else None

            if evicted is not None:
                self._cache.pop(evicted.value, None)
                self._used -= 1

            if least_freq is not None and not least_freq:
                del self._freq_map[self._min]

        node = DLLNode(key)
        data = DataNode(key=key, value=value, frequency=1, node=node)
        self._bucket(1).append(node)
        self._cache[key] = data

        self._used += 1
        self._min = 1


class DLL:
    __slots__ = ("head", "tail")

    def __init__(self) -> None:
        self.head: DLLNode = DLLNode()
        self.tail: DLLNode = DLLNode()

        self.head.later, self.tail.previous = self.tail, self.head

    def append(self, node: DLLNode) -> None:
        tail_prev: DLLNode | None = self.tail.previous
        tail: DLLNode | None = self.tail

        assert tail_prev and tail

        tail_prev.later = node
        tail.previous = node

        node.later = tail
        node.previous = tail_prev

    def popleft(self) -> DLLNode | None:
        node: DLLNode | None = self.head.later
        # An empty list points head.later at the tail sentinel. Removing that
        # sentinel used to break the list and evict a nonsense key.
        if node is None or node is self.tail:
            return None

        self.remove(node)
        return node

    def remove(self, node: DLLNode | None) -> None:
        if node is None:
            return

        node_prev: DLLNode | None = node.previous
        node_later: DLLNode | None = node.later

        assert node_prev and node_later

        node_prev.later = node_later
        node_later.previous = node_prev

        node.later = None
        node.previous = None

    def __bool__(self) -> bool:
        return self.head.later != self.tail
