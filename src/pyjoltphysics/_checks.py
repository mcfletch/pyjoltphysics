"""Conversions of integer inputs that refuse a value their type cannot hold.

NumPy casts -1 to 4294967295 and 1.5 to 1 when it converts to an unsigned
type; a uid, group, mask, filter or material index converted that way names
something other than what the caller wrote.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def whole_numbers(value: npt.ArrayLike, dtype: type[np.integer], name: str) -> np.ndarray:
    """Return ``value`` as an array of ``dtype``; ValueError where a value is not a whole number it holds."""
    array = np.asarray(value)
    info = np.iinfo(dtype)
    if array.size == 0:
        return array.astype(dtype)
    if array.dtype.kind == 'b':
        return array.astype(dtype)
    if array.dtype.kind == 'f':
        if not (np.isfinite(array).all() and (array == np.floor(array)).all()):
            raise ValueError('%s holds a value that is not a whole number' % name)
    elif array.dtype.kind not in 'iu':
        raise ValueError('%s holds %s where whole numbers are needed' % (name, array.dtype))
    low, high = array.min(), array.max()
    if low < info.min or high > info.max:
        raise ValueError(
            '%s holds %s, outside %d to %d'
            % (name, low if low < info.min else high, info.min, info.max)
        )
    return array.astype(dtype)


def whole_number(value: int, low: int, high: int, name: str) -> int:
    """Return ``value`` as an int; ValueError where it is not a whole number from ``low`` to ``high``."""
    if isinstance(value, float) and not value.is_integer():
        raise ValueError('%s is a whole number, not %r' % (name, value))
    number = int(value)
    if not low <= number <= high:
        raise ValueError('%s is from %d to %d, not %r' % (name, low, high, value))
    return number
