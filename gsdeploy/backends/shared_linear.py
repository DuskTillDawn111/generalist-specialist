"""CPU reference for exact shared linear blocks; not a model/GPU backend.

Convention: X[batch, in] @ W[in, out]. Search finds the largest *contiguous,
axis-aligned rectangular* byte-identical block at possibly different origins
in two weights. It does not search arbitrary row/column subsets or approximate
matches. The exhaustive alignment search has a declared cell-comparison cap.
Float arithmetic is algebraically equivalent, not necessarily bitwise identical
to a single independent GEMM because residual accumulation changes reduction
order. Only finite float32/float64 arrays with identical dtype are supported.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


DEFAULT_MAX_SEARCH_CELLS = 2_000_000


class SearchBudgetExceeded(ValueError):
    """No search result was produced; no partial result is called maximal."""


def _matrix(value, name):
    value = np.asarray(value)
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError(f'{name} must be a nonempty 2D matrix')
    if value.dtype.kind != 'f' or value.dtype.itemsize not in (4, 8):
        raise ValueError(f'{name} requires float32 or float64')
    if not np.isfinite(value).all():
        raise ValueError(f'{name} must contain finite values')
    return value


def weight_identity(weight):
    """Hash logical C-order bytes *and* exact dtype/shape, including byte order."""
    weight = _matrix(weight, 'weight')
    header = json.dumps(dict(dtype=weight.dtype.str, shape=list(weight.shape)),
                        sort_keys=True, separators=(',', ':')).encode()
    digest = hashlib.sha256(header + b'\0' + np.ascontiguousarray(weight).tobytes()).hexdigest()
    return dict(sha256=digest, dtype=weight.dtype.str, shape=list(weight.shape))


@dataclass(frozen=True)
class CommonRectangle:
    left_origin: tuple[int, int]
    right_origin: tuple[int, int]
    shape: tuple[int, int]

    @property
    def area(self):
        return self.shape[0] * self.shape[1]


@dataclass(frozen=True)
class SearchResult:
    status: str
    rectangle: CommonRectangle | None
    comparisons: int
    maximum_search_cells: int
    scope: str = 'maximum_area_contiguous_axis_aligned_byte_equal_rectangle'


def find_max_common_contiguous(left, right, *, max_search_cells=DEFAULT_MAX_SEARCH_CELLS):
    """Exact maximum within the stated search domain, deterministic on ties.

    Ties use (left row, left col, right row, right col, height, width).
    Every relative row/column alignment is examined, then a histogram/stack
    largest-rectangle algorithm scans its equality grid. Total equality cells
    equal left.size * right.size. Exact full-weight identity is a fast path.
    Exceeding the cap raises before enumeration and returns no candidate.
    """
    left, right = _matrix(left, 'left'), _matrix(right, 'right')
    if left.dtype != right.dtype:
        raise ValueError('Shared GEMM requires identical weight dtype/byte order')
    if type(max_search_cells) is not int or max_search_cells < 1:
        raise ValueError('max_search_cells must be a positive integer')
    li, ri = weight_identity(left), weight_identity(right)
    if li == ri and np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes():
        rect = CommonRectangle((0, 0), (0, 0), left.shape)
        return SearchResult('FULL_WEIGHT_MATCH', rect, 0, max_search_cells)
    comparisons = int(left.size) * int(right.size)
    if comparisons > max_search_cells:
        raise SearchBudgetExceeded(f'Exact contiguous search needs {comparisons} equality cells; '
                                   f'cap={max_search_cells}; no partial maximum returned')
    byte_dtype = np.dtype((np.void, left.dtype.itemsize))
    a = np.ascontiguousarray(left).view(byte_dtype).reshape(left.shape)
    b = np.ascontiguousarray(right).view(byte_dtype).reshape(right.shape)
    best, best_key = None, None
    for dr in range(1-right.shape[0], left.shape[0]):
        ar, br = max(dr, 0), max(-dr, 0)
        height = min(left.shape[0]-ar, right.shape[0]-br)
        for dc in range(1-right.shape[1], left.shape[1]):
            ac, bc = max(dc, 0), max(-dc, 0)
            width = min(left.shape[1]-ac, right.shape[1]-bc)
            equal = a[ar:ar+height, ac:ac+width] == b[br:br+height, bc:bc+width]
            heights = [0] * width
            for row in range(height):
                for col in range(width):
                    heights[col] = heights[col]+1 if equal[row, col] else 0
                stack = []
                for col in range(width+1):
                    h = heights[col] if col < width else 0
                    start = col
                    while stack and stack[-1][1] > h:
                        start, old_h = stack.pop()
                        rect = CommonRectangle((ar+row-old_h+1, ac+start),
                            (br+row-old_h+1, bc+start), (old_h, col-start))
                        key = (-rect.area, *rect.left_origin, *rect.right_origin, *rect.shape)
                        if best_key is None or key < best_key:
                            best, best_key = rect, key
                    if h and (not stack or stack[-1][1] < h):
                        stack.append((start, h))
    return SearchResult('FOUND' if best else 'NO_COMMON_BLOCK', best, comparisons, max_search_cells)


@dataclass(frozen=True)
class ResidualBlock:
    origin: tuple[int, int]
    weight: np.ndarray


def _copy_readonly(value):
    result = np.array(value, copy=True, order='C')
    result.flags.writeable = False
    return result


def _residuals(weight, origin, shape):
    if origin is None:
        return (ResidualBlock((0, 0), _copy_readonly(weight)),)
    row, col = origin
    h, w = shape
    # Disjoint partition of the complement; no zero-mask full GEMM.
    regions = [(0, row, 0, weight.shape[1]), (row+h, weight.shape[0], 0, weight.shape[1]),
               (row, row+h, 0, col), (row, row+h, col+w, weight.shape[1])]
    return tuple(ResidualBlock((r0, c0), _copy_readonly(weight[r0:r1, c0:c1]))
                 for r0, r1, c0, c1 in regions if r1 > r0 and c1 > c0)


@dataclass(frozen=True)
class SharedLinearPlan:
    search: SearchResult
    identities: tuple[dict, dict]
    common_weight: np.ndarray | None
    residuals: tuple[tuple[ResidualBlock, ...], tuple[ResidualBlock, ...]]

    def reconstruct(self):
        """CPU audit/debug reconstruction, not part of the execution fast path."""
        weights = []
        rect = self.search.rectangle
        for branch, identity in enumerate(self.identities):
            value = np.zeros(identity['shape'], dtype=np.dtype(identity['dtype']))
            for block in self.residuals[branch]:
                r, c = block.origin; h, w = block.weight.shape
                value[r:r+h, c:c+w] = block.weight
            if rect is not None:
                r, c = (rect.left_origin, rect.right_origin)[branch]; h, w = rect.shape
                value[r:r+h, c:c+w] = self.common_weight
            weights.append(value)
        return tuple(weights)

    def audit(self):
        """Detect altered packed weights before use; not a GPU performance claim."""
        if tuple(weight_identity(w) for w in self.reconstruct()) != self.identities:
            raise ValueError('Packed weight identity changed')
        return True

    def execute(self, left_input, right_input):
        """Return ((left_output, right_output), explicit reference cost report).

        Concatenation merges batch rows only for the common input-column slice.
        Both branches may have different batch, input and output widths.
        No bias, activation, quantization, model generate or runtime integration.
        """
        self.audit()
        inputs = (_matrix(left_input, 'left_input'), _matrix(right_input, 'right_input'))
        for value, identity in zip(inputs, self.identities):
            if value.dtype.str != identity['dtype'] or value.shape[1] != identity['shape'][0]:
                raise ValueError('Input dtype/width must match its exact weight identity')
        outputs = tuple(np.zeros((x.shape[0], identity['shape'][1]), dtype=x.dtype)
                        for x, identity in zip(inputs, self.identities))
        rect = self.search.rectangle
        common_macs = concat_bytes = common_output_bytes = 0
        calls = additions = residual_macs = residual_temp_bytes = 0
        if rect is not None:
            h, w = rect.shape
            pieces = [x[:, origin[0]:origin[0]+h]
                      for x, origin in zip(inputs, (rect.left_origin, rect.right_origin))]
            joined = np.concatenate(pieces, axis=0)
            product = joined @ self.common_weight
            split = inputs[0].shape[0]
            for output, origin, part in zip(outputs, (rect.left_origin, rect.right_origin),
                                           (product[:split], product[split:])):
                output[:, origin[1]:origin[1]+w] += part
                additions += part.size
            calls += 1
            common_macs = sum(x.shape[0] for x in inputs) * rect.area
            concat_bytes, common_output_bytes = joined.nbytes, product.nbytes
        for x, output, blocks in zip(inputs, outputs, self.residuals):
            for block in blocks:
                r, c = block.origin; h, w = block.weight.shape
                part = x[:, r:r+h] @ block.weight
                output[:, c:c+w] += part
                residual_macs += x.shape[0] * h * w
                residual_temp_bytes = max(residual_temp_bytes, part.nbytes)
                additions += part.size
                calls += 1
        itemsize = inputs[0].dtype.itemsize
        independent_bytes = sum(int(np.prod(i['shape'])) for i in self.identities) * itemsize
        packed_bytes = (0 if self.common_weight is None else self.common_weight.nbytes) + sum(
            b.weight.nbytes for blocks in self.residuals for b in blocks)
        report = dict(schema='gsdeploy.cpu_shared_linear.v2', GPU_started=False,
            deployed_model_backend=False, search_status=self.search.status,
            search_scope=self.search.scope, search_comparison_cells=self.search.comparisons,
            original_weight_identities=self.identities, common_area=0 if rect is None else rect.area,
            independent_macs=sum(x.shape[0]*int(np.prod(i['shape'])) for x, i in zip(inputs, self.identities)),
            common_macs=common_macs, residual_macs=residual_macs,
            executed_macs=common_macs+residual_macs, independent_gemm_calls=2,
            executed_gemm_calls=calls, residual_gemm_calls=sum(map(len, self.residuals)),
            output_accumulation_elements=additions, concat_copy_bytes=concat_bytes,
            common_product_bytes=common_output_bytes, split_copy_bytes=0,
            largest_residual_product_bytes=residual_temp_bytes,
            output_buffer_bytes=sum(y.nbytes for y in outputs),
            independent_weight_bytes=independent_bytes, packed_weight_bytes=packed_bytes,
            weight_pack_copy_bytes=packed_bytes,
            common_weight_bytes=0 if self.common_weight is None else self.common_weight.nbytes,
            residual_weight_bytes=sum(b.weight.nbytes for blocks in self.residuals for b in blocks),
            weight_storage_bytes_saved=independent_bytes-packed_bytes,
            audit_reconstruction_bytes=independent_bytes,
            cost_scope='Logical array payloads/MACs; excludes BLAS workspaces, allocator/RSS, metadata, timing and hardware cache counters',
            arithmetic_scope='Algebraically equivalent; floating reduction order may differ; no FLOP reduction claimed',
            benefit_claim=False)
        return outputs, report


def prepare(left_weight, right_weight, *, max_search_cells=DEFAULT_MAX_SEARCH_CELLS):
    """Pack one exact common block plus both disjoint residuals into owned copies."""
    left, right = _matrix(left_weight, 'left_weight'), _matrix(right_weight, 'right_weight')
    search = find_max_common_contiguous(left, right, max_search_cells=max_search_cells)
    rect = search.rectangle
    if rect is None:
        common = None; origins = (None, None); shape = (0, 0)
    else:
        origins, shape = (rect.left_origin, rect.right_origin), rect.shape
        r, c = origins[0]; h, w = shape
        common = _copy_readonly(left[r:r+h, c:c+w])
    plan = SharedLinearPlan(search, (weight_identity(left), weight_identity(right)), common,
        tuple(_residuals(weight, origin, shape) for weight, origin in zip((left, right), origins)))
    plan.audit()
    return plan
