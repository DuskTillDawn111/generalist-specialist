"""Request-scoped affine-module metadata for an USHER-inspired estimator.

No torch import, tensor value read, device synchronization or inference rewrite.
Only explicitly identifiable leaf affine modules are covered. Logical tensor
sizes and declared GEMM FLOPs are features, never measured memory traffic/peak.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time
import weakref

_ACTIVE = weakref.WeakKeyDictionary()


def tensor_metadata(value):
    """Read tensor metadata without transferring/inspecting its contents."""
    shape = getattr(value, 'shape', None)
    element_size = getattr(value, 'element_size', None)
    if shape is None or not callable(element_size):
        return None
    shape = [int(x) for x in shape]
    size = int(element_size())
    if any(x < 0 for x in shape) or size <= 0:
        raise ValueError('Invalid tensor shape/element size')
    return dict(shape=shape, dtype=str(getattr(value, 'dtype', 'UNKNOWN')),
                element_size_bytes=size, logical_bytes=math.prod(shape)*size)


def affine_declaration(module):
    """Accept declared leaf linear maps; wrapper parents are not double-counted."""
    if list(module.children()):
        return None
    dimensions = None
    for left, right in (('in_features', 'out_features'), ('infeatures', 'outfeatures')):
        a, b = getattr(module, left, None), getattr(module, right, None)
        if type(a) is int and type(b) is int and a > 0 and b > 0:
            dimensions = (a, b)
            break
    if dimensions is None:
        return None
    # Shape attributes alone on an arbitrary module do not establish affine
    # semantics. Recognize the actual linear implementation class hierarchy.
    names = [cls.__name__.lower() for cls in type(module).__mro__]
    supported = {'linear', 'linear8bitlt', 'linear4bit', 'quantlinear'}
    if not supported.intersection(names):
        return None
    a, b = dimensions
    return dict(operator_kind='linear_declared', in_features=a, out_features=b,
                implementation=type(module).__module__+'.'+type(module).__name__,
                declaration_basis='recognized_leaf_linear_class_and_declared_dimensions')


def owned_tensor_metadata(module):
    """Report direct tensors separately, retaining aliases without summing storage.

    Packed quantized weights, scales and zero-points keep their actual dtype and
    shape. Logical tensor bytes are not unique storage bytes, dense/dequantized
    weight size, bandwidth demand or a guarantee about a fused backend.
    """
    rows = []
    for kind, iterator in (('parameter', module.named_parameters(recurse=False)),
                           ('buffer', module.named_buffers(recurse=False))):
        for name, tensor in iterator:
            meta = tensor_metadata(tensor)
            if meta is not None:
                rows.append(dict(name=name, kind=kind, **meta))
    return rows


class OperatorObserver:
    def __init__(self, root, request_id, *, nvtx=None, sink=None,
                 clock=time.perf_counter_ns, max_records=200000):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('Nonempty request ID required')
        if type(max_records) is not int or max_records < 1:
            raise ValueError('Positive bounded observation budget required')
        self.root, self.request_id = root, request_id
        self.nvtx, self.sink, self.clock = nvtx, sink, clock
        self.max_records = max_records
        self.records, self.errors, self.modules, self.stacks = [], [], {}, {}
        self.coverage_truncated = False

    def _error(self, where, exc):
        self.errors.append(dict(where=where, error_type=type(exc).__name__))

    def _emit(self, row, edge):
        if self.sink is not None:
            try:
                self.sink(dict(request_id=self.request_id, segment_id=row['id'],
                               kind='gpu', event=edge, perf_ns=row[edge+'_ns'],
                               nvtx_name=row['nvtx_name'], operator_module=row['module']))
            except Exception as exc:
                self._error('sink', exc)

    def before(self, module, args, kwargs):
        stack = self.stacks.setdefault(id(module), [])
        if len(self.records) >= self.max_records:
            self.coverage_truncated = True
            stack.append(None)
            return None
        name, declaration, tensors = self.modules[id(module)]
        row = dict(id=f'affine_{len(self.records):07d}', module=name, **declaration,
                   direct_tensors=tensors, input=None, output=None, dense_gemm_flops=None,
                   resource_measurements=None,
                   scope='module host call range; actual kernels require CUPTI association')
        # Linear.forward accepts its first tensor by position or the input kwarg.
        value = args[0] if args else kwargs.get('input')
        try:
            row['input'] = tensor_metadata(value)
            if row['input'] is not None:
                shape = row['input']['shape']
                if shape and shape[-1] == declaration['in_features']:
                    row['dense_gemm_flops'] = 2*math.prod(shape[:-1])*declaration['in_features']*declaration['out_features']
                else:
                    row['unsupported_reason'] = 'INPUT_LAST_DIMENSION_DOES_NOT_MATCH_DECLARATION'
        except Exception as exc:
            self._error('input_metadata', exc)
        row['nvtx_name'] = f"GS_STAGE:{self.request_id}:gpu:{row['id']}"
        pushed = False
        if self.nvtx is not None:
            try:
                self.nvtx.range_push(row['nvtx_name'])
                pushed = True
            except Exception as exc:
                self._error('nvtx_push', exc)
        row['start_ns'] = self.clock()
        self.records.append(row)
        stack.append((row, pushed))
        self._emit(row, 'start')
        return None

    def after(self, module, args, kwargs, output):
        stack = self.stacks.get(id(module), [])
        if not stack:
            self.errors.append(dict(where='unpaired_forward_end', error_type='UnpairedBoundary'))
            return None
        item = stack.pop()
        if item is None:
            return None
        row, pushed = item
        # Timestamp precedes post-hook metadata extraction, mirroring pre-hook.
        row['end_ns'] = self.clock()
        try:
            row['output'] = tensor_metadata(output)
        except Exception as exc:
            self._error('output_metadata', exc)
        row['end_basis'] = 'always_call_forward_hook'
        self._emit(row, 'end')
        if pushed:
            try:
                self.nvtx.range_pop()
            except Exception as exc:
                self._error('nvtx_pop', exc)
        return None  # Never replace output.

    def document(self):
        return dict(schema='gsdeploy.affine_operator_observations.v2', request_id=self.request_id,
                    clock='host_perf_counter_ns', modules=len(self.modules), records=self.records,
                    instrumentation_errors=self.errors, coverage_truncated=self.coverage_truncated,
                    graph_coverage='recognized leaf affine calls only; attention/functional/fused operations may be absent',
                    flops_basis='declared dense GEMM arithmetic; excludes bias, dequantization, LoRA wrapper and backend overhead',
                    memory_basis='logical tensor metadata; not unique storage, memory traffic or intermediate peak',
                    gpu_synchronization_added=False, tensor_values_read=False,
                    native_outputs_modified=False, training_target_available=False)


@contextmanager
def observe_affine_operators(root, request_id, *, nvtx=None, sink=None,
                             clock=time.perf_counter_ns, max_records=200000):
    """Install/remove observer hooks; no persistent graph/backend modifications."""
    if root in _ACTIVE:
        raise ValueError('Root already has an active affine observer')
    observer = OperatorObserver(root, request_id, nvtx=nvtx, sink=sink,
                                clock=clock, max_records=max_records)
    _ACTIVE[root] = observer
    handles = []
    try:
        for name, module in root.named_modules():
            declaration = affine_declaration(module)
            if declaration is None:
                continue
            tensors = owned_tensor_metadata(module)
            observer.modules[id(module)] = (name, declaration, tensors)
            handles.append(module.register_forward_pre_hook(observer.before, with_kwargs=True))
            handles.append(module.register_forward_hook(observer.after, with_kwargs=True, always_call=True))
        yield observer
    finally:
        for handle in reversed(handles):
            handle.remove()
        # Normal/exception forwards use always_call. Missing end hooks mean the
        # observation is incomplete: never fabricate a kernel/host end boundary.
        if any(stack for stack in observer.stacks.values()):
            observer.errors.append(dict(where='context_exit', error_type='UnclosedForwardBoundary'))
        _ACTIVE.pop(root, None)
