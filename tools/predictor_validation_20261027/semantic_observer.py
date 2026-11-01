"""Opt-in language-forward observations, without changing generation semantics.

Use one context per request, around executor.infer(). The first language forward
is labelled prefill by position, subsequent calls decode. Sequence length and
cache presence are recorded separately: this does not assert cache use or one
generated token per call. No tensor values, labels or outputs are inspected.
Hooks never synchronize CUDA. Forward wall ranges require CUPTI attribution to
establish actual GPU activity and must not be treated as pure kernel durations.
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import time
import weakref

SCHEMA = "gsdeploy.semantic_forward_observation.v1"
_ACTIVE = weakref.WeakKeyDictionary()


def language_module(executor):
    if executor.family == "chatqa":
        model, seen = executor.model, set()
        # Frozen ChatQA loading applies PeftModel.from_pretrained after the
        # LlamaForCausalLM load. PeftModelForCausalLM.generate delegates through
        # the LoRA tuner to base_model.model.generate, bypassing PeftModel's
        # __call__ and therefore hooks on the outer wrapper. Observe the actual
        # causal-LM module, retaining all LoRA layer hooks and active adapters.
        while model.__class__.__module__.startswith("peft."):
            if id(model) in seen:
                raise ValueError("Cyclic PEFT language module ownership")
            seen.add(id(model))
            getter = getattr(model, "get_base_model", None)
            if callable(getter):
                candidate = getter()
            else:
                base = getattr(model, "base_model", None)
                candidate = getattr(base, "model", None) if base is not None else None
            if candidate is None or candidate is model:
                raise ValueError("Cannot resolve PEFT causal-LM generation owner")
            model = candidate
        # A forwarded generate method may be bound to an actual causal LM.
        # Never unwrap arbitrary .model attributes (e.g. Llama's decoder or
        # quantization proxies): that loses the LM head/generation boundary.
        generate = getattr(model, "generate", None)
        bound_owner = getattr(generate, "__self__", model)
        if bound_owner is not model:
            if not callable(getattr(bound_owner, "forward", None)):
                raise ValueError("Forwarded generate owner has no forward")
            model = bound_owner
        return model
    if executor.family == "minicpm45":
        model = executor.model
        # Same ownership rule as frozen compact_memory.minicpm_native_model;
        # assignment to a GPTQ proxy is deliberately avoided.
        for _ in range(2):
            owned = vars(model)
            if "llm" in owned or "llm" in owned.get("_modules", {}):
                return model.llm
            model = getattr(model, "model", None)
            if model is None:
                break
        raise ValueError("Cannot resolve native MiniCPM45 language module")
    raise ValueError("Semantic observer supports only ChatQA and MiniCPM45")


class SemanticObserver:
    def __init__(self, owner, request_id, *, nvtx=None, sink=None, clock=time.perf_counter_ns):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Nonempty request ID required")
        self.owner, self.request_id = owner, request_id
        self.nvtx, self.sink, self.clock = nvtx, sink, clock
        self.records, self.errors, self._stack = [], [], []
        # Signature is only read. Crucially, forward/generate are never replaced.
        parameters = list(inspect.signature(owner.forward).parameters.values())
        positional = [p.name for p in parameters if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        self._positions = {name: positional.index(name) for name in ("input_ids", "inputs_embeds", "past_key_values") if name in positional}

    def _input(self, args, kwargs, name):
        if name in kwargs:
            return kwargs[name]
        index = self._positions.get(name)
        return args[index] if index is not None and index < len(args) else None

    def _emit(self, row, edge):
        if self.sink is None:
            return
        try:
            self.sink({"request_id": self.request_id, "event": edge,
                       "segment_id": row["id"], "kind": "gpu", "semantic_phase": row["semantic_phase"],
                       "forward_index": row["forward_index"], "perf_ns": row[edge + "_ns"],
                       "nvtx_name": row["nvtx_name"], "input_sequence_length": row["input_sequence_length"],
                       "input_batch_size": row["input_batch_size"]})
        except Exception as exc:
            # An optional telemetry sink must not alter inference control flow.
            self.errors.append({"where": "sink", "type": type(exc).__name__})

    def before(self, module, args, kwargs):
        index = len(self.records)
        semantic = "prefill" if index == 0 else "decode"
        row = {"id": f"{semantic}_{index:04d}", "kind": "gpu", "semantic_phase": semantic,
               "classification_basis": "first_language_forward_in_request" if index == 0 else "subsequent_language_forward_in_request",
               "forward_index": index, "input_sequence_length": None, "input_batch_size": None,
               "input_token_positions": None, "input_kind": None,
               "past_key_values_present": self._input(args, kwargs, "past_key_values") is not None,
               "phase_scope": "language forward Python call wall range; asynchronous CUDA requires CUPTI",
               "demand": None, "synchronization_added": False}
        for name, rank, sequence_axis in (("input_ids", 2, -1), ("inputs_embeds", 3, -2)):
            value = self._input(args, kwargs, name)
            if value is not None:
                shape = getattr(value, "shape", None)
                if shape is not None and len(shape) == rank:
                    batch, length = int(shape[0]), int(shape[sequence_axis])
                    row.update(input_kind=name, input_sequence_length=length,
                               input_batch_size=batch, input_token_positions=batch*length)
                    break
        row["nvtx_name"] = f"GS_STAGE:{self.request_id}:gpu:{row['id']}"
        pushed = False
        if self.nvtx is not None:
            try:
                self.nvtx.range_push(row["nvtx_name"])
                pushed = True
            except Exception as exc:
                self.errors.append({"where": "nvtx_push", "type": type(exc).__name__})
        row["start_ns"] = self.clock()
        self.records.append(row)
        self._stack.append((row, pushed))
        self._emit(row, "start")
        return None

    def after(self, module, args, kwargs, output):
        if self._stack:
            self._finish("always_call_forward_hook")
        return None  # Never replace module output.

    def _finish(self, basis):
        row, pushed = self._stack.pop()
        row["end_ns"] = self.clock()
        row["end_basis"] = basis
        self._emit(row, "end")
        if pushed:
            try:
                self.nvtx.range_pop()
            except Exception as exc:
                self.errors.append({"where": "nvtx_pop", "type": type(exc).__name__})

    def document(self):
        return {"schema": SCHEMA, "request_id": self.request_id, "clock": "time.perf_counter_ns",
                "hook_owner_class": self.owner.__class__.__module__ + "." + self.owner.__class__.__name__,
                "records": self.records, "instrumentation_errors": self.errors,
                "token_scope": "tensor-shape positions only; not observed generated token IDs/count",
                "classification_scope": "language-forward position; not proof of KV cache state or GPU execution",
                "gpu_synchronization_added": False}


@contextmanager
def observe_language_forwards(executor, request_id, *, nvtx=None, sink=None, clock=time.perf_counter_ns):
    """Install request-scoped hooks and remove them even if forward raises.

    The owner must not execute concurrent requests. Nested observers on the same
    module are rejected, preventing accidental double attribution. Existing
    model hooks and all generation kwargs/signatures remain unchanged.
    """
    owner = language_module(executor)
    if owner in _ACTIVE:
        raise ValueError("Language module already has an active semantic observer")
    observer = SemanticObserver(owner, request_id, nvtx=nvtx, sink=sink, clock=clock)
    _ACTIVE[owner] = observer
    handles = []
    try:
        handles.append(owner.register_forward_pre_hook(observer.before, with_kwargs=True))
        handles.append(owner.register_forward_hook(observer.after, with_kwargs=True, always_call=True))
        yield observer
    finally:
        while observer._stack:
            observer._finish("context_exit_fallback")
        for handle in reversed(handles):
            handle.remove()
        _ACTIVE.pop(owner, None)
