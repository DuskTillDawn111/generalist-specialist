"""Opt-in nested forward observations; original complete worker stays unchanged.

CUDA events measure stream elapsed time (including gaps and interference),
not pure kernel time. No per-forward synchronization is introduced. This is a
new execution context requiring matched overhead checks and fresh templates.
"""
from functools import wraps
import sys
import threading
import time

CURRENT = None


def sequence_length(args, kwargs):
    value = kwargs.get('input_ids')
    if value is None:
        value = kwargs.get('inputs_embeds')
    if value is None and args:
        value = args[0]
    shape = getattr(value, 'shape', ())
    return int(shape[1]) if len(shape) >= 2 else None


class DetailRecorder:
    def __init__(self, torch, request_id):
        self.torch, self.rid = torch, request_id
        self.records, self.events, self.ordinals = [], [], {}
        self.lock = threading.RLock()

    def call(self, original, role, args, kwargs):
        with self.lock:
            if len(self.records) >= 4096:
                raise RuntimeError('Semantic event budget exceeded; no silent truncation')
            ordinal = self.ordinals.get(role, 0)
            self.ordinals[role] = ordinal + 1
            length = sequence_length(args, kwargs) if role == 'language' else None
            label = role
            if role == 'language':
                label = ('language_prefill' if ordinal == 0 else 'language_decode') if length == 1 else (
                    'language_multi_token' if length is not None and length > 1 else 'language_unknown_shape')
            stream = self.torch.cuda.current_stream()
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            record = dict(request_id=self.rid, role=label, component=role, ordinal=ordinal,
                          sequence_length=length, thread_id=threading.get_ident(),
                          stream_id=int(stream.cuda_stream), device=int(stream.device.index),
                          host_start_ns=time.perf_counter_ns())
            self.records.append(record); self.events.append((start, end, record))
            start.record(stream)
            self.torch.cuda.nvtx.range_push(f'GS_DETAIL:{self.rid}:{label}:{ordinal}')
        try:
            return original(*args, **kwargs)
        finally:
            end.record(stream)
            self.torch.cuda.nvtx.range_pop()
            record['host_end_ns'] = time.perf_counter_ns()
            record['end_current_stream_id'] = int(self.torch.cuda.current_stream().cuda_stream)

    def finish(self):
        # The complete worker's original final sync (or native OCR sync) has
        # already happened. Query does not force a new device synchronization.
        origin = self.events[0][0] if self.events else None
        for start, end, record in self.events:
            known = bool(start.query() and end.query() and origin.query())
            record['device_event_complete'] = known
            record['device_elapsed_ms'] = float(start.elapsed_time(end)) if known else None
            record['device_start_from_first_event_ms'] = float(origin.elapsed_time(start)) if known else None
            record['device_end_from_first_event_ms'] = float(origin.elapsed_time(end)) if known else None
            record['timing_scope'] = 'CUDA event stream interval; contains scheduling gaps, not a kernel union'


def observe_method(owner, method, role):
    original = getattr(owner, method)
    @wraps(original)
    def wrapped(*args, **kwargs):
        recorder = CURRENT
        return original(*args, **kwargs) if recorder is None else recorder.call(original, role, args, kwargs)
    setattr(owner, method, wrapped)


def attach_result(result):
    if CURRENT is not None:
        result['semantic_phase_schema'] = 'gsdeploy.semantic_native_observation.v1'
        result['semantic_phase_events'] = CURRENT.records
        result['semantic_phase_scope'] = ('nested host forward calls and asynchronous CUDA events; '
                                         'new observer context; not pure GPU phases or prediction templates')
    return result


def install_compact(adapter):
    executor = adapter.executor
    observe_method(executor, 'prepare', 'prepare_and_media')
    if executor.family in {'minicpm', 'minicpm45'}:
        from gsdeploy.backends.compact_native import minicpm_native_model
        inner = minicpm_native_model(adapter.model)
        observe_method(inner.llm, 'forward', 'language')
        for name, role in [('vpm','vision_encoder'),('resampler','vision_resampler'),
                           ('apm','audio_encoder'),('audio_projection_layer','audio_projection')]:
            if getattr(inner, name, None) is not None:
                observe_method(getattr(inner, name), 'forward', role)
    elif executor.family == 'siglip':
        observe_method(adapter.model, 'get_image_features', 'image_features')
    else:
        # Unsupported components remain explicit; do not invent phase labels.
        observe_method(adapter.model, 'forward', 'model_forward')
    original = executor.infer
    @wraps(original)
    def infer(*args, **kwargs):
        return attach_result(original(*args, **kwargs))
    executor.infer = infer


def main():
    global CURRENT
    source = sys.argv[sys.argv.index('--source-root')+1]
    sys.path.insert(0, source)
    import complete_observed_worker as complete
    from gsdeploy.backends import compact_native
    original_adapter = compact_native.ProfileAdapter
    class CompactAdapter(original_adapter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs); install_compact(self)
    compact_native.ProfileAdapter = CompactAdapter
    original_module = complete.module
    def module(name, path):
        value = original_module(name, path)
        original = value.Adapter
        class OCRAdapter(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                for component, role in [('det','ocr_detection'),('rec','ocr_recognition')]:
                    observe_method(self.models[component][0], 'forward', role)
            def run(self, sample):
                return attach_result(super().run(sample))
        value.Adapter = OCRAdapter
        return value
    complete.module = module
    class CompleteDetailObserver(complete.CompleteObserver):
        def __init__(self, torch, request_id, start_ns):
            global CURRENT
            CURRENT = DetailRecorder(torch, request_id)
            self.detail = CURRENT
            super().__init__(torch, request_id, start_ns)
        def finish(self):
            global CURRENT
            self.detail.finish()
            result = super().finish()
            CURRENT = None
            return result
    complete.CompleteObserver = CompleteDetailObserver
    complete.main()


if __name__ == '__main__':main()
