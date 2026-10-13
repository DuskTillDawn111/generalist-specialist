"""Durable exact-configuration cache. No rounding of routing or failed-run retries."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path

from .result import jsonable


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


class RuntimeEvaluationCache:
    def __init__(self, directory, max_unique=6, repetitions=3):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'runtime_cache_manifest.json'
        self.max_unique, self.repetitions = max_unique, repetitions

    @contextmanager
    def locked(self):
        with (self.directory / '.runtime_cache.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                self.data = (json.loads(self.path.read_text()) if self.path.exists() else
                             dict(schema_version=1, max_unique_runtime_evaluations=self.max_unique,
                                  repetitions_per_evaluation=self.repetitions, cache_hits=0,
                                  configurations={}, run_attempts=[]))
                if (self.data['max_unique_runtime_evaluations'] != self.max_unique or
                        self.data['repetitions_per_evaluation'] != self.repetitions):
                    raise ValueError('Cannot change a frozen cache budget')
                yield self
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def flush(self):
        save_json(self.path, self.data)

    def reserve(self, identity):
        key = digest(identity)
        entries = self.data['configurations']
        if key not in entries:
            if len(entries) >= self.max_unique:
                raise RuntimeError('Unique runtime evaluation budget exhausted; no extra measurement permitted')
            entries[key] = dict(identity=identity, status='reserved', repetitions=[], aggregate=None)
            self.flush()
        elif entries[key]['identity'] != jsonable(identity):
            raise RuntimeError('Cache identity collision')
        return key, entries[key]
