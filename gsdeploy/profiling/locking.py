"""One repository-wide GPU lease, independent of result-directory names."""
from contextlib import contextmanager
import fcntl
from pathlib import Path


@contextmanager
def gpu_lease():
    path = Path(__file__).resolve().parents[2] / 'artifacts/.gpu-experiment.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
