"""Separate strict deterministic runtime pilot; historical workers stay unchanged."""
import os
import sys


def configure(torch):
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if workspace != ':4096:8':
        raise ValueError('Deterministic pilot requires explicit CUBLAS_WORKSPACE_CONFIG=:4096:8')
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    source = sys.argv[sys.argv.index('--source-root')+1]
    sys.path.insert(0, source)
    import torch
    configure(torch)
    mode = os.environ.get('GS_DETERMINISTIC_OBSERVER')
    if mode == 'complete':
        import complete_observed_worker as worker
    elif mode == 'semantic':
        import semantic_observed_worker as worker
    else:
        raise ValueError('Explicit complete/semantic deterministic pilot observer required')
    worker.main()


if __name__ == '__main__':
    main()
