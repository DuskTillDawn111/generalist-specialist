"""Isolated candidate only; never installed in frozen or active workers.

Replace only MiniCPM chat's TTS-bound index scan, retaining its original CUDA
outputs['full_sequences'] tensor. CPU tests cannot qualify CUDA equivalence,
latency, async dependencies, audio behavior, or a new execution identity.
"""


def scan_integer_ids(ids, tts_bos_token, tts_eos_token, teacher_forcing=False):
    bos_indices, eos_indices = [], []
    for index, value in enumerate(ids):
        if value == tts_bos_token:
            bos_indices.append(index + 1)
        elif value == tts_eos_token:
            if teacher_forcing and index == len(ids) - 1:
                continue
            eos_indices.append(index)
    return (bos_indices[-1] if bos_indices else -1,
            eos_indices[-1] if eos_indices else None)


def tts_bounds_bulk(full_sequence_row, tts_bos_token, tts_eos_token, teacher_forcing=False):
    """One blocking bulk DtoH; no mutation or relocation of source tensors."""
    ids = full_sequence_row.detach().cpu().tolist()
    return scan_integer_ids(ids, tts_bos_token, tts_eos_token, teacher_forcing)
