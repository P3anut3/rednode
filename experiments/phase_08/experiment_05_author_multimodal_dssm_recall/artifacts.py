"""Atomic products, immutable completion contracts and resource guardrails."""
import contextlib
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import psutil
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    def finite(v):
        if isinstance(v, dict):
            return {k: finite(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [finite(x) for x in v]
        if isinstance(v, (float, np.floating)) and not np.isfinite(v):
            return None
        return v
    tmp.write_text(json.dumps(finite(value), ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_parquet(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def complete(folder, files, extra=None):
    folder = Path(folder)
    atomic_json(folder / 'complete.json', {
        'complete': True, 'test_opened': False,
        'files': {str(Path(p).relative_to(folder)): sha256(p) for p in files},
        **(extra or {}),
    })


def verified(folder):
    folder = Path(folder)
    marker = json.loads((folder / 'complete.json').read_text())
    if not marker.get('complete') or marker.get('test_opened') is not False:
        raise RuntimeError(f'invalid marker: {folder}')
    for relative, expected in marker['files'].items():
        path = folder / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'artifact changed: {path}')
    return marker


@contextlib.contextmanager
def stage_lock(path):
    """Fail closed on stale locks; user must inspect PID rather than overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, json.dumps({'pid': os.getpid(), 'started': time.time()}).encode())
    os.close(fd)
    try:
        yield
    finally:
        path.unlink()


class Guard:
    def __init__(self, seconds, output, max_rss_gib=16, min_available_gib=8,
                 disk_need_gib=0):
        self.deadline = time.monotonic() + seconds
        self.output = Path(output)
        self.max_rss = max_rss_gib * 2**30
        self.min_available = min_available_gib * 2**30
        self.disk_need = disk_need_gib * 2**30
        self.peak_rss = 0
        self.check()

    def check(self):
        proc = psutil.Process()
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        self.peak_rss = max(self.peak_rss, rss)
        if time.monotonic() > self.deadline:
            raise TimeoutError('stage deadline exceeded; no completion marker')
        if rss > self.max_rss or psutil.virtual_memory().available < self.min_available:
            raise MemoryError(f'resource guard: process-tree RSS={rss / 2**30:.2f} GiB')
        parent = self.output
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < self.disk_need + 2 * 2**30:
            raise OSError('insufficient disk safety margin')


def cuda_preflight(device):
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU inference/retrieval fallback prohibited')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    # Strict FP32 retrieval, no silent TF32 approximation.
    torch.backends.cuda.matmul.allow_tf32 = False
    return {'device': device, 'gpu': torch.cuda.get_device_name(device)}
