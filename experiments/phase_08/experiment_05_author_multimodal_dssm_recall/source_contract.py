"""Load only audited pure AST nodes: never execute author's filesystem/CLI globals."""
import ast
import json

import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import sha256
from .config import COMMIT, EXPERIMENT

NODES = {'DSSMModel', '_fuse_history_modal_vecs', 'summarize_history_sequence',
         'compute_inbatch_loss'}


def author_namespace():
    manifest = json.loads((EXPERIMENT / 'source_manifest.json').read_text())
    if manifest['commit'] != COMMIT:
        raise RuntimeError('source commit mismatch')
    for name, digest in manifest['files'].items():
        if sha256(EXPERIMENT / 'source' / name) != digest:
            raise RuntimeError(f'locked author source changed: {name}')
    path = EXPERIMENT / 'source/dssm_trainer.py'
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                and n.name in NODES]
    if len(selected) != len(NODES):
        raise RuntimeError('author pure nodes not found')
    env = dict(torch=torch, nn=nn, F=F, EMB_DIM=768, SEQ_TOWER_DIM=128,
               HIDDEN_DIM=128, HIST_TEXT_WEIGHT=.72, HIST_IMG_WEIGHT=.28,
               HIST_RECENT_WINDOW=5, INBATCH_TEMPERATURE=.07)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), env)
    return env
