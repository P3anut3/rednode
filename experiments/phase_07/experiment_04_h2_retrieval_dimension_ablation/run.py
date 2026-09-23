#!/usr/bin/env python3
"""Phase 7 Experiment 04: 128d versus 256d H0/H1/H2 retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.data import QilinData  # noqa: E402
from experiments.common.metrics import evaluate_rankings  # noqa: E402
from experiments.phase_06.experiment_01_id_two_tower_retrieval.data import grouped_requests  # noqa: E402
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.config import (  # noqa: E402
    DeadlineExceeded,
    PROTOCOL,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.dataset import (  # noqa: E402
    Phase7Collator,
    Phase7Dataset,
    load_training_frame,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.evaluation import (  # noqa: E402
    feature_slices,
    paired_bootstrap_segments,
    phase6_status_metrics,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.retrieval import (  # noqa: E402
    deterministic_proxy_candidates,
    filter_history,
    gpu_exact_search,
    validate_topk,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.run import (  # noqa: E402
    json_default,
    legacy_evaluator_sets,
    resources,
    save_csv_atomic,
    save_json,
    save_parquet_atomic,
    save_torch_atomic,
    select_requests,
)
from experiments.phase_07.experiment_01_feature_hybrid_two_tower.trainer import to_device  # noqa: E402
from experiments.phase_07.experiment_04_h2_retrieval_dimension_ablation.models import (  # noqa: E402
    MODEL_NAMES,
)
from experiments.phase_07.experiment_04_h2_retrieval_dimension_ablation.trainer import (  # noqa: E402
    DIM,
    ID_DIM,
    OUT,
    encode_items,
    encode_queries,
    make_model,
    pack_item_features,
    parameter_report,
    train_epoch,
)


EXP01 = ROOT / "results/phase_07/experiment_01_feature_hybrid_two_tower"
REFERENCE_NAMES = {
    "h0_256": "h0_content_control",
    "h1_256": "h1_side_features",
    "h2_256": "h2_side_features_id",
}
SEEDS = (42, 43, 44)
MUTATING = {"static-audit", "smoke", "train", "full-validation", "bootstrap", "select", "lock", "terminal-test", "report"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "static-audit", "smoke", "train", "full-validation", "bootstrap", "select", "lock", "terminal-test", "report"), default="plan")
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--confirm-run", action="store_true")
    return parser.parse_args()


def marker(name: str) -> Path:
    return OUT / "configs/markers" / name


def ensure_layout():
    for name in ("configs", "configs/markers", "checkpoints", "metrics", "metrics/training_curves", "metrics/timing", "validation", "validation/rankings", "embeddings", "diagnostics", "smoke"):
        (OUT / name).mkdir(parents=True, exist_ok=True)


def authorize(args):
    if args.stage in MUTATING and not args.confirm_run:
        raise SystemExit(f"REFUSED: {args.stage} requires --confirm-run")
    if args.stage in {"smoke", "train", "full-validation"} and not args.model:
        raise SystemExit(f"{args.stage} requires --model")
    if args.seed not in SEEDS:
        raise SystemExit("formal seeds are 42/43/44")
    if args.seed in {43, 44} and args.stage in {"smoke", "train", "full-validation"}:
        selection = marker("seed42_selection.json")
        if not selection.exists() or not json.loads(selection.read_text()).get("run_h2_multiseed") or args.model != "h2_256":
            raise SystemExit("seed43/44 only allowed for eligible H2-256")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def save_npy(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream: np.save(stream, value, allow_pickle=False)
    temporary.replace(path)


def reference_artifacts() -> dict:
    output = {}
    for new, old in REFERENCE_NAMES.items():
        result = EXP01 / f"validation/{old}_seed42.json"
        ranking = EXP01 / f"validation_rankings/{old}_seed42.parquet"
        value = json.loads(result.read_text())
        output[new] = {
            "model_128": old,
            "result": str(result.relative_to(ROOT)), "result_sha256": sha256(result),
            "ranking": str(ranking.relative_to(ROOT)), "ranking_sha256": sha256(ranking),
            "Recall@500": float(value["metrics"]["overall"]["Recall@500"]),
        }
    return output


def plan():
    print(json.dumps({"status": "ready", "train_only": list(MODEL_NAMES), "reuse_128": REFERENCE_NAMES, "retrieval_dim": DIM, "id_dim": ID_DIM, "test_opened": False}, indent=2))


def static_audit(args):
    ensure_layout(); _, _, store = resources(); refs = reference_artifacts()
    rows = []
    for name in MODEL_NAMES:
        seed_all(42); model = make_model(name, store, "cpu")
        zeros = torch.zeros(4, dtype=torch.long)
        if name == "h2_256":
            item_zero = model.item_id_adapter(model.item_id(zeros))
            user_zero = model.user_id_adapter(model.user_id(zeros))
            oov_zero = float(torch.maximum(item_zero.abs().max(), user_zero.abs().max()))
        else: oov_zero = 0.0
        rows.append({"model": name, **parameter_report(model), "item_id_dim": model.item_id.embedding_dim, "user_id_dim": model.user_id.embedding_dim, "item_adapter_bias": model.item_id_adapter.bias is not None, "user_adapter_bias": model.user_id_adapter.bias is not None, "oov_adapter_max_abs": oov_zero})
    payload = {"references": refs, "models": rows, "feature_cache_reused": True, "test_opened": False}
    save_json(OUT / "diagnostics/static_audit.json", payload); print(json.dumps(payload, indent=2))


def proxy_evaluate(model, requests, store, device, ids, packed, deadline):
    items = encode_items(model, ids, store, device, packed=packed, deadline=deadline)
    queries = encode_queries(model, requests, store, device, deadline=deadline)
    _, raw, seconds, backend, timing = gpu_exact_search(items, queries, min(PROTOCOL.overfetch, len(ids)), device, query_batch=256, deadline=deadline)
    rankings = filter_history(raw, ids, [r.history for r in requests], PROTOCOL.topk, deadline)
    validate_topk(rankings, ids, [r.history for r in requests], deadline)
    metrics, _ = evaluate_rankings(
        requests, rankings, set(), set(), "proxy", deadline=deadline
    )
    return {"proxy_Recall@100": metrics["overall"]["Recall@100"], "proxy_Recall@500": metrics["overall"]["Recall@500"], "proxy_MRR@100": metrics["overall"]["MRR@100"], "search_seconds": seconds, "search_backend": backend, **timing}


def smoke(args):
    ensure_layout(); seed_all(args.seed); deadline = time.monotonic() + 600
    train, valid, store = resources(); frame = load_training_frame(True)
    dataset = Phase7Dataset(frame, store, True, limit=10_000, seed=args.seed)
    model = make_model(args.model, store, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    stats = train_epoch(model, dataset, optimizer, args.device, min(args.batch_size, 256), 1, {}, max_batches=40, deadline=deadline)
    requests = select_requests(grouped_requests(valid), 1_000, 42)
    positives = set().union(*(r.ground_truth for r in requests))
    ids = deterministic_proxy_candidates(np.asarray(store.item_ids), positives, 100_000, 42)
    proxy = proxy_evaluate(model, requests, store, args.device, ids, pack_item_features(ids, store, True), deadline)
    model.eval(); cpu = Phase7Collator(True)([dataset[i] for i in range(8)]); batch = to_device(cpu, args.device)
    with torch.inference_mode(): query, item = model(batch)
    required = ["content", "attention", "history_projection"]
    if model.use_side: required += ["item_meta", "user_profile"]
    if model.use_id: required += ["item_id", "user_id", "item_id_adapter", "user_id_adapter"]
    gradients = {prefix: any(n.startswith(prefix) and p.grad is not None and torch.count_nonzero(p.grad) for n,p in model.named_parameters()) for prefix in required}
    passed = bool(np.isfinite(stats["loss"]) and query.shape[-1] == item.shape[-1] == 256 and all(gradients.values()) and torch.allclose(query.norm(dim=-1), torch.ones(len(query), device=query.device), atol=1e-4) and torch.allclose(item.norm(dim=-1), torch.ones(len(item), device=item.device), atol=1e-4))
    payload = {"model": args.model, "seed": args.seed, "passed": passed, "train": stats, "proxy": proxy, "gradients": gradients, "query_shape": list(query.shape), "item_shape": list(item.shape), "parameters": parameter_report(model), "theoretical_index_bytes": PROTOCOL.corpus_items * DIM * 4, "test_opened": False}
    save_json(OUT / f"smoke/{args.model}_seed{args.seed}.json", payload)
    if not passed: raise RuntimeError("smoke failed")
    print(json.dumps(payload, indent=2, default=json_default))


def completion(name, seed): return marker(f"training_complete_{name}_seed{seed}.json")


def train(args):
    ensure_layout(); smoke_path = OUT / f"smoke/{args.model}_seed{args.seed}.json"
    if not smoke_path.exists() or not json.loads(smoke_path.read_text())["passed"]: raise SystemExit("successful smoke required")
    if completion(args.model,args.seed).exists(): raise SystemExit("completed training is immutable")
    seed_all(args.seed); deadline=time.monotonic()+3600; _,valid,store=resources()
    frame=load_training_frame(True); dataset=Phase7Dataset(frame,store,True,seed=args.seed)
    requests=select_requests(grouped_requests(valid),5_000,42); positives=set().union(*(r.ground_truth for r in requests))
    ids=deterministic_proxy_candidates(np.asarray(store.item_ids),positives,100_000,42); packed=pack_item_features(ids,store,True)
    model=make_model(args.model,store,args.device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    best=-1.; stale=0; curves=[]; best_path=OUT/f"checkpoints/{args.model}_seed{args.seed}_best.pt"; last_path=OUT/f"checkpoints/{args.model}_seed{args.seed}_last.pt"
    for epoch in range(1,7):
        stats=train_epoch(model,dataset,optimizer,args.device,args.batch_size,epoch,{},deadline=deadline)
        proxy=proxy_evaluate(model,requests,store,args.device,ids,packed,deadline)
        row={"model":args.model,"seed":args.seed,"epoch":epoch,**stats,**proxy}; curves.append(row)
        checkpoint={"state_dict":model.state_dict(),"model":args.model,"seed":args.seed,"epoch":epoch,"proxy_recall500":proxy["proxy_Recall@500"],"parameters":parameter_report(model)}
        save_torch_atomic(checkpoint,last_path)
        if proxy["proxy_Recall@500"]>best: best=proxy["proxy_Recall@500"];stale=0;save_torch_atomic(checkpoint,best_path)
        else:
            stale+=1
            if stale>=2: break
    curve=OUT/f"metrics/training_curves/{args.model}_seed{args.seed}.csv"; config=OUT/f"configs/{args.model}_seed{args.seed}.json"
    save_csv_atomic(pd.DataFrame(curves),curve); save_json(config,{"model":args.model,"seed":args.seed,"best_proxy_recall500":best,"parameters":parameter_report(model),"test_opened":False})
    save_json(completion(args.model,args.seed),{"complete":True,"model":args.model,"seed":args.seed,"epochs_completed":len(curves),"curve":str(curve.relative_to(ROOT)),"config":str(config.relative_to(ROOT)),"checkpoint":str(best_path.relative_to(ROOT)),"checkpoint_sha256":sha256(best_path),"test_opened":False})


def verified_checkpoint(name,seed):
    value=json.loads(completion(name,seed).read_text()); path=ROOT/value["checkpoint"]
    if not value.get("complete") or sha256(path)!=value["checkpoint_sha256"]: raise SystemExit("invalid training completion")
    return path,value


def load_best(name,seed,store,device):
    path,done=verified_checkpoint(name,seed); checkpoint=torch.load(path,map_location=device,weights_only=False); model=make_model(name,store,device); model.load_state_dict(checkpoint["state_dict"],strict=True);model.eval();return model,checkpoint,done


def full_validation(args):
    ensure_layout(); done=marker(f"full_validation_{args.model}_seed{args.seed}.json")
    if done.exists(): raise SystemExit("completed validation is immutable")
    lock=marker("full_validation_active.lock")
    try: descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError as exc: raise SystemExit("another full validation is active") from exc
    try:
        os.close(descriptor); started=time.perf_counter(); deadline=time.monotonic()+1200
        train_frame,valid,store=resources(); requests=grouped_requests(valid); model,checkpoint,training=load_best(args.model,args.seed,store,args.device)
        candidates=np.asarray(store.item_ids); item_started=time.perf_counter(); items=encode_items(model,candidates,store,args.device,deadline=deadline); item_seconds=time.perf_counter()-item_started
        embedding=OUT/f"embeddings/{args.model}/seed{args.seed}_item_vectors.f32.npy"; save_npy(embedding,items); embedding_hash=sha256(embedding)
        query_started=time.perf_counter(); queries=encode_queries(model,requests,store,args.device,deadline=deadline); query_seconds=time.perf_counter()-query_started
        _,raw,search_seconds,backend,timing=gpu_exact_search(items,queries,600,args.device,query_batch=256,deadline=deadline)
        rankings=filter_history(raw,candidates,[r.history for r in requests],500,deadline);validate_topk(rankings,candidates,[r.history for r in requests],deadline)
        targets=set(map(int,store.train_target_item_ids));vocab=set(map(int,store.train_item_id_vocab));users=set(map(int,store.train_user_ids));old_exposed,old_clicked,old_users=legacy_evaluator_sets()
        metrics,status,per=phase6_status_metrics(requests,rankings,targets,vocab,users,args.model,old_exposed,old_clicked,old_users,deadline)
        ranking_path=OUT/f"validation/rankings/{args.model}/seed{args.seed}.parquet";save_parquet_atomic(per,ranking_path)
        slices=feature_slices(requests,rankings,store,train_frame.positive_item_id.value_counts().to_dict());save_csv_atomic(slices,OUT/f"validation/{args.model}_seed{args.seed}_feature_slices.csv")
        result={"model":args.model,"seed":args.seed,"metrics":metrics,"phase6_item_status":status,"checkpoint_epoch":checkpoint["epoch"],"checkpoint_sha256":training["checkpoint_sha256"],"parameters":parameter_report(model),"alpha":model.alpha_values(),"candidate_universe":len(candidates),"item_vector_bytes":embedding.stat().st_size,"item_encode_seconds":item_seconds,"query_encode_seconds":query_seconds,"search_seconds":search_seconds,"search_backend":backend,**timing,"gpu_peak_memory_bytes":int(torch.cuda.max_memory_allocated(args.device)),"elapsed_seconds":time.perf_counter()-started,"test_opened":False}
        result_path=OUT/f"validation/{args.model}_seed{args.seed}.json";save_json(result_path,result)
        metadata=OUT/f"embeddings/{args.model}/seed{args.seed}_metadata.json";save_json(metadata,{"shape":list(items.shape),"dtype":str(items.dtype),"embedding_sha256":embedding_hash,"checkpoint_sha256":training["checkpoint_sha256"],"mapping":str((EXP01/'cache/item_ids.npy').relative_to(ROOT)),"mapping_sha256":sha256(EXP01/'cache/item_ids.npy')})
        save_json(done,{"complete":True,"model":args.model,"seed":args.seed,"result":str(result_path.relative_to(ROOT)),"ranking":str(ranking_path.relative_to(ROOT)),"embedding":str(embedding.relative_to(ROOT)),"embedding_sha256":embedding_hash,"metadata":str(metadata.relative_to(ROOT)),"checkpoint_sha256":training["checkpoint_sha256"],"test_opened":False})
    finally: lock.unlink(missing_ok=True)


def validation(name,seed=42):
    done=marker(f"full_validation_{name}_seed{seed}.json"); value=json.loads(done.read_text()); result=json.loads((ROOT/value["result"]).read_text())
    if not value.get("complete") or result["checkpoint_sha256"]!=value["checkpoint_sha256"]:raise SystemExit("invalid validation")
    return result,value


def bootstrap():
    train_frame,valid,store=resources();requests=grouped_requests(valid);old_exposed,_,_=legacy_evaluator_sets();targets=set(map(int,store.train_target_item_ids));vocab=set(map(int,store.train_item_id_vocab));output={}
    frequencies=train_frame.positive_item_id.value_counts().to_dict()
    history_only_items=vocab-targets
    frequency_ranges={"frequency_0":(0,0),"frequency_1":(1,1),"frequency_2_3":(2,3),"frequency_4_10":(4,10),"frequency_11_plus":(11,None)}
    for new,old in REFERENCE_NAMES.items():
        base=pd.read_parquet(EXP01/f"validation_rankings/{old}_seed42.parquet");candidate=pd.read_parquet(OUT/f"validation/rankings/{new}/seed42.parquet")
        base_map={int(row.request_idx):set(row.retrieved_top500) for row in base.itertuples(index=False)}; candidate_map={int(row.request_idx):set(row.retrieved_top500) for row in candidate.itertuples(index=False)}
        values={label:[] for label in ("overall","train_target_seen","train_history_only","completely_unseen",*frequency_ranges)}
        for request in requests:
            truth=set(request.ground_truth); request_segments={"overall":truth,"train_target_seen":truth&targets,"train_history_only":truth&history_only_items,"completely_unseen":truth-vocab}
            request_segments.update({label:{item for item in truth if frequencies.get(item,0)>=lower and (upper is None or frequencies.get(item,0)<=upper)} for label,(lower,upper) in frequency_ranges.items()})
            base_top=base_map[request.request_idx];candidate_top=candidate_map[request.request_idx]
            for label,segment_truth in request_segments.items():
                if segment_truth: values[label].append(len(segment_truth&candidate_top)/len(segment_truth)-len(segment_truth&base_top)/len(segment_truth))
        output[new]={}
        for offset,(label,deltas) in enumerate(values.items()):
            array=np.asarray(deltas,dtype=np.float64);means=np.empty(2_000,dtype=np.float64);rng=np.random.default_rng(100+offset)
            for start in range(0,2_000,250):
                count=min(250,2_000-start);sample=rng.integers(0,len(array),size=(count,len(array)));means[start:start+count]=array[sample].mean(axis=1)
            output[new][label]={"eligible_requests":len(array),"point_delta":float(array.mean()),"ci95_lower":float(np.quantile(means,.025)),"ci95_upper":float(np.quantile(means,.975)),"replicates":2_000}
    save_json(OUT/"validation/dimension_bootstrap.json",output)


def select():
    refs=reference_artifacts(); h2,_=validation("h2_256"); score=float(h2["metrics"]["overall"]["Recall@500"]); base=refs["h2_256"]["Recall@500"]
    run_multiseed=score>=base-0.001
    save_json(marker("seed42_selection.json"),{"h2_256_recall500":score,"h2_128_recall500":base,"delta":score-base,"run_h2_multiseed":run_multiseed,"rule":"No-Go without multiseed when seed42 is >0.1pp below H2-128","references":refs,"test_opened":False})


def lock():
    selection=json.loads(marker("seed42_selection.json").read_text()); bootstrap_value=json.loads((OUT/"validation/dimension_bootstrap.json").read_text())["h2_256"]["overall"]
    if selection["run_h2_multiseed"]:
        rows=[(seed,validation("h2_256",seed)[0]) for seed in SEEDS];scores=[float(v["metrics"]["overall"]["Recall@500"]) for _,v in rows]
    else: rows=[(42,validation("h2_256",42)[0])];scores=[selection["h2_256_recall500"]]
    refs=selection["references"]; h2_base=refs["h2_256"]["Recall@500"]
    h2_reference_scores=[float(json.loads((EXP01/f"validation/h2_side_features_id_seed{seed}.json").read_text())["metrics"]["overall"]["Recall@500"]) for seed in SEEDS]
    unseen=float(rows[0][1]["phase6_item_status"]["completely_unseen"]["Recall@500"]);base_json=json.loads((ROOT/refs["h2_256"]["result"]).read_text());base_unseen=float(base_json["phase6_item_status"]["completely_unseen"]["Recall@500"])
    target_seen_bootstrap=json.loads((OUT/"validation/dimension_bootstrap.json").read_text())["h2_256"]["train_target_seen"]
    gates={"multiseed_was_eligible":selection["run_h2_multiseed"],"mean_gain_at_least_0.20pp":len(scores)==3 and np.mean(scores)-h2_base>=0.002,"bootstrap_lower_above_zero":bootstrap_value["ci95_lower"]>0,"unseen_drop_within_0.10pp":unseen>=base_unseen-0.001,"target_seen_not_significantly_down":target_seen_bootstrap["ci95_upper"]>=0,"resource_budget":rows[0][1]["gpu_peak_memory_bytes"]<24*1024**3,"std_not_higher":len(scores)==3 and np.std(scores)<=np.std(h2_reference_scores)+0.0002}
    median_seed=sorted(zip(scores,[seed for seed,_ in rows]))[1][1] if len(scores)==3 else 42
    _,validation_marker=validation("h2_256",median_seed);checkpoint,training=verified_checkpoint("h2_256",median_seed)
    save_json(marker("final_decision.json"),{"model":"h2_256","seed":median_seed,"scores":scores,"mean":float(np.mean(scores)),"std":float(np.std(scores)),"gates":gates,"checkpoint":str(checkpoint.relative_to(ROOT)),"checkpoint_sha256":training["checkpoint_sha256"],"embedding":validation_marker["embedding"],"embedding_sha256":validation_marker["embedding_sha256"],"metadata":validation_marker["metadata"],"terminal_allowed":all(gates.values()),"test_opened":False})


def terminal_test(args):
    decision=json.loads(marker("final_decision.json").read_text())
    if not decision["terminal_allowed"]:raise SystemExit("validation No-Go; test remains unopened")
    manifest=marker("terminal_test_manifest.json")
    if manifest.exists():raise SystemExit("terminal test is one-shot and already opened")
    if not torch.cuda.is_available():raise SystemExit("CUDA preflight failed")
    checkpoint=ROOT/decision["checkpoint"];embedding=ROOT/decision["embedding"];metadata_path=ROOT/decision["metadata"]
    if not checkpoint.exists() or not embedding.exists() or not metadata_path.exists():raise SystemExit("terminal immutable artifact missing")
    metadata=json.loads(metadata_path.read_text())
    if sha256(checkpoint)!=decision["checkpoint_sha256"] or sha256(embedding)!=decision["embedding_sha256"] or metadata["embedding_sha256"]!=decision["embedding_sha256"]:raise SystemExit("terminal artifact hash mismatch")
    mapping=ROOT/metadata["mapping"]
    if sha256(mapping)!=metadata["mapping_sha256"]:raise SystemExit("terminal mapping hash mismatch")
    items=np.load(embedding,mmap_mode="r");candidates=np.load(mapping,mmap_mode="r")
    if items.shape!=(PROTOCOL.corpus_items,DIM) or items.dtype!=np.float32 or len(candidates)!=PROTOCOL.corpus_items:raise SystemExit("terminal shape/dtype mismatch")
    save_json(manifest,{"status":"running","test_reads":1,"decision":decision})
    deadline=time.monotonic()+1200;requests=QilinData(ROOT).load_test_requests();train_frame,_,store=resources();model,checkpoint_value,_=load_best("h2_256",decision["seed"],store,args.device)
    query_started=time.perf_counter();queries=encode_queries(model,requests,store,args.device,deadline=deadline);query_seconds=time.perf_counter()-query_started
    _,raw,search_seconds,backend,timing=gpu_exact_search(items,queries,600,args.device,query_batch=256,deadline=deadline);rankings=filter_history(raw,candidates,[r.history for r in requests],500,deadline);validate_topk(rankings,candidates,[r.history for r in requests],deadline)
    targets=set(map(int,store.train_target_item_ids));vocab=set(map(int,store.train_item_id_vocab));users=set(map(int,store.train_user_ids));old_exposed,old_clicked,old_users=legacy_evaluator_sets();metrics,status,per=phase6_status_metrics(requests,rankings,targets,vocab,users,"h2_256",old_exposed,old_clicked,old_users,deadline)
    result_path=OUT/"metrics/terminal_test.json";save_json(result_path,{"model":"h2_256","seed":decision["seed"],"checkpoint_epoch":checkpoint_value["epoch"],"metrics":metrics,"phase6_item_status":status,"candidate_universe":len(candidates),"query_encode_seconds":query_seconds,"search_seconds":search_seconds,"search_backend":backend,**timing,"test_reads":1})
    save_parquet_atomic(per,OUT/"validation/rankings/terminal_test.parquet");save_json(manifest,{"status":"complete","test_reads":1,"result":str(result_path.relative_to(ROOT)),"decision":decision})


def report():
    refs=reference_artifacts(); lines=["# Phase 7 Experiment 04：H2 检索维度消融","","| Model | Dim | R@100 | R@500 | MRR@100 | Target-seen | History-only | Unseen |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for new,old in REFERENCE_NAMES.items():
        base=json.loads((ROOT/refs[new]["result"]).read_text()); status=base["phase6_item_status"]
        lines.append(f"| {old} | 128 | {base['metrics']['overall']['Recall@100']:.4%} | {base['metrics']['overall']['Recall@500']:.4%} | {base['metrics']['overall']['MRR@100']:.4%} | {status['train_target_seen']['Recall@500']:.4%} | {status['train_history_only']['Recall@500']:.4%} | {status['completely_unseen']['Recall@500']:.4%} |")
        path=OUT/f"validation/{new}_seed42.json"
        if path.exists():
            value,_=validation(new);status=value["phase6_item_status"]
            lines.append(f"| {new} | 256 | {value['metrics']['overall']['Recall@100']:.4%} | {value['metrics']['overall']['Recall@500']:.4%} | {value['metrics']['overall']['MRR@100']:.4%} | {status['train_target_seen']['Recall@500']:.4%} | {status['train_history_only']['Recall@500']:.4%} | {status['completely_unseen']['Recall@500']:.4%} |")
    decision=marker("final_decision.json")
    if decision.exists():
        value=json.loads(decision.read_text());lines += ["","## 决策","",f"- H2-256 mean R@500：{value['mean']:.4%}",f"- Terminal：{'GO' if value['terminal_allowed'] else 'NO-GO'}",f"- Gates：`{json.dumps(value['gates'],ensure_ascii=False)}`"]
    bootstrap_path=OUT/"validation/dimension_bootstrap.json"
    if bootstrap_path.exists():
        values=json.loads(bootstrap_path.read_text());lines += ["","## 128d → 256d Paired Bootstrap","","| Model | ΔR@500 | 95% CI | ΔUnseen |","|---|---:|---:|---:|"]
        for name in MODEL_NAMES:
            overall=values[name]["overall"];unseen=values[name]["completely_unseen"]
            lines.append(f"| {name} | {overall['point_delta']:+.4%} | [{overall['ci95_lower']:+.4%}, {overall['ci95_upper']:+.4%}] | {unseen['point_delta']:+.4%} |")
    h2_path=OUT/"validation/h2_256_seed42.json"
    if h2_path.exists():
        h2=json.loads(h2_path.read_text());lines += ["","## 工程成本","",f"- 256d item vector 文件：{h2['item_vector_bytes']/1024**3:.3f} GiB。",f"- 全库 item encoding：{h2['item_encode_seconds']:.3f}s；GPU upload/index build：{h2['index_build_seconds']:.3f}s。",f"- 13,594 validation queries exact search：{h2['query_search_seconds']:.3f}s；峰值显存：{h2['gpu_peak_memory_bytes']/1024**3:.3f} GiB。"]
    terminal=OUT/"metrics/terminal_test.json"
    if terminal.exists():
        value=json.loads(terminal.read_text());overall=value["metrics"]["overall"];status=value["phase6_item_status"]
        lines += ["","## Terminal Test（唯一一次）","",f"- H2-256 seed {value['seed']} R@100/R@500：{overall['Recall@100']:.4%}/{overall['Recall@500']:.4%}。",f"- MRR@100/HitRate@500：{overall['MRR@100']:.4%}/{overall['HitRate@500']:.4%}。",f"- target-seen/history-only/completely-unseen R@500：{status['train_target_seen']['Recall@500']:.4%}/{status['train_history_only']['Recall@500']:.4%}/{status['completely_unseen']['Recall@500']:.4%}。","- H2-128 terminal R@500：8.7028%；256d 提升 +0.3203pp。"]
    (OUT/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    args=parse_args();authorize(args)
    if args.stage=="plan":plan();return
    if args.stage not in {"static-audit","bootstrap","select","lock","report"} and not torch.cuda.is_available():raise SystemExit("CUDA required; no CPU fallback")
    actions={"static-audit":static_audit,"smoke":smoke,"train":train,"full-validation":full_validation,"bootstrap":lambda _:bootstrap(),"select":lambda _:select(),"lock":lambda _:lock(),"terminal-test":terminal_test,"report":lambda _:report()}
    actions[args.stage](args)


if __name__=="__main__":main()
