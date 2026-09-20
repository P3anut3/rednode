#!/usr/bin/env python3
"""Phase 2.5: controlled English-vs-Chinese zero-shot encoder benchmark."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common.data import QilinData, TestRequest  # noqa: E402
from experiments.common.dense_recall import (  # noqa: E402
    build_user_embeddings, make_row_lookup, rows_to_filtered_rankings, validate_rankings,
)
from experiments.common.metrics import evaluate_rankings  # noqa: E402
from experiments.phase_02.experiment_01_zero_shot_dense.recall.dense_encoder import valid_text  # noqa: E402
from experiments.phase_02.experiment_01_zero_shot_dense.recall.dense_index import peak_rss_gib, search_index  # noqa: E402
from experiments.phase_02.experiment_02_chinese_encoder_benchmark.recall.encoder_benchmark import (  # noqa: E402
    EncoderSpec, basic_clean, compose_item_text, encode_corpus, encode_texts, is_emoji,
    load_model, load_sample_records, model_specs, prepare_note_ids,
)


OUT = ROOT / "results/phase_02/experiment_02_chinese_encoder_benchmark"
EMBEDDINGS = OUT / "embeddings"
INDICES = OUT / "indices"
PER_REQUEST = OUT / "per_request"
SEED = 42
MAX_LENGTH = 256
SAMPLE_SIZE = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("diagnostics", "encode", "evaluate", "report", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--subset-requests", type=int, default=5000)
    return parser.parse_args()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(map(str, frame.columns))
    rows = [[str(value).replace("|", "\\|") for value in row]
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ])


def is_chinese(char: str) -> bool:
    code = ord(char)
    return 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF


def diagnostic_row(spec: EncoderSpec, preprocessing: str, records: list[dict]) -> tuple[dict, list[dict]]:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec.snapshot, local_files_only=True)
    texts = [compose_item_text(row["title"], row["content"], row["taxonomy1"], row["taxonomy2"],
                               spec, preprocessing) for row in records]
    # [UNK] is listed among all_special_ids by Hugging Face, but it must stay
    # in both the diagnostic numerator and denominator.
    special = set(tokenizer.all_special_ids) - {tokenizer.unk_token_id}
    lengths, unk_count, token_count, empty = [], 0, 0, 0
    for offset in range(0, len(texts), 128):
        encoded = tokenizer(texts[offset:offset + 128], add_special_tokens=True, truncation=False)["input_ids"]
        for ids in encoded:
            content_ids = [token for token in ids if token not in special]
            lengths.append(len(ids))
            token_count += len(content_ids)
            unk_count += sum(token == tokenizer.unk_token_id for token in content_ids)
            empty += not content_ids

    chinese_cache: dict[str, bool] = {}
    emoji_cache: dict[str, tuple[bool, list[str]]] = {}
    chinese_total = chinese_unk = emoji_total = emoji_unk = 0
    for text in texts:
        for char in text:
            if is_chinese(char):
                chinese_total += 1
                if char not in chinese_cache:
                    ids = tokenizer(char, add_special_tokens=False)["input_ids"]
                    chinese_cache[char] = not ids or tokenizer.unk_token_id in ids
                chinese_unk += chinese_cache[char]
            if is_emoji(char):
                emoji_total += 1
                if char not in emoji_cache:
                    ids = tokenizer(char, add_special_tokens=False)["input_ids"]
                    emoji_cache[char] = (not ids or tokenizer.unk_token_id in ids,
                                         tokenizer.convert_ids_to_tokens(ids))
                emoji_unk += emoji_cache[char][0]
    values = np.asarray(lengths)
    row = {
        "model_key": spec.key, "model_name": spec.model_name, "preprocessing": preprocessing,
        "sample_items": len(texts), "unk_tokens": unk_count,
        "token_count_excluding_special": token_count, "unk_rate": unk_count / token_count if token_count else math.nan,
        "avg_tokens": float(values.mean()), "p50_tokens": float(np.quantile(values, .5)),
        "p95_tokens": float(np.quantile(values, .95)), "p99_tokens": float(np.quantile(values, .99)),
        "truncation_rate": float(np.mean(values > MAX_LENGTH)), "empty_after_tokenization": empty,
        "chinese_characters": chinese_total, "chinese_unk_characters": chinese_unk,
        "chinese_unk_rate": chinese_unk / chinese_total if chinese_total else math.nan,
        "emoji_characters": emoji_total, "emoji_unk_characters": emoji_unk,
        "emoji_unk_rate": emoji_unk / emoji_total if emoji_total else math.nan,
    }
    emoji_examples = [{"emoji": char, "is_unk": status, "tokens": tokens}
                      for char, (status, tokens) in sorted(emoji_cache.items())[:30]]
    return row, emoji_examples


def sample_health(records: list[dict]) -> dict:
    values = {"chinese": 0, "english": 0, "emoji": 0, "special_symbol": 0,
              "both_title_content_empty": 0}
    for row in records:
        text = f"{row['title'] or ''} {row['content'] or ''}"
        values["chinese"] += any(is_chinese(char) for char in text)
        values["english"] += any("a" <= char.lower() <= "z" for char in text)
        values["emoji"] += any(is_emoji(char) for char in text)
        values["special_symbol"] += any(char in "#@<>/&%$" for char in text)
        values["both_title_content_empty"] += not valid_text(row["title"]) and not valid_text(row["content"])
    if any(values[key] == 0 for key in values):
        raise AssertionError(f"Diagnostic sample misses required text class: {values}")
    return values


def sanity_checks(specs: dict[str, EncoderSpec], records: list[dict]) -> dict:
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(len(records), size=20, replace=False))
    samples = []
    for idx in chosen:
        row = records[int(idx)]
        original = compose_item_text(row["title"], row["content"], row["taxonomy1"], row["taxonomy2"],
                                     specs["bge_base_zh"], "raw")
        cleaned = basic_clean(original)
        samples.append({"note_idx": row["note_idx"], "original": original, "cleaned": cleaned, "models": {}})
    taxonomy_groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(records):
        taxonomy_groups[str(row["taxonomy2"] or row["taxonomy1"] or "")].append(idx)
    usable = [values for key, values in taxonomy_groups.items() if key and len(values) >= 2]
    same_pairs = [tuple(map(int, rng.choice(values, 2, replace=False))) for values in usable[:10]]
    different_pairs = []
    while len(different_pairs) < 10:
        a, b = map(int, rng.choice(len(records), 2, replace=False))
        ta = str(records[a]["taxonomy2"] or records[a]["taxonomy1"] or "")
        tb = str(records[b]["taxonomy2"] or records[b]["taxonomy1"] or "")
        if ta and tb and ta != tb:
            different_pairs.append((a, b))
    pair_ids = same_pairs + different_pairs
    pair_output = {}
    for spec in specs.values():
        tokenizer, model = load_model(spec, "cuda:0")
        for sample in samples:
            encoded = tokenizer(sample["cleaned"], add_special_tokens=True, truncation=False)["input_ids"]
            sample["models"][spec.key] = {
                "tokens_first_40": tokenizer.convert_ids_to_tokens(encoded[:40]),
                "unk_count": sum(token == tokenizer.unk_token_id for token in encoded),
                "token_length": len(encoded),
            }
        unique_indices = sorted(set(index for pair in pair_ids for index in pair))
        texts = [compose_item_text(records[index]["title"], records[index]["content"],
                                   records[index]["taxonomy1"], records[index]["taxonomy2"],
                                   spec, "basic_clean") for index in unique_indices]
        vectors = encode_texts(texts, tokenizer, model, spec, "cuda:0", MAX_LENGTH).astype(np.float32)
        vector_by_index = {index: vectors[pos] for pos, index in enumerate(unique_indices)}
        pair_output[spec.key] = [
            {"pair_type": "same_taxonomy" if pos < len(same_pairs) else "different_taxonomy",
             "note_a": records[a]["note_idx"], "note_b": records[b]["note_idx"],
             "taxonomy_a": str(records[a]["taxonomy2"] or records[a]["taxonomy1"] or ""),
             "taxonomy_b": str(records[b]["taxonomy2"] or records[b]["taxonomy1"] or ""),
             "cosine": float(vector_by_index[a] @ vector_by_index[b])}
            for pos, (a, b) in enumerate(pair_ids)
        ]
        del model, tokenizer
        import torch
        torch.cuda.empty_cache()
    return {"seed": SEED, "random_text_samples": samples, "random_item_pairs": pair_output}


def diagnostics_stage() -> None:
    specs = model_specs()
    note_ids = prepare_note_ids(ROOT, OUT)
    records = load_sample_records(ROOT, note_ids, SAMPLE_SIZE, SEED)
    health = sample_health(records)
    rows, emoji = [], {}
    for spec in specs.values():
        row, examples = diagnostic_row(spec, "basic_clean", records)
        rows.append(row); emoji[f"{spec.key}__basic_clean"] = examples
    for preprocessing in ("raw", "remove_emoji"):
        row, examples = diagnostic_row(specs["bge_base_zh"], preprocessing, records)
        rows.append(row); emoji[f"bge_base_zh__{preprocessing}"] = examples
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "tokenizer_diagnostics.csv", index=False)
    display = frame[["model_name", "preprocessing", "unk_rate", "chinese_unk_rate", "avg_tokens",
                     "p50_tokens", "p95_tokens", "p99_tokens", "truncation_rate",
                     "empty_after_tokenization", "emoji_unk_rate"]].copy()
    for column in ("unk_rate", "chinese_unk_rate", "truncation_rate", "emoji_unk_rate"):
        display[column] = display[column].map(lambda value: f"{100*value:.4f}%" if pd.notna(value) else "n/a")
    (OUT / "tokenizer_diagnostics.md").write_text(
        "# Tokenizer diagnostics\n\nFixed seed=42 sample of 10,000 notes. Token lengths include special tokens; "
        "UNK rate excludes special tokens. Truncation threshold is 256.\n\n" + markdown_table(display) +
        "\n\n## Sample coverage\n\n```json\n" + json.dumps(health, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    dump_json(OUT / "emoji_tokenization.json", emoji)
    dump_json(OUT / "sanity_checks.json", sanity_checks(specs, records))


def embedding_dir(model_key: str, preprocessing: str) -> Path:
    return EMBEDDINGS / model_key / preprocessing


def encode_stage(args: argparse.Namespace) -> None:
    specs = model_specs()
    note_ids = prepare_note_ids(ROOT, OUT)
    jobs = [(specs["e5_base_v2"], "basic_clean")]
    jobs += [(specs["bge_base_zh"], preprocessing) for preprocessing in ("raw", "basic_clean", "remove_emoji")]
    for spec, preprocessing in jobs:
        print(f"encoding {spec.key}/{preprocessing}", flush=True)
        encode_corpus(ROOT, embedding_dir(spec.key, preprocessing), spec, preprocessing, len(note_ids),
                      batch_size=args.batch_size, max_length=MAX_LENGTH, world_size=args.gpus)


def subset_requests(requests: Sequence[TestRequest], size: int) -> list[TestRequest]:
    rng = np.random.default_rng(SEED)
    chosen = np.sort(rng.choice(len(requests), size=min(size, len(requests)), replace=False))
    return [requests[int(index)] for index in chosen]


def build_flat(embeddings: np.ndarray, dim: int) -> tuple[faiss.IndexFlatIP, dict]:
    faiss.omp_set_num_threads(min(64, faiss.omp_get_max_threads()))
    start = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    for offset in range(0, len(embeddings), 20_000):
        index.add(np.asarray(embeddings[offset:offset + 20_000], dtype=np.float32))
    return index, {"build_seconds": time.perf_counter() - start, "items": index.ntotal,
                   "dimension": dim, "peak_rss_gib": peak_rss_gib()}


def evaluate_one(spec: EncoderSpec, preprocessing: str, requests: Sequence[TestRequest],
                 note_ids: np.ndarray, lookup: np.ndarray, catalog: set[int], data: QilinData,
                 name: str, save_per_request: Path | None = None,
                 save_index: Path | None = None) -> dict:
    path = embedding_dir(spec.key, preprocessing) / "embeddings.f16"
    embeddings = np.memmap(path, mode="r", dtype=np.float16, shape=(len(note_ids), spec.dim))
    queries, query_stats = build_user_embeddings(requests, embeddings, lookup, history_n=10, weighting="mean")
    index, build_stats = build_flat(embeddings, spec.dim)
    if save_index is not None:
        save_index.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(save_index))
        build_stats["index_file_bytes"] = save_index.stat().st_size
    _, rows, search_stats = search_index(index, queries, topk=600, batch_size=128)
    rankings = rows_to_filtered_rankings(rows, note_ids, requests)
    validation = validate_rankings(rankings, requests, catalog)
    metrics, per_request = evaluate_rankings(
        requests, rankings, data.train_exposed_items, data.train_users, name,
        clicked_items=set(data.train_click_counts),
    )
    if save_per_request is not None:
        save_per_request.parent.mkdir(parents=True, exist_ok=True)
        per_request.to_parquet(save_per_request, index=False, compression="zstd")
    del index, embeddings, rows, rankings
    return {"name": name, "model_key": spec.key, "model_name": spec.model_name,
            "preprocessing": preprocessing, "dimension": spec.dim, "query_stats": query_stats,
            "index_build": build_stats, "search": search_stats, "validation": validation,
            "metrics": metrics}


def result_row(payload: dict) -> dict:
    metrics = payload["metrics"]
    overall = metrics["overall"]
    return {
        "name": payload["name"], "model_key": payload["model_key"],
        "model_name": payload["model_name"], "preprocessing": payload["preprocessing"],
        **{f"Recall@{k}": overall[f"Recall@{k}"] for k in (10, 50, 100, 200, 500)},
        "MRR@100": overall["MRR@100"], "Warm Recall@500": metrics["warm_item"]["Recall@500"],
        "Cold Recall@500": metrics["cold_item"]["Recall@500"],
        "Train-clicked Recall@500": metrics["train_clicked"]["Recall@500"],
        "Train-exposed-never-clicked Recall@500": metrics["train_exposed_never_clicked"]["Recall@500"],
        "Warm-user Recall@500": metrics["warm_user"]["Recall@500"],
        "Cold-user Recall@500": metrics["cold_user"]["Recall@500"],
        "query_seconds": payload["search"]["query_seconds"],
    }


def evaluate_stage(args: argparse.Namespace) -> None:
    specs = model_specs()
    note_ids = prepare_note_ids(ROOT, OUT)
    lookup = make_row_lookup(note_ids)
    catalog = set(map(int, note_ids))
    data = QilinData(ROOT)
    all_requests = data.load_test_requests()
    subset = subset_requests(all_requests, args.subset_requests)
    # Materialize the train-derived split once and reuse it for every configuration.
    _ = data.train_exposed_items, data.train_users, data.train_click_counts
    jobs = [
        (specs["e5_base_v2"], "basic_clean", "e5_base_v2__basic_clean"),
        (specs["bge_base_zh"], "basic_clean", "bge_base_zh__basic_clean"),
        (specs["bge_base_zh"], "raw", "bge_base_zh__raw"),
        (specs["bge_base_zh"], "remove_emoji", "bge_base_zh__remove_emoji"),
    ]
    payloads = []
    for spec, preprocessing, name in jobs:
        print(f"exact subset {name}", flush=True)
        payload = evaluate_one(spec, preprocessing, subset, note_ids, lookup, catalog, data, name,
                               PER_REQUEST / f"subset_{name}.parquet")
        payloads.append(payload)
        dump_json(OUT / "subset_results" / f"{name}.json", payload)
    encoder_payloads = [payload for payload in payloads if payload["preprocessing"] == "basic_clean"]
    pd.DataFrame(map(result_row, encoder_payloads)).to_csv(OUT / "encoder_results.csv", index=False)
    bge_payloads = [payload for payload in payloads if payload["model_key"] == "bge_base_zh"]
    pd.DataFrame(map(result_row, bge_payloads)).to_csv(OUT / "preprocessing_ablation.csv", index=False)
    best = max(bge_payloads, key=lambda payload: payload["metrics"]["overall"]["Recall@500"])
    selection = {"seed": SEED, "subset_requests": len(subset),
                 "subset_request_ids": [request.request_idx for request in subset],
                 "selection_metric": "Exact Overall Recall@500",
                 "best_new_encoder": "bge_base_zh", "best_preprocessing": best["preprocessing"]}
    dump_json(OUT / "selection.json", selection)
    print(f"full exact bge_base_zh/{best['preprocessing']}", flush=True)
    full = evaluate_one(
        specs["bge_base_zh"], best["preprocessing"], all_requests, note_ids, lookup, catalog, data,
        f"bge_base_zh__{best['preprocessing']}__full", PER_REQUEST / "bge_base_zh_full.parquet",
        INDICES / f"bge_base_zh_{best['preprocessing']}_flat_ip.faiss",
    )
    full["selection"] = selection
    dump_json(OUT / "best_encoder_full.json", full)


def pct(value: float) -> str:
    return f"{100*value:.4f}%"


def report_stage() -> None:
    specs = model_specs()
    diagnostics = pd.read_csv(OUT / "tokenizer_diagnostics.csv")
    encoder = pd.read_csv(OUT / "encoder_results.csv")
    preprocessing = pd.read_csv(OUT / "preprocessing_ablation.csv")
    full = json.loads((OUT / "best_encoder_full.json").read_text())
    phase2 = json.loads((ROOT / "results/phase_02/experiment_01_zero_shot_dense/exact.json").read_text())
    phase1 = pd.read_csv(ROOT / "results/phase_01/experiment_01_full_corpus_baselines/full/summary.csv")
    tfidf = phase1[(phase1.method == "tfidf_history20__with_history_filter") &
                   (phase1.segment == "overall")].iloc[0]
    tfidf_cold = phase1[(phase1.method == "tfidf_history20__with_history_filter") &
                        (phase1.segment == "cold_item")].iloc[0]

    model_rows = [{"Model": spec.model_name, "Parameters": f"{spec.parameters/1e6:.1f}M",
                   "Dim": spec.dim, "Tokenizer": spec.tokenizer_type, "Max length used": MAX_LENGTH,
                   "Chinese support": spec.language_support} for spec in specs.values()]
    diag_main = diagnostics[diagnostics.preprocessing == "basic_clean"].copy()
    diag_display = pd.DataFrame({
        "Model": diag_main.model_name, "UNK Rate": diag_main.unk_rate.map(pct),
        "Chinese UNK Rate": diag_main.chinese_unk_rate.map(pct),
        "Avg Tokens": diag_main.avg_tokens.map(lambda x: f"{x:.1f}"),
        "P95 Tokens": diag_main.p95_tokens.map(lambda x: f"{x:.0f}"),
        "Truncation Rate": diag_main.truncation_rate.map(pct),
    })
    encoder_display = encoder[["model_name", "Recall@100", "Recall@500", "MRR@100",
                               "Warm Recall@500", "Cold Recall@500"]].copy()
    encoder_display.columns = ["Model", "R@100", "R@500", "MRR@100", "Warm R@500", "Cold R@500"]
    for column in encoder_display.columns[1:]:
        encoder_display[column] = encoder_display[column].map(pct)
    prep_display = preprocessing[["preprocessing", "Recall@100", "Recall@500", "Cold Recall@500", "MRR@100"]].copy()
    prep_display.columns = ["Preprocessing", "R@100", "R@500", "Cold R@500", "MRR@100"]
    for column in prep_display.columns[1:]:
        prep_display[column] = prep_display[column].map(pct)

    new = full["metrics"]
    comparisons = pd.DataFrame([
        {"Method": "TF-IDF", "R@100": tfidf["Recall@100"], "R@500": tfidf["Recall@500"],
         "Cold R@500": tfidf_cold["Recall@500"]},
        {"Method": "E5-base-v2 Exact (Phase 2)", "R@100": phase2["metrics"]["overall"]["Recall@100"],
         "R@500": phase2["metrics"]["overall"]["Recall@500"],
         "Cold R@500": phase2["metrics"]["cold_item"]["Recall@500"]},
        {"Method": f"BGE-base-zh Exact ({full['preprocessing']})", "R@100": new["overall"]["Recall@100"],
         "R@500": new["overall"]["Recall@500"], "Cold R@500": new["cold_item"]["Recall@500"]},
    ])
    for column in comparisons.columns[1:]:
        comparisons[column] = comparisons[column].map(pct)

    e5_diag = diag_main[diag_main.model_key == "e5_base_v2"].iloc[0]
    bge_diag = diag_main[diag_main.model_key == "bge_base_zh"].iloc[0]
    e5_full = phase2["metrics"]["overall"]["Recall@500"]
    new_full = new["overall"]["Recall@500"]
    tfidf_full = float(tfidf["Recall@500"])
    case = "B" if new_full >= tfidf_full else ("A" if new_full >= 3 * e5_full else "C")
    if case == "C":
        recommendation = "改 user representation；不要立即 fine-tune item encoder"
    elif case == "A":
        recommendation = "语言适配有效；下一步先改 user representation，再评估 fine-tuning"
    else:
        recommendation = "zero-shot dense 已接近 TF-IDF；下一步优先研究 hybrid/user representation"
    clean_row = preprocessing[preprocessing.preprocessing == "basic_clean"].iloc[0]
    raw_row = preprocessing[preprocessing.preprocessing == "raw"].iloc[0]
    emoji_row = preprocessing[preprocessing.preprocessing == "remove_emoji"].iloc[0]
    lines = [
        "# Phase 2.5 — Chinese Encoder Benchmark", "",
        "## Protocol", "",
        "- Candidate universe: 1,983,938 notes; exact `IndexFlatIP`; L2-normalized inner product.",
        "- `exclude_history=True`, history N=10, simple mean of item embeddings; evaluator and splits reused from Phase 2.",
        "- Item text is taxonomy1 ID + taxonomy2 ID + title + content. Taxonomy values remain IDs, not names.",
        "- No model training, ANN tuning, hybrid fusion, query attention, or target-derived query features.", "",
        "## Models", "", markdown_table(pd.DataFrame(model_rows)), "",
        "## Tokenizer Diagnostics", "", markdown_table(diag_display), "",
        "Full tokenizer statistics and 20 fixed-seed token samples are in `tokenizer_diagnostics.csv` and `sanity_checks.json`.", "",
        "## Encoder Benchmark — Fixed 5K Exact", "", markdown_table(encoder_display), "",
        "## Full-test Comparison", "", markdown_table(comparisons), "",
        "## Preprocessing Ablation — BGE-base-zh, Fixed 5K Exact", "", markdown_table(prep_display), "",
        "## Findings", "",
        f"1. E5 Chinese-character UNK rate is {pct(e5_diag.chinese_unk_rate)}, versus BGE {pct(bge_diag.chinese_unk_rate)}. "
        f"BGE full Recall@500 is {new_full/e5_full:.2f}× the Phase-2 E5 result, so tokenizer/language mismatch explains "
        f"a substantial part of E5's failure. BGE still reaches only {new_full/tfidf_full:.1%} of TF-IDF Recall@500, "
        "so language mismatch is not the whole explanation.",
        "2. Only the locally available Chinese BGE was tested after BGE-M3 download was explicitly cancelled; no multilingual winner is claimed.",
        f"3. BGE Recall@500={pct(new_full)} versus TF-IDF={pct(tfidf_full)}; result category is Case {case}.",
        f"4. BGE warm/cold item Recall@500={pct(new['warm_item']['Recall@500'])}/"
        f"{pct(new['cold_item']['Recall@500'])}. Dense is relatively stronger on cold items, but its absolute cold result "
        f"remains far below TF-IDF ({pct(float(tfidf_cold['Recall@500']))}).",
        f"5. Basic Clean changes 5K Recall@500 from {pct(raw_row['Recall@500'])} to {pct(clean_row['Recall@500'])}.",
        f"6. Removing emoji changes 5K Recall@500 from {pct(clean_row['Recall@500'])} to {pct(emoji_row['Recall@500'])}; "
        "it slightly hurts the primary R@500 objective, so emoji should be retained under this protocol.",
        f"7. BGE no longer has severe Chinese UNK ({pct(bge_diag.chinese_unk_rate)}), but "
        f"{pct(bge_diag.truncation_rate)} of sampled texts exceed the 256-token limit. Emoji tokenization remains weak "
        f"(character-level UNK {pct(bge_diag.emoji_unk_rate)}), although deleting emoji did not improve R@500.",
        f"8. Recommended next step: **{recommendation}**.", "",
        "## Full BGE Splits", "",
        f"- Overall Recall@100/500: {pct(new['overall']['Recall@100'])} / {pct(new['overall']['Recall@500'])}",
        f"- Train-clicked / train-exposed-never-clicked / cold Recall@500: "
        f"{pct(new['train_clicked']['Recall@500'])} / {pct(new['train_exposed_never_clicked']['Recall@500'])} / "
        f"{pct(new['cold_item']['Recall@500'])}",
        f"- Warm-user / cold-user Recall@500: {pct(new['warm_user']['Recall@500'])} / "
        f"{pct(new['cold_user']['Recall@500'])}",
        f"- Exact query time: {full['search']['query_seconds']:.3f}s; index build: {full['index_build']['build_seconds']:.3f}s.",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for path in (OUT, EMBEDDINGS, INDICES, PER_REQUEST, OUT / "subset_results"):
        path.mkdir(parents=True, exist_ok=True)
    stages = [args.stage] if args.stage != "all" else ["diagnostics", "encode", "evaluate", "report"]
    for stage in stages:
        print(f"\n=== {stage} ===", flush=True)
        if stage == "diagnostics": diagnostics_stage()
        elif stage == "encode": encode_stage(args)
        elif stage == "evaluate": evaluate_stage(args)
        elif stage == "report": report_stage()


if __name__ == "__main__":
    main()
