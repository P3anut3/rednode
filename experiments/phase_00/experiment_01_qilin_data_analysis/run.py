#!/usr/bin/env python3
"""Analyze the local Qilin recommendation parquet files without modifying them.

Run from the project root:
    python experiments/phase_00/experiment_01_qilin_data_analysis/run.py

The implementation intentionally reads notes shard-by-shard and recommendation
rows in batches.  Only compact arrays/aggregates are retained for the 2M-note
catalog; nested candidate records are expanded only for the much smaller
recommendation logs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
RESULT_DIR = ROOT / "results/phase_00/experiment_01_qilin_data_analysis"
OUT = RESULT_DIR / "outputs"
REPORT = RESULT_DIR / "summary.md"
SPLITS = ("recommendation_train", "recommendation_test")
FEEDBACK = ("click", "like", "collect", "comment", "share")
Q_BASIC = (0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
Q_USER = (0.50, 0.75, 0.90, 0.95, 0.99)


def parquet_files(dataset: str) -> list[Path]:
    files = sorted((DATA / dataset).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under data/{dataset}/")
    return files


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def python_type_tree(value: Any) -> Any:
    """Describe actual pandas/Python nesting without assuming Arrow metadata."""
    desc: dict[str, Any] = {"outer": type(value).__name__}
    if isinstance(value, np.ndarray):
        desc["numpy_dtype"] = str(value.dtype)
        if len(value):
            desc["element"] = python_type_tree(value[0])
    elif isinstance(value, (list, tuple)) and value:
        desc["element"] = python_type_tree(value[0])
    elif isinstance(value, dict):
        desc["fields"] = {k: type(v).__name__ for k, v in value.items()}
    return desc


def inspect_schema(dataset: str) -> dict[str, Any]:
    files = parquet_files(dataset)
    row_count = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    first_schema = pq.ParquetFile(files[0]).schema_arrow
    for f in files[1:]:
        if not pq.ParquetFile(f).schema_arrow.equals(first_schema):
            raise ValueError(f"Schema mismatch in {f}")
    # Reading two rows from only the first row group avoids loading a full shard.
    sample_table = pq.ParquetFile(files[0]).read_row_group(0).slice(0, 2)
    sample_df = sample_table.to_pandas()
    nested = {}
    for col in sample_df.columns:
        value = sample_df.iloc[0][col]
        if isinstance(value, (np.ndarray, list, tuple, dict)):
            nested[col] = python_type_tree(value)
    return {
        "shape": [int(row_count), len(first_schema)],
        "files": [str(f.relative_to(ROOT)) for f in files],
        "columns": first_schema.names,
        "arrow_dtypes": {field.name: str(field.type) for field in first_schema},
        "pandas_dtypes": {c: str(t) for c, t in sample_df.dtypes.items()},
        "samples": [jsonable(x) for x in sample_df.to_dict(orient="records")],
        "nested_python_types": nested,
    }


def describe(values: Iterable[Any], quantiles: tuple[float, ...]) -> dict[str, float | int]:
    s = pd.Series(values, dtype="float64").dropna()
    if s.empty:
        return {"count": 0}
    result: dict[str, float | int] = {
        "count": int(s.size),
        "mean": float(s.mean()),
        "min": float(s.min()),
    }
    for q in quantiles:
        result[f"p{int(q * 100):02d}"] = float(s.quantile(q))
    result["max"] = float(s.max())
    return result


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    value = float(value)
    if value.is_integer():
        return f"{int(value):,}"
    return f"{value:,.{digits}f}"


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "（无数据）"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(map(str, cols)) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        cells = []
        for col, value in zip(cols, row):
            if isinstance(value, float):
                col_lower = str(col).lower()
                if any(token in col_lower for token in ("ratio", "rate", "ctr")):
                    cells.append(fmt_pct(value))
                else:
                    cells.append(fmt_num(value))
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def stats_table(named_stats: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for metric, values in named_stats.items():
        rows.append({"metric": metric, **values})
    return pd.DataFrame(rows)


def analyze_recommendation(dataset: str) -> dict[str, Any]:
    request_parts: list[pd.DataFrame] = []
    candidate_parts: list[pd.DataFrame] = []
    history_chunks: list[np.ndarray] = []

    for file in parquet_files(dataset):
        pf = pq.ParquetFile(file)
        columns = [
            "request_idx", "session_idx", "user_idx",
            "recent_clicked_note_idxs", "rec_result_details_with_idx",
        ]
        for batch in pf.iter_batches(batch_size=5_000, columns=columns):
            names = batch.schema.names
            col = {name: batch.column(names.index(name)).to_pylist() for name in names}
            request_rows = []
            candidate_rows = []
            for request_idx, session_idx, user_idx, history, details in zip(
                col["request_idx"], col["session_idx"], col["user_idx"],
                col["recent_clicked_note_idxs"], col["rec_result_details_with_idx"],
            ):
                history = history or []
                details = details or []
                history_arr = np.asarray(history, dtype=np.int64)
                if history_arr.size:
                    history_chunks.append(history_arr)
                clicked_now = {
                    int(x["note_idx"]) for x in details
                    if x.get("note_idx") is not None and (x.get("click") or 0) == 1
                }
                overlap = clicked_now.intersection(map(int, history_arr))
                request_rows.append({
                    "request_idx": int(request_idx),
                    "session_idx": int(session_idx),
                    "user_idx": int(user_idx),
                    "history_length": int(history_arr.size),
                    "candidate_count": len(details),
                    "positive_click_count": sum((x.get("click") or 0) == 1 for x in details),
                    "clicked_note_in_history_count": len(overlap),
                    "has_clicked_note_in_history": bool(overlap),
                })
                for x in details:
                    row = {
                        "request_idx": int(request_idx),
                        "session_idx": int(session_idx),
                        "user_idx": int(user_idx),
                    }
                    row.update(x)
                    candidate_rows.append(row)
            request_parts.append(pd.DataFrame.from_records(request_rows))
            if candidate_rows:
                candidate_parts.append(pd.DataFrame.from_records(candidate_rows))

    requests = pd.concat(request_parts, ignore_index=True)
    candidates = pd.concat(candidate_parts, ignore_index=True)
    for field in FEEDBACK:
        if field not in candidates:
            candidates[field] = 0
        candidates[field] = pd.to_numeric(candidates[field], errors="coerce").fillna(0).astype("int64")
    candidates["note_idx"] = pd.to_numeric(candidates["note_idx"], errors="raise").astype("int64")

    # request_idx should identify a row. Preserve observed row counts but flag violations.
    duplicate_request_rows = int(requests.duplicated("request_idx", keep=False).sum())
    user_stats = requests.groupby("user_idx", sort=False).agg(
        request_count=("request_idx", "nunique"),
        session_count=("session_idx", "nunique"),
        history_length=("history_length", "max"),
        history_length_min=("history_length", "min"),
    )
    user_clicks = candidates.groupby("user_idx")["click"].sum().rename("positive_click_count")
    user_stats = user_stats.join(user_clicks, how="left").fillna({"positive_click_count": 0})
    user_stats["positive_click_count"] = user_stats["positive_click_count"].astype("int64")

    history_all = np.concatenate(history_chunks) if history_chunks else np.array([], dtype=np.int64)
    return {
        "requests": requests,
        "candidates": candidates,
        "users": user_stats,
        "history_all": history_all,
        "history_unique": np.unique(history_all),
        "duplicate_request_rows": duplicate_request_rows,
    }


def positive_histogram(user_stats: pd.DataFrame) -> pd.DataFrame:
    values = user_stats["positive_click_count"]
    bins = [(-np.inf, -1, "invalid"), (0, 0, "0"), (1, 1, "1"), (2, 2, "2"),
            (3, 3, "3"), (4, 4, "4"), (5, 5, "5"), (6, 9, "6-9"),
            (10, 19, "10-19"), (20, 49, "20-49"), (50, 99, "50-99"),
            (100, np.inf, "100+")]
    rows = []
    for low, high, label in bins:
        count = int(((values >= low) & (values <= high)).sum())
        if count or label != "invalid":
            rows.append({"positive_click_bin": label, "users": count, "ratio": count / len(values)})
    return pd.DataFrame(rows)


def item_behavior(candidates: pd.DataFrame) -> pd.DataFrame:
    result = candidates.groupby("note_idx", sort=False).agg(**{
        "exposure_count": ("note_idx", "size"),
        **{f"{field}_count": (field, "sum") for field in FEEDBACK},
    })
    return result.reset_index()


def tail_summary(item: pd.DataFrame, count_col: str, population: str) -> dict[str, Any]:
    positive = item.loc[item[count_col] > 0, count_col]
    denom = len(positive)
    return {
        "population": population,
        "unique_notes": int(denom),
        "count_once": int((positive == 1).sum()),
        "ratio_once": float((positive == 1).mean()) if denom else 0.0,
        "ratio_le_3": float((positive <= 3).mean()) if denom else 0.0,
        "ratio_le_5": float((positive <= 5).mean()) if denom else 0.0,
        "ratio_ge_10": float((positive >= 10).mean()) if denom else 0.0,
        "ratio_ge_100": float((positive >= 100).mean()) if denom else 0.0,
        "total_events": int(positive.sum()),
    }


def valid_string_mask(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return ~normalized.isin(["", "nan", "none", "null", "<na>"])


def analyze_notes() -> dict[str, Any]:
    note_id_parts: list[np.ndarray] = []
    title_lengths: list[np.ndarray] = []
    content_lengths: list[np.ndarray] = []
    combined_lengths: list[np.ndarray] = []
    totals = defaultdict(int)

    columns = ["note_idx", "note_title", "note_content", "taxonomy1_id", "taxonomy2_id", "taxonomy3_id"]
    for file in parquet_files("notes"):
        pf = pq.ParquetFile(file)
        for batch in pf.iter_batches(batch_size=100_000, columns=columns):
            df = batch.to_pandas()
            ids = df["note_idx"].to_numpy(dtype=np.int64, copy=False)
            note_id_parts.append(ids.copy())
            title_ok = valid_string_mask(df["note_title"])
            content_ok = valid_string_mask(df["note_content"])
            taxonomy_masks = [valid_string_mask(df[f"taxonomy{i}_id"]) for i in (1, 2, 3)]
            any_taxonomy = taxonomy_masks[0] | taxonomy_masks[1] | taxonomy_masks[2]
            all_taxonomy = taxonomy_masks[0] & taxonomy_masks[1] & taxonomy_masks[2]
            title_len = df["note_title"].fillna("").astype(str).str.len().to_numpy(dtype=np.int32)
            content_len = df["note_content"].fillna("").astype(str).str.len().to_numpy(dtype=np.int32)
            title_len[~title_ok.to_numpy()] = 0
            content_len[~content_ok.to_numpy()] = 0
            title_lengths.append(title_len)
            content_lengths.append(content_len)
            combined_lengths.append(title_len + content_len)
            totals["rows"] += len(df)
            totals["title_nonempty"] += int(title_ok.sum())
            totals["content_nonempty"] += int(content_ok.sum())
            totals["any_text"] += int((title_ok | content_ok).sum())
            totals["any_taxonomy"] += int(any_taxonomy.sum())
            totals["all_taxonomy"] += int(all_taxonomy.sum())
            for i, mask in enumerate(taxonomy_masks, 1):
                totals[f"taxonomy{i}_nonempty"] += int(mask.sum())

    note_ids = np.concatenate(note_id_parts)
    return {
        "note_ids": note_ids,
        "unique_note_ids": np.unique(note_ids),
        "duplicate_note_ids": int(len(note_ids) - len(np.unique(note_ids))),
        "totals": dict(totals),
        "title_length": describe(np.concatenate(title_lengths), Q_BASIC),
        "content_length": describe(np.concatenate(content_lengths), Q_BASIC),
        "combined_text_length": describe(np.concatenate(combined_lengths), Q_BASIC),
    }


def membership_summary(source: np.ndarray, targets: dict[str, np.ndarray]) -> dict[str, Any]:
    source = np.unique(np.asarray(source, dtype=np.int64))
    result: dict[str, Any] = {"unique_source": int(len(source))}
    for name, target in targets.items():
        count = int(np.isin(source, np.asarray(target, dtype=np.int64), assume_unique=False).sum())
        result[f"in_{name}_count"] = count
        result[f"in_{name}_ratio"] = count / len(source) if len(source) else 0.0
    return result


def save_outputs(
    schemas: dict[str, Any], rec: dict[str, dict[str, Any]], notes: dict[str, Any],
    train_items: pd.DataFrame, tables: dict[str, pd.DataFrame], summary: dict[str, Any],
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "schema_and_samples.json").write_text(
        json.dumps(schemas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_items.to_parquet(OUT / "train_item_behavior.parquet", index=False)
    rec["recommendation_train"]["users"].reset_index().to_parquet(
        OUT / "train_user_behavior.parquet", index=False
    )
    for dataset in SPLITS:
        short = dataset.removeprefix("recommendation_")
        rec[dataset]["requests"].to_parquet(OUT / f"{short}_request_behavior.parquet", index=False)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)


def build_report(
    schemas: dict[str, Any], rec: dict[str, dict[str, Any]], notes: dict[str, Any],
    train_items: pd.DataFrame, summary: dict[str, Any], tables: dict[str, pd.DataFrame],
) -> str:
    train = rec["recommendation_train"]
    test = rec["recommendation_test"]
    s_train = summary["splits"]["recommendation_train"]
    s_test = summary["splits"]["recommendation_test"]
    n_total = notes["totals"]["rows"]
    click_tail = summary["train_item_tail"]["click"]
    exp_tail = summary["train_item_tail"]["exposure"]
    cold = summary["cold_start"]
    hist = summary["history_coverage"]
    content = summary["notes_content"]

    lines = [
        "# Qilin Recommendation Data Analysis",
        "",
        "> 本报告由 `experiments/phase_00/experiment_01_qilin_data_analysis/run.py` 直接读取本地 parquet 生成。"
        " 所有覆盖率均以当前文件实际内容为准；未使用 README 假定。",
        "",
        "## 1. Real Schema Inspection",
        "",
    ]
    for dataset in ("recommendation_train", "recommendation_test", "notes", "user_feat"):
        info = schemas[dataset]
        lines += [
            f"### `{dataset}`",
            "",
            f"- shape: `{tuple(info['shape'])}`",
            f"- columns: `{info['columns']}`",
            f"- Arrow dtype: `{info['arrow_dtypes']}`",
            f"- pandas dtype（前两行转换后）: `{info['pandas_dtypes']}`",
            f"- 嵌套字段真实 Python 类型: `{info['nested_python_types']}`",
            "- 前 2 条样本：",
            "",
            "```json",
            json.dumps(info["samples"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    lines += [
        "实际候选结构确认：`rec_result_details_with_idx` 在 pandas 中为 `numpy.ndarray`，元素为 `dict`；"
        "dict 字段是 `click, collect, comment, like, note_idx, page_time, position, request_timestamp, share`。",
        "",
        "## 2. Recommendation Train / Test Scale",
        "",
        markdown_table(tables["split_scale"]),
        "",
        "候选数量分布（按 recommendation 文件的每一行/request）：",
        "",
        markdown_table(tables["candidate_count_distribution"]),
        "",
        f"`request_idx` 重复行检查：train={train['duplicate_request_rows']}，test={test['duplicate_request_rows']}。",
        "",
        "## 3. User Behavior Sparsity (Train)",
        "",
        "`history_length` 取同一用户在训练 request 中观测到的最大历史长度；另检查了用户内最小/最大是否一致。",
        "",
        markdown_table(tables["user_metric_distribution"]),
        "",
        markdown_table(tables["user_positive_thresholds"]),
        "",
        "用户正样本数量直方图：",
        "",
        markdown_table(tables["user_positive_histogram"]),
        "",
        "## 4. Positive Feedback per Request",
        "",
        markdown_table(tables["request_positive_distribution"]),
        "",
        "每个 request 的 click=1 数量分布：",
        "",
        markdown_table(tables["request_click_count_stats"]),
        "",
        markdown_table(tables["feedback_rates"]),
        "",
        "`like/collect/comment/share` 这里只作为观测分布，未定义为训练 label。",
        "",
        "## 5. Item Behavior Long Tail (Train)",
        "",
        markdown_table(tables["item_tail_summary"]),
        "",
        f"- 全库 note 行数 / unique note：{n_total:,} / {len(notes['unique_note_ids']):,}。",
        f"- 训练曝光 unique note / 全库：{len(train_items):,} / {n_total:,} = {fmt_pct(len(train_items)/n_total)}。",
        f"- 训练点击 unique note / 全库：{click_tail['unique_notes']:,} / {n_total:,} = {fmt_pct(click_tail['unique_notes']/n_total)}。",
        f"- 训练候选曝光总次数：{len(train['candidates']):,}；点击总次数：{int(train['candidates']['click'].sum()):,}。",
        "",
        "比率分母说明：click 长尾以“至少被点击过一次的 note”为分母；exposure 长尾以“至少曝光过一次的 note”为分母。",
        "这直接刻画纯 ID item embedding 能获得监督的 item 范围。",
        "",
        "## 6. Train → Test Cold Start",
        "",
        markdown_table(tables["cold_start"]),
        "",
        f"test 正点击交互共 {cold['test_positive_interactions']:,} 条，其中 train 从未曝光 item 上的正点击 "
        f"{cold['cold_positive_interactions']:,} 条（{fmt_pct(cold['cold_positive_interaction_ratio'])}）。",
        "这里的 cold item 定义为“在 train recommendation 候选曝光中未出现”，而不是“不在 notes 全库”。",
        "",
        "## 7. `recent_clicked_note_idxs` Coverage and Leakage Check",
        "",
        markdown_table(tables["history_coverage"]),
        "",
        "历史序列长度：",
        "",
        markdown_table(tables["history_length_distribution"]),
        "",
        markdown_table(tables["history_overlap"]),
        "",
        "若当前 request 的 clicked note 已在 history 中，这可能只是重复消费，也可能说明历史快照在当前行为之后构造。"
        "数据没有历史事件时间戳，因而不能仅凭重合证明泄漏；但在建模前必须按 request timestamp 重新确认历史截断逻辑。",
        "",
        "## 8. Position Bias",
        "",
        markdown_table(tables["position_first20"]),
        "",
        summary["position_bias_assessment"],
        "这里只描述相关性，不做因果校正。",
        "",
        "## 9. Notes Content Availability",
        "",
        markdown_table(tables["notes_availability"]),
        "",
        "文本字符长度分布（空值按 0）：",
        "",
        markdown_table(tables["text_length_distribution"]),
        "",
        "- Content retriever 输入：优先拼接 `note_title` + `note_content`；二者是实际自然语言字段。"
        " `content_length` 是数值统计而不是文本正文。",
        "- Item-side categorical feature：`taxonomy1_id/taxonomy2_id/taxonomy3_id`（层级类目）、"
        "`note_type`、`commercial_flag`；图像/视频元数据可作数值或模态辅助特征。",
        "- `image_path` 是路径列表，不是可直接输入文本编码器的内容。",
        "- 空值判断同时把空串及字符串哨兵 `nan/none/null/<na>` 视为空；当前样本中 taxonomy 确有字符串 `nan`。",
        "",
        "## Data Quality / Anomaly Notes",
        "",
        f"- notes duplicate `note_idx`: {notes['duplicate_note_ids']:,}。",
        f"- train/test duplicate request rows: {train['duplicate_request_rows']:,} / {test['duplicate_request_rows']:,}。",
        f"- train/test 候选 note 不在 notes 全库的 unique 数："
        f"{summary['catalog']['train_candidate_missing_catalog']:,} / "
        f"{summary['catalog']['test_candidate_missing_catalog']:,}。",
        "- train 与 test 的 0-click request 都是 0；数据显然不是包含自然零反馈流量的完整曝光日志，"
        "很可能按至少一个正反馈筛选过。候选内反馈率可用于描述当前样本，但不能直接当作线上 CTR，"
        "负例构造与离线评估需明确这一选择偏差。",
        f"- train 中同一用户 history 长度发生变化的用户数：{summary['history_user_length_varies_train']:,}。",
        f"- train 当前点击与 history 重合的 request：{summary['leakage']['train_overlap_requests']:,} "
        f"（{fmt_pct(summary['leakage']['train_overlap_request_ratio'])}）；"
        f"test 为 {summary['leakage']['test_overlap_requests']:,} "
        f"（{fmt_pct(summary['leakage']['test_overlap_request_ratio'])}）。",
        "",
        "## Retrieval Feasibility Assessment",
        "",
        f"1. **纯 `user_id -> item_id` ID-based DSSM：不适合作为全库主方案。** "
        f"训练点击只覆盖全库 {fmt_pct(click_tail['unique_notes']/n_total)} 的 item，"
        f"且用户正点击中位数为 {fmt_num(s_train['user_metrics']['positive_click_count']['p50'])}。"
        "模型可以做已见用户/已见 item 的实验基线，但未获点击监督的绝大多数 item embedding 无法得到可靠学习。",
        f"2. **点击监督不足以覆盖约 200 万 item。** 被点击 unique note 为 {click_tail['unique_notes']:,}，"
        f"仅占全库 {fmt_pct(click_tail['unique_notes']/n_total)}；即使按曝光算，也只覆盖 {fmt_pct(len(train_items)/n_total)}。",
        f"3. **ItemCF / UserCF 的最大瓶颈是交互矩阵极稀疏和 item 长尾。** "
        f"被点击 item 中 {fmt_pct(click_tail['ratio_once'])} 只点击 1 次；"
        f"用户正点击 p50={fmt_num(s_train['user_metrics']['positive_click_count']['p50'])}。"
        "ItemCF 共现边弱，UserCF 用户重叠更弱且计算/邻居稳定性差；此外所有 request 都带正反馈，"
        "无法从本日志估计自然零点击流量。",
        f"4. **Content-based retrieval 是必要的。** 全库有文本比例为 {fmt_pct(content['any_text_ratio'])}，"
        "它能为无推荐曝光/无点击的 item 产生向量，并服务 cold item；taxonomy 可补充类别语义。",
        "5. **Hybrid retrieval 比纯 ID retrieval 更合理。** 内容塔保证全库和 cold-item 可编码，"
        "ID/行为特征用于已见 item 与活跃用户的个性化；Popularity/ItemCF 可作为独立召回通道或融合特征。",
        f"6. **test cold item 会影响召回评估。** test 正点击中 {fmt_pct(cold['cold_positive_interaction_ratio'])} "
        "落在 train 未曝光 item；若比例高，纯 ID 方法的总体 Recall 上限会被显著压低，应同时报告 warm/cold 分层指标。",
        "7. **可构造能力：** Popularity baseline=足够；ItemCF=可构造但长尾严重；UserCF=可构造但预计最稀疏；"
        "content-based bi-encoder=足够（title/content + taxonomy）；hybrid DSSM=特征足够但不应只靠 ID；"
        "Faiss 全库 ANN retrieval=数据层面足够（全库 note_idx 与高覆盖文本可建索引），但本报告未训练模型或建索引。",
        "8. **第一版召回实验建议：** Random（校验）、Global Popularity、Category-aware Popularity、"
        "ItemCF（recent clicks）、content BM25/TF-IDF、content bi-encoder ANN、"
        "以及 content + popularity + ItemCF 的 hybrid。统一报告 Recall@K/HitRate@K，"
        "并按 warm item、train-unexposed cold item、warm/cold user 分层；纯 ID two-tower 只作为受限对照。",
        "",
        "### Overall judgment",
        "",
        "当前 Qilin 数据足以设计并评估“面向约 200 万 notes 的全库召回”，也足以建立多类 baseline；"
        "但它不支持把纯 ID-based retrieval 当作覆盖全库的充分方案。决定性约束是训练推荐日志对全库的曝光/点击覆盖、"
        "item 点击长尾、用户正反馈稀疏以及 test cold item。第一版应以可覆盖全库的内容召回为骨架，"
        "再融合 popularity 与协同信号，并进行 cold/warm 分层评估。",
        "",
        "## Reproducibility Outputs",
        "",
        "- `results/phase_00/experiment_01_qilin_data_analysis/outputs/schema_and_samples.json`",
        "- `results/phase_00/experiment_01_qilin_data_analysis/outputs/summary.json`",
        "- `results/phase_00/experiment_01_qilin_data_analysis/outputs/train_item_behavior.parquet`",
        "- `results/phase_00/experiment_01_qilin_data_analysis/outputs/train_user_behavior.parquet`",
        "- `results/phase_00/experiment_01_qilin_data_analysis/outputs/train_request_behavior.parquet` / `test_request_behavior.parquet`",
        "- `results/phase_00/experiment_01_qilin_data_analysis/outputs/*.csv`（报告各统计表）",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    schemas = {name: inspect_schema(name) for name in (*SPLITS, "notes", "user_feat")}
    rec = {name: analyze_recommendation(name) for name in SPLITS}
    train = rec["recommendation_train"]
    test = rec["recommendation_test"]
    notes = analyze_notes()
    train_items = item_behavior(train["candidates"])

    train_exposed = train_items["note_idx"].to_numpy(dtype=np.int64)
    train_clicked = train_items.loc[train_items["click_count"] > 0, "note_idx"].to_numpy(dtype=np.int64)
    test_exposed = np.unique(test["candidates"]["note_idx"].to_numpy(dtype=np.int64))
    test_clicked = np.unique(test["candidates"].loc[test["candidates"]["click"] > 0, "note_idx"].to_numpy(dtype=np.int64))
    note_ids = notes["unique_note_ids"]

    split_rows = []
    candidate_dist_rows = []
    split_summary: dict[str, Any] = {}
    for name in SPLITS:
        data = rec[name]
        req, cand, users = data["requests"], data["candidates"], data["users"]
        split_rows.append({
            "split": name, "request_rows": len(req), "unique_requests": req["request_idx"].nunique(),
            "unique_users": req["user_idx"].nunique(), "unique_sessions": req["session_idx"].nunique(),
            "candidate_exposures": len(cand), "unique_candidate_notes": cand["note_idx"].nunique(),
        })
        candidate_stats = describe(req["candidate_count"], Q_BASIC)
        candidate_dist_rows.append({"split": name, **candidate_stats})
        split_summary[name] = {
            "scale": split_rows[-1],
            "candidate_count": candidate_stats,
            "user_metrics": {
                metric: describe(users[metric], Q_USER)
                for metric in ("request_count", "session_count", "history_length", "positive_click_count")
            },
        }

    user_metric_rows = []
    for metric, values in split_summary["recommendation_train"]["user_metrics"].items():
        user_metric_rows.append({"metric": metric, **values})
    pos = train["users"]["positive_click_count"]
    threshold_table = pd.DataFrame([
        {"condition": "== 1", "users": int((pos == 1).sum()), "ratio": float((pos == 1).mean())},
        {"condition": "<= 3", "users": int((pos <= 3).sum()), "ratio": float((pos <= 3).mean())},
        {"condition": "<= 5", "users": int((pos <= 5).sum()), "ratio": float((pos <= 5).mean())},
        {"condition": ">= 10", "users": int((pos >= 10).sum()), "ratio": float((pos >= 10).mean())},
    ])

    request_positive_rows = []
    request_click_stat_rows = []
    for name in SPLITS:
        clicks = rec[name]["requests"]["positive_click_count"]
        request_click_stat_rows.append({"split": name, **describe(clicks, Q_BASIC)})
        request_positive_rows += [
            {"split": name, "bucket": "0 click", "requests": int((clicks == 0).sum()), "ratio": float((clicks == 0).mean())},
            {"split": name, "bucket": "1 click", "requests": int((clicks == 1).sum()), "ratio": float((clicks == 1).mean())},
            {"split": name, "bucket": "2+ click", "requests": int((clicks >= 2).sum()), "ratio": float((clicks >= 2).mean())},
        ]
    feedback_rows = []
    for name in SPLITS:
        cand = rec[name]["candidates"]
        for field in FEEDBACK:
            positives = int(cand[field].sum())
            feedback_rows.append({
                "split": name, "feedback": field, "positive_interactions": positives,
                "positive_rate_per_exposure": positives / len(cand),
                "requests_with_positive": int(cand.loc[cand[field] > 0, "request_idx"].nunique()),
            })

    click_tail = tail_summary(train_items, "click_count", "clicked notes")
    exposure_tail = tail_summary(train_items, "exposure_count", "exposed notes")
    tail_table = pd.DataFrame([
        {"signal": "click", **click_tail}, {"signal": "exposure", **exposure_tail}
    ])

    clicked_membership = membership_summary(test_clicked, {
        "train_exposure": train_exposed, "train_click": train_clicked, "notes_catalog": note_ids,
    })
    exposure_membership = membership_summary(test_exposed, {
        "train_exposure": train_exposed, "train_click": train_clicked, "notes_catalog": note_ids,
    })
    train_users = train["requests"]["user_idx"].unique()
    test_users = test["requests"]["user_idx"].unique()
    seen_test_users = int(np.isin(test_users, train_users).sum())
    positive_test = test["candidates"].loc[test["candidates"]["click"] > 0]
    cold_positive = int((~positive_test["note_idx"].isin(train_exposed)).sum())
    cold = {
        "test_clicked_unique": len(test_clicked),
        "clicked_in_train_exposure_count": clicked_membership["in_train_exposure_count"],
        "clicked_in_train_exposure_ratio": clicked_membership["in_train_exposure_ratio"],
        "clicked_in_train_click_count": clicked_membership["in_train_click_count"],
        "clicked_in_train_click_ratio": clicked_membership["in_train_click_ratio"],
        "clicked_never_in_train_count": len(test_clicked) - clicked_membership["in_train_exposure_count"],
        "clicked_never_in_train_ratio": 1 - clicked_membership["in_train_exposure_ratio"],
        "test_exposed_unique": len(test_exposed),
        "exposed_in_train_exposure_ratio": exposure_membership["in_train_exposure_ratio"],
        "exposed_never_in_train_ratio": 1 - exposure_membership["in_train_exposure_ratio"],
        "test_users": len(test_users),
        "users_seen_in_train_count": seen_test_users,
        "users_seen_in_train_ratio": seen_test_users / len(test_users),
        "cold_users_count": len(test_users) - seen_test_users,
        "cold_users_ratio": 1 - seen_test_users / len(test_users),
        "test_positive_interactions": len(positive_test),
        "cold_positive_interactions": cold_positive,
        "cold_positive_interaction_ratio": cold_positive / len(positive_test),
    }
    cold_table = pd.DataFrame([
        {"entity": "test clicked unique note", "condition": "in train exposure", "count": cold["clicked_in_train_exposure_count"], "ratio": cold["clicked_in_train_exposure_ratio"]},
        {"entity": "test clicked unique note", "condition": "in train click", "count": cold["clicked_in_train_click_count"], "ratio": cold["clicked_in_train_click_ratio"]},
        {"entity": "test clicked unique note", "condition": "never in train recommendation", "count": cold["clicked_never_in_train_count"], "ratio": cold["clicked_never_in_train_ratio"]},
        {"entity": "test exposure unique note", "condition": "in train exposure", "count": int(exposure_membership["in_train_exposure_count"]), "ratio": cold["exposed_in_train_exposure_ratio"]},
        {"entity": "test exposure unique note", "condition": "never in train recommendation", "count": len(test_exposed)-int(exposure_membership["in_train_exposure_count"]), "ratio": cold["exposed_never_in_train_ratio"]},
        {"entity": "test user", "condition": "seen in train", "count": seen_test_users, "ratio": cold["users_seen_in_train_ratio"]},
        {"entity": "test user", "condition": "completely cold", "count": cold["cold_users_count"], "ratio": cold["cold_users_ratio"]},
    ])

    history_rows = []
    history_summary: dict[str, Any] = {}
    for name in SPLITS:
        coverage = membership_summary(rec[name]["history_unique"], {
            "notes_catalog": note_ids, "train_exposure": train_exposed, "train_click": train_clicked,
        })
        history_summary[name] = coverage
        history_rows.append({
            "split": name, "history_occurrences": len(rec[name]["history_all"]),
            "unique_history_notes": coverage["unique_source"],
            "in_notes_count": coverage["in_notes_catalog_count"], "in_notes_ratio": coverage["in_notes_catalog_ratio"],
            "in_train_exposure_count": coverage["in_train_exposure_count"], "in_train_exposure_ratio": coverage["in_train_exposure_ratio"],
            "in_train_click_count": coverage["in_train_click_count"], "in_train_click_ratio": coverage["in_train_click_ratio"],
        })
    history_len_rows = []
    overlap_rows = []
    leakage: dict[str, Any] = {}
    for name in SPLITS:
        req = rec[name]["requests"]
        history_len_rows.append({"split": name, **describe(req["history_length"], Q_BASIC)})
        overlap_count = int(req["has_clicked_note_in_history"].sum())
        overlap_rows.append({
            "split": name, "requests_with_current_click_in_history": overlap_count,
            "ratio_all_requests": overlap_count / len(req),
            "ratio_positive_requests": overlap_count / max(1, int((req["positive_click_count"] > 0).sum())),
            "overlapping_unique_clicks_sum": int(req["clicked_note_in_history_count"].sum()),
        })
        key = "train" if name.endswith("train") else "test"
        leakage[f"{key}_overlap_requests"] = overlap_count
        leakage[f"{key}_overlap_request_ratio"] = overlap_count / len(req)

    position = train["candidates"].groupby("position", dropna=False).agg(
        exposure=("note_idx", "size"), click=("click", "sum")
    ).reset_index().sort_values("position")
    position["ctr"] = position["click"] / position["exposure"]
    position_first20 = position.head(20).copy()
    valid_pos = position.dropna(subset=["position"])
    if len(valid_pos) >= 2:
        first_ctr = float(valid_pos.iloc[0]["ctr"])
        twentieth_ctr = float(valid_pos.iloc[min(19, len(valid_pos)-1)]["ctr"])
        ratio = first_ctr / twentieth_ctr if twentieth_ctr > 0 else float("inf")
        pos_assessment = (
            f"首位 CTR={fmt_pct(first_ctr)}，第 {int(valid_pos.iloc[min(19, len(valid_pos)-1)]['position'])} 位 "
            f"CTR={fmt_pct(twentieth_ctr)}，首位约为后者 {fmt_num(ratio, 2)} 倍。"
            + ("CTR 随靠后位置总体明显降低，存在明显 position bias。" if ratio >= 1.5 else "首尾差异有限，未见强烈单调位置偏差。")
        )
    else:
        pos_assessment = "position 有效取值不足，无法判断位置偏差。"

    nt = notes["totals"]
    content_summary = {
        "total_notes": nt["rows"],
        "title_nonempty_ratio": nt["title_nonempty"] / nt["rows"],
        "content_nonempty_ratio": nt["content_nonempty"] / nt["rows"],
        "any_text_ratio": nt["any_text"] / nt["rows"],
        "any_taxonomy_ratio": nt["any_taxonomy"] / nt["rows"],
        "all_taxonomy_ratio": nt["all_taxonomy"] / nt["rows"],
    }
    train_candidate_missing_catalog = int((~np.isin(train_exposed, note_ids)).sum())
    test_candidate_missing_catalog = int((~np.isin(test_exposed, note_ids)).sum())
    notes_availability = pd.DataFrame([
        {"field/condition": "note_title nonempty", "count": nt["title_nonempty"], "ratio": nt["title_nonempty"] / nt["rows"]},
        {"field/condition": "note_content nonempty", "count": nt["content_nonempty"], "ratio": nt["content_nonempty"] / nt["rows"]},
        {"field/condition": "title OR content nonempty", "count": nt["any_text"], "ratio": nt["any_text"] / nt["rows"]},
        {"field/condition": "any taxonomy level", "count": nt["any_taxonomy"], "ratio": nt["any_taxonomy"] / nt["rows"]},
        {"field/condition": "all 3 taxonomy levels", "count": nt["all_taxonomy"], "ratio": nt["all_taxonomy"] / nt["rows"]},
        *[{"field/condition": f"taxonomy{i}_id nonempty", "count": nt[f"taxonomy{i}_nonempty"], "ratio": nt[f"taxonomy{i}_nonempty"] / nt["rows"]} for i in (1, 2, 3)],
    ])
    text_lengths = pd.DataFrame([
        {"field": "note_title", **notes["title_length"]},
        {"field": "note_content", **notes["content_length"]},
        {"field": "title+content", **notes["combined_text_length"]},
    ])

    summary = {
        "splits": split_summary,
        "train_user_positive_thresholds": threshold_table.to_dict(orient="records"),
        "train_item_tail": {"click": click_tail, "exposure": exposure_tail},
        "catalog": {
            "rows": nt["rows"], "unique_notes": len(note_ids),
            "train_exposed_unique": len(train_items), "train_clicked_unique": len(train_clicked),
            "train_exposed_catalog_ratio": len(train_items) / nt["rows"],
            "train_clicked_catalog_ratio": len(train_clicked) / nt["rows"],
            "train_candidate_missing_catalog": train_candidate_missing_catalog,
            "test_candidate_missing_catalog": test_candidate_missing_catalog,
        },
        "cold_start": cold,
        "history_coverage": history_summary,
        "leakage": leakage,
        "history_user_length_varies_train": int((train["users"]["history_length"] != train["users"]["history_length_min"]).sum()),
        "position_bias_assessment": pos_assessment,
        "notes_content": content_summary,
    }
    tables = {
        "split_scale": pd.DataFrame(split_rows),
        "candidate_count_distribution": pd.DataFrame(candidate_dist_rows),
        "user_metric_distribution": pd.DataFrame(user_metric_rows),
        "user_positive_thresholds": threshold_table,
        "user_positive_histogram": positive_histogram(train["users"]),
        "request_positive_distribution": pd.DataFrame(request_positive_rows),
        "request_click_count_stats": pd.DataFrame(request_click_stat_rows),
        "feedback_rates": pd.DataFrame(feedback_rows),
        "item_tail_summary": tail_table,
        "cold_start": cold_table,
        "history_coverage": pd.DataFrame(history_rows),
        "history_length_distribution": pd.DataFrame(history_len_rows),
        "history_overlap": pd.DataFrame(overlap_rows),
        "position_first20": position_first20,
        "position_all": position,
        "notes_availability": notes_availability,
        "text_length_distribution": text_lengths,
    }
    save_outputs(schemas, rec, notes, train_items, tables, summary)
    REPORT.write_text(build_report(schemas, rec, notes, train_items, summary, tables), encoding="utf-8")
    print(f"Wrote report: {REPORT.relative_to(ROOT)}")
    print(f"Wrote intermediate outputs: {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    argparse.ArgumentParser(description="Analyze the local Qilin recommendation parquet datasets.").parse_args()
    main()
