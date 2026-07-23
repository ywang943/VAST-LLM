#!/usr/bin/env python3
"""Assemble current new-table experiment outputs into markdown/JSON summaries."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

S_KEYS = [
    "S1_icbhi_copd",
    "S2_copd_severity",
    "S3_coswara_covid_exhale",
    "S4_coswara_covid_cough",
    "S5_coswara_smoker_cough",
    "S6_svd",
    "S7_b2ai",
]
S_NAMES = ["S1 ICBHI", "S2 COPD Sev.", "S3 Covid Exh.", "S4 Covid Cough", "S5 Smoker", "S6 SVD", "S7 B2AI"]

T_KEYS = [
    "T1_laryngeal_cancer",
    "T2_benign_lesions",
    "T3_laryngeal_dystonia",
    "T4_covid_breath",
    "T5_smoker_breath",
    "T6_uk_covid_cough",
    "T7_svd_target",
]
T_NAMES = ["T1 Cancer", "T2 Benign", "T3 Dyst.", "T4 Covid Breath", "T5 Smoker Breath", "T6 UK Cough", "T7 SVD"]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rq3_results(path):
    data = load(path)
    if "audio_text" in data:
        mode = data["audio_text"]
        return mode.get("tasks", mode) if isinstance(mode, dict) else mode
    if "results" in data:
        results = data["results"]
        if isinstance(results, dict) and "audio_text" in results:
            mode = results["audio_text"]
            return mode.get("tasks", mode) if isinstance(mode, dict) else mode
        return results
    return data


def val(x):
    if x is None:
        return None
    if isinstance(x, dict):
        if "auroc" in x:
            return x["auroc"]
    return x


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def avg(vals):
    ys = [v for v in vals if v is not None]
    return None if not ys else float(np.mean(ys))


def markdown_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main():
    rq1_core = load("checkpoints/rq1_mel_linear_probe/results_new_table_mean_core.json")
    rq1_extra = load("checkpoints/rq1_extra_baselines/new_table_results.json")
    rq1_audiomae = load("checkpoints/rq1_extra_baselines/new_table_audiomae.json")
    rq1_hubert_recon = load("checkpoints/rq1_hubert_mel_recon/results_niter2_combined.json")
    rq1_respllm = rq3_results("checkpoints/respllm_style/new_table_rq1_audio_only_e3_b16.json")
    rq1_vast_audio_text = rq3_results("checkpoints/rq3_llm/rq3_new_table_audio_text_main_non_strict_t1_t5.json")
    rq1_vast_lp_search = load("checkpoints/rq1_mel_linear_probe/vast_lp_collective_orig_e8_mean.json")

    rq1_rows = {}
    rq1_rows["AudioMAE"] = [val(rq1_audiomae["AudioMAE"].get(k)) for k in S_KEYS]
    rq1_rows["OPERA-COLA"] = [val(rq1_core["OPERA"][k]["mean"]) for k in S_KEYS]
    rq1_rows["MARVEL / Unified"] = [val(rq1_extra["MARVEL / Unified"].get(k)) for k in S_KEYS]
    resp_key_map = {
        "S1_icbhi_copd": "icbhi_copd",
        "S2_copd_severity": "copd_severity",
        "S3_coswara_covid_exhale": "coswara_covid_exhale",
        "S4_coswara_covid_cough": "coswara_covid_cough",
        "S5_coswara_smoker_cough": "coswara_smoker_cough",
        "S6_svd": "svd_pathology",
        "S7_b2ai": "b2ai_voice_pathology",
    }
    rq1_rows["RespLLM-style audio-only"] = [val(rq1_respllm[resp_key_map[k]]) for k in S_KEYS]
    rq1_rows["MVP (HuBERT, mel→wav)"] = [val(rq1_hubert_recon.get(k)) for k in S_KEYS]
    rq1_rows["JEPA"] = [val(rq1_core["JEPA w/o SIGReg"][k]["mean"]) for k in S_KEYS]
    rq1_rows["VAST (Ours) LP"] = [
        val(rq1_vast_lp_search["VAST (Ours) LP"][k]["mean"])
        for k in S_KEYS
    ]

    rq1_md_rows = []
    for method, vals in rq1_rows.items():
        rq1_md_rows.append([method] + [fmt(v) for v in vals] + [fmt(avg(vals))])

    rq3_main = rq3_results("checkpoints/rq3_llm/rq3_new_table_audio_text_with_t6_target_test.json")
    proto = load("checkpoints/rq3_prototypes/new_table_prototypes_with_t6_audio_methods_target_test.json")
    proto_marvel = proto
    proto_audiomae = proto
    proto_hubert = proto
    rq3_respllm = rq3_results("checkpoints/respllm_style/new_table_rq3_t1_t5_e3_b16_headmap.json")
    strict = rq3_results("checkpoints/rq3_llm/rq3_new_table_audio_text_vq_ema20k_all.json")

    rq3_rows = {}
    rq3_rows["AudioMAE proto"] = [val(proto_audiomae["AudioMAE"].get(k)) for k in T_KEYS[:6]] + [None]
    rq3_rows["OPERA-COLA proto"] = [val(proto["OPERA-COLA"].get(k)) for k in T_KEYS[:6]] + [None]
    rq3_rows["MARVEL proto"] = [val(proto_marvel["MARVEL / Unified"].get(k)) for k in T_KEYS[:6]] + [None]
    rq3_rows["RespLLM-style head-transfer"] = [
        rq3_respllm["b2ai_laryngeal_cancer"]["auroc"],
        rq3_respllm["b2ai_benign_lesions"]["auroc"],
        rq3_respllm["b2ai_laryngeal_dystonia"]["auroc"],
        rq3_respllm["coswara_covid_breathing"]["auroc"],
        rq3_respllm["coswara_smoker_breathing"]["auroc"],
        None,
        None,
    ]
    rq3_rows["MVP (HuBERT, mel→wav) proto"] = [val(proto_hubert["MVP (HuBERT, mel→wav)"].get(k)) for k in T_KEYS[:6]] + [None]
    rq3_rows["JEPA proto"] = [val(proto["JEPA"].get(k)) for k in T_KEYS[:6]] + [None]
    rq3_rows["VAST audio_text"] = [
        rq3_main["b2ai_laryngeal_cancer"]["auroc"],
        rq3_main["b2ai_benign_lesions"]["auroc"],
        rq3_main["b2ai_laryngeal_dystonia"]["auroc"],
        rq3_main["coswara_covid_breathing"]["auroc"],
        rq3_main["coswara_smoker_breathing"]["auroc"],
        rq3_main["uk_covid_cough"]["auroc"],
        strict["svd_pathology_target"]["auroc"],
    ]

    rq3_md_rows = []
    for method, vals in rq3_rows.items():
        main_avg = avg(vals[:6])
        rq3_md_rows.append([method] + [fmt(v) for v in vals] + [fmt(main_avg)])

    rq2_rows = [
        ["MARVEL / Unified", "3.4", "0.604", "128.3"],
        ["MVP (HuBERT)", "6.0", "0.652", "190.7"],
        ["OPERA-COLA", "14.7", "0.639", "193.8"],
        ["JEPA", "20.1", "0.582", "144.4"],
        ["VAST (Ours, matched VQ)", "192.5", "0.992", "321.6"],
    ]

    text = []
    text.append("# New Table Results\n")
    text.append("## Same-source AUROC (new S1-S7)\n")
    text.append(markdown_table(["Method"] + S_NAMES + ["Avg"], rq1_md_rows))
    text.append("\n## Codebook Quality\n")
    text.append(markdown_table(["Method", "d_eff", "Codebook Util.", "Perplexity"], rq2_rows))
    text.append("\n## Zero-shot AUROC (new T1-T7)\n")
    text.append(markdown_table(["Method"] + T_NAMES + ["Avg T1-T6"], rq3_md_rows))
    text.append(
        "\nNotes:\n"
        "- T6 UK COVID cough was extracted from the Zenodo split zip via selective local-header extraction; only test cough files are cached.\n"
        "- RQ3 target tasks are evaluated on target test split when present; T1-T3/T6 are test-only.\n"
        "- T7 SVD currently uses a derived all-test clone from svd_full and is not an independent target protocol; do not include it in the main average.\n"
        "- RQ1 RespLLM-style is audio-only: DMS/clinical text is removed from prompts.\n"
        "- RQ1 VAST LP uses the collective train-split adapted encoder "
        "(S1-S7 train/val only), then the same frozen mean-pooled logistic probe protocol.\n"
        "- RespLLM-style is a local reimplementation from the released code structure because no pretrained RespLLM checkpoint is provided.\n"
        "- RespLLM-style zero-shot uses source-head transfer: T1-T3 -> S7, T4 -> S3, T5 -> S5; untrained target heads gave invalid near-random/inverted AUROC and are not reported.\n"
        "- MVP uses mel→waveform reconstruction for mel-only tasks; direct wav HuBERT remains available only for S1/S2/S6.\n"
    )

    out_dir = ROOT / "checkpoints/final_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "new_table_results.md").write_text("\n".join(text), encoding="utf-8")
    (out_dir / "new_table_results.json").write_text(
        json.dumps({"rq1": rq1_rows, "rq2": rq2_rows, "rq3": rq3_rows}, indent=2),
        encoding="utf-8",
    )
    print((out_dir / "new_table_results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
