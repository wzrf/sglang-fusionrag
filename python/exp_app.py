#!/usr/bin/env python3
"""

"""

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

from utils import compute_f1, _exact_match_score

JUDGE_CACHE: Dict[str, Dict[str, Any]] = {}
JUDGE_CACHE_FILE: Optional[str] = None

@dataclass
class Sample:
    sample_id: int
    level: str  # main / sub
    main_id: int
    sub_id: Optional[int]
    main_question: str
    question: str
    answer: str
    doc_ids: List[int]
    should_test: bool


class DocumentPoolLoader:
    def __init__(self, pool_path: str):
        if not os.path.exists(pool_path):
            raise FileNotFoundError(f"Document pool not found: {pool_path}")
        self.pool: Dict[int, str] = {}
        with open(pool_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        for doc in docs:
            if "id" in doc and "text" in doc:
                self.pool[int(doc["id"])] = doc["text"]

    def get_document(self, doc_id: int) -> str:
        if doc_id not in self.pool:
            raise ValueError(f"Document ID {doc_id} not found in pool")
        return self.pool[doc_id]


def normalize_sub_question(text: str) -> str:
    text = text.strip()
    if text.startswith("Intermediate query"):
        p = text.find(":")
        if p != -1:
            return text[p + 1 :].strip()
    return text


def normalize_sub_answer(text: str) -> str:
    text = text.strip()
    if text.startswith("Intermediate answer"):
        p = text.find(":")
        if p != -1:
            return text[p + 1 :].strip()
    return text


def _to_int_doc_ids(values: Any) -> List[int]:
    out: List[int] = []
    if not isinstance(values, list):
        return out
    for x in values:
        try:
            out.append(int(x))
        except Exception:
            continue
    return out


def _is_untestable_sub_answer(ans: str) -> bool:
    s = (ans or "").strip()
    return ("No relevant information found" in s) or ("没有相关信息" in s)


def load_dataset_samples(
    data_path: str,
    max_samples: Optional[int] = None,
    eval_level: str = "both",
    skip_untestable: bool = True,
    respect_llm_judge: bool = False,
    use_supported_docs: bool = False,
) -> List[Sample]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Sample] = []
    sid = 0
    for main_id, item in enumerate(data):
        main_q = item.get("question", "")
        sub_items = item.get("intermediate_context", [])

        should_test_main = True
        if respect_llm_judge and item.get("llm_judge", True) is False:
            should_test_main = False
        for sub in sub_items:
            ans = normalize_sub_answer(sub.get("answer", ""))
            if _is_untestable_sub_answer(ans):
                should_test_main = False
                break

        if skip_untestable and not should_test_main:
            continue

        # Gather a question-level doc set, consistent with original logic.
        main_doc_ids: List[int] = []
        seen_main_doc_ids = set()
        for sub in sub_items:
            if use_supported_docs:
                raw_ids = sub.get("retrieve docs supported", [])
            else:
                raw_ids = sub.get("retrieve docs", [])
            for d in _to_int_doc_ids(raw_ids):
                if d not in seen_main_doc_ids:
                    seen_main_doc_ids.add(d)
                    main_doc_ids.append(d)

        # Main-question sample
        if eval_level in ("both", "main"):
            if not main_doc_ids:
                main_doc_ids = _to_int_doc_ids(item.get("retrieved_results", []))
            samples.append(
                Sample(
                    sample_id=sid,
                    level="main",
                    main_id=main_id,
                    sub_id=None,
                    main_question=main_q,
                    question=main_q,
                    answer=(item.get("answer", "") or "").strip(),
                    doc_ids=main_doc_ids,
                    should_test=should_test_main,
                )
            )
            sid += 1
            if max_samples is not None and len(samples) >= max_samples:
                return samples

        # Sub-question samples
        if eval_level in ("both", "sub"):
            for sub_id, sub in enumerate(sub_items):
                if use_supported_docs:
                    doc_ids = _to_int_doc_ids(sub.get("retrieve docs supported", []))
                else:
                    doc_ids = _to_int_doc_ids(sub.get("retrieve docs", []))
                samples.append(
                    Sample(
                        sample_id=sid,
                        level="sub",
                        main_id=main_id,
                        sub_id=sub_id,
                        main_question=main_q,
                        question=normalize_sub_question(sub.get("query", "")),
                        answer=normalize_sub_answer(sub.get("answer", "")),
                        doc_ids=doc_ids,
                        should_test=should_test_main,
                    )
                )
                sid += 1
                if max_samples is not None and len(samples) >= max_samples:
                    return samples
    return samples


def _post_completion(base_url: str, payload: Dict[str, Any]) -> requests.Response:
    return requests.post(
        f"{base_url}/v1/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
    )


def _md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _cache_file_paths(cache_dir: str, doc_text: str) -> Tuple[str, str]:
    h = _md5_text(doc_text)
    folder = os.path.join(cache_dir, h)
    return os.path.join(folder, "metadata.json"), os.path.join(folder, f"{h}.pt")


def has_doc_cache(cache_dir: str, doc_text: str) -> bool:
    meta, tensor = _cache_file_paths(cache_dir, doc_text)
    return os.path.exists(meta) and os.path.exists(tensor)


def build_cache_for_doc(
    base_url: str,
    model: str,
    doc_text: str,
    preprocess: bool,
    preprocess_prefix: str,
    max_tokens: int,
) -> bool:
    prefix_prompt = preprocess_prefix if preprocess else ""
    prompt = prefix_prompt + doc_text
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "fusionrag_params": {
            "save_cache": True,
            "preprocess": preprocess,
            "save_preprocess_cache": preprocess,
            "prefix_prompt": prefix_prompt,
        },
    }
    resp = _post_completion(base_url, payload)
    if resp.status_code != 200:
        print(f"[FAIL] build_cache status={resp.status_code}")
        print(resp.text)
        return False
    return True


def infer_answer(
    base_url: str,
    model: str,
    system_prompt: str,
    question: str,
    docs: List[str],
    max_tokens: int,
    cache_source: str,
    match_prefix_mode: str,
) -> str:
    docs_joined = "".join(docs)
    prompt = f"{system_prompt}\n{docs_joined}\nQuestion：{question}\nAnswer："

    if match_prefix_mode == "docs_only":
        prefix_prompt = docs_joined
    else:
        prefix_prompt = f"{system_prompt}\n{docs_joined}"

    fusionrag_params: Dict[str, Any] = {
        "save_cache": False,
        "prefix_prompt": prefix_prompt,
        "recompute_debug": False,
    }
    # Explicitly force cache source on inference.
    # preprocess=True  -> use preprocess cache only
    # preprocess=False -> use raw cache only
    if cache_source == "preprocess":
        fusionrag_params["preprocess"] = True
    elif cache_source == "raw":
        fusionrag_params["preprocess"] = False
    else:
        raise ValueError(f"Unsupported cache_source={cache_source}, expected preprocess/raw")

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "fusionrag_params": fusionrag_params,
    }
    resp = _post_completion(base_url, payload)
    if resp.status_code != 200:
        raise RuntimeError(f"Inference failed: status={resp.status_code}, body={resp.text}")

    body = resp.json()
    if not body.get("choices"):
        return ""
    return (body["choices"][0].get("text") or "").strip()


def _load_judge_cache(cache_root: str) -> None:
    global JUDGE_CACHE, JUDGE_CACHE_FILE
    os.makedirs(cache_root, exist_ok=True)
    JUDGE_CACHE_FILE = os.path.join(cache_root, "judge_cache_dataset_pipeline.json")
    if os.path.exists(JUDGE_CACHE_FILE):
        with open(JUDGE_CACHE_FILE, "r", encoding="utf-8") as f:
            JUDGE_CACHE = json.load(f)


def _save_judge_cache() -> None:
    if JUDGE_CACHE_FILE is None:
        return
    with open(JUDGE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(JUDGE_CACHE, f, ensure_ascii=False, indent=2)


def _derive_accuracy_path(output_json: str) -> str:
    base, ext = os.path.splitext(output_json)
    if ext.lower() == ".json":
        return f"{base}.accuracy.json"
    return f"{output_json}.accuracy.json"


def _write_outputs(
    rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_csv: str,
    output_json: str,
    output_acc: str,
) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "level",
                "main_id",
                "sub_id",
                "should_test",
                "main_question",
                "question",
                "ground_truth",
                "predicted",
                "doc_ids",
                "f1",
                "em",
                "judge_correct",
                "judge_reason",
            ],
        )
        writer.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["doc_ids"] = json.dumps(r2["doc_ids"], ensure_ascii=False)
            writer.writerow(r2)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    acc_only = {
        "num_samples": summary.get("num_samples", 0),
        "avg_f1": summary.get("avg_f1", 0.0),
        "avg_em": summary.get("avg_em", 0.0),
        "main_avg_f1": summary.get("main_avg_f1", 0.0),
        "main_avg_em": summary.get("main_avg_em", 0.0),
        "sub_avg_f1": summary.get("sub_avg_f1", 0.0),
        "sub_avg_em": summary.get("sub_avg_em", 0.0),
        "strict_main_total": summary.get("strict_main_total", 0),
        "strict_main_correct": summary.get("strict_main_correct", 0),
        "strict_main_acc": summary.get("strict_main_acc", 0.0),
        "build_cache_source": summary.get("build_cache_source"),
        "inference_cache_source": summary.get("inference_cache_source"),
        "eval_level": summary.get("eval_level"),
        "skip_untestable": summary.get("skip_untestable"),
        "respect_llm_judge": summary.get("respect_llm_judge"),
        "use_supported_docs": summary.get("use_supported_docs"),
        "match_prefix_mode": summary.get("match_prefix_mode"),
        "max_query_samples": summary.get("max_query_samples"),
        "cache_root": summary.get("cache_root"),
        "output_csv": summary.get("output_csv"),
        "output_json": summary.get("output_json"),
    }
    with open(output_acc, "w", encoding="utf-8") as f:
        json.dump(acc_only, f, ensure_ascii=False, indent=2)


def judge_answer_with_openai(
    openai_client: OpenAI,
    openai_model: str,
    question: str,
    predicted_answer: str,
    ground_truth_answer: str,
) -> Tuple[bool, str]:
    question = question.strip()
    predicted_answer = predicted_answer.strip()
    ground_truth_answer = ground_truth_answer.strip()

    cache_key = f"{question}|||{predicted_answer}|||{ground_truth_answer}"
    if cache_key in JUDGE_CACHE:
        c = JUDGE_CACHE[cache_key]
        return c["is_correct"], c["reason"]

    prompt = f"""你是一个答案评估专家。请判断预测答案是否正确回答问题。

    问题: {question}
    标准答案: {ground_truth_answer}
    预测答案: {predicted_answer}

    输出格式：
    判断: [正确/错误]
    原因: [简要说明]"""

    try:
        resp = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "你是一个专业的答案评估专家。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
        )
        text = (resp.choices[0].message.content or "").strip()
        is_correct = False
        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("判断") and (":" in s or "：" in s):
                value = s.split(":", 1)[-1].split("：", 1)[-1].strip()
                if "正确" in value:
                    is_correct = True
                elif "错误" in value:
                    is_correct = False
                break

        JUDGE_CACHE[cache_key] = {"is_correct": is_correct, "reason": text}
        _save_judge_cache()
        return is_correct, text
    except Exception as e:
        return False, f"OpenAI judge error: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset-level prefix-aware preprocess KV evaluation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30002)
    parser.add_argument("--model", default="DeepSeek-V3.2")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--dataset_name", default="musique")
    parser.add_argument("--doc_pool_path", default=None)
    parser.add_argument("--cache_root", default="/mnt/data3/shm/fusionrag_tree_cache/DeepSeek-v3.2")
    parser.add_argument(
        "--cache_source",
        default="preprocess",
        choices=["preprocess", "raw"],
        help="Inference cache source: preprocess or raw. This will be sent as fusionrag_params.preprocess.",
    )
    parser.add_argument("--build_cache_source", default="preprocess", choices=["preprocess", "raw"])
    parser.add_argument("--preprocess_prefix", default="【预处理前缀】请牢记以下文档内容：")
    parser.add_argument("--cache_build_max_tokens", type=int, default=1)
    parser.add_argument("--gen_max_tokens", type=int, default=128)
    parser.add_argument("--system_prompt", default="你是一个历史专家，请根据材料回答问题。")
    parser.add_argument("--match_prefix_mode", default="docs_only", choices=["docs_only", "system_plus_docs"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--max_query_samples",
        type=int,
        default=None,
        help="Maximum number of query samples to run (applied after sample construction).",
    )
    parser.add_argument(
        "--eval_level",
        default="both",
        choices=["both", "main", "sub"],
        help="Evaluate main questions, sub questions, or both.",
    )
    parser.add_argument(
        "--skip_untestable",
        dest="skip_untestable",
        action="store_true",
        help="Skip main groups where any sub-answer is 'No relevant information found/没有相关信息'.",
    )
    parser.add_argument(
        "--no-skip_untestable",
        dest="skip_untestable",
        action="store_false",
        help="Do not skip untestable groups.",
    )
    parser.set_defaults(skip_untestable=True)
    parser.add_argument(
        "--respect_llm_judge",
        action="store_true",
        help="Also skip samples with llm_judge=False, matching original optional rule.",
    )
    parser.add_argument(
        "--use_supported_docs",
        action="store_true",
        help="Use intermediate_context[*]['retrieve docs supported'] instead of 'retrieve docs'.",
    )
    parser.add_argument("--output_csv", default="/tmp/fusionrag_dataset_eval.csv")
    parser.add_argument("--output_json", default="/tmp/fusionrag_dataset_eval.json")

    parser.add_argument("--enable_openai_judge", action="store_true")
    parser.add_argument('--openai_base_url', type=str, default='https://api.deepseek.com/v1',
                        help='OpenAI API base URL')
    parser.add_argument('--openai_api_key', type=str, default='sk-519d391217894b6e91e7c2ebf2a9f4df',
                        help='OpenAI API key')
    parser.add_argument('--openai_model', type=str, default='deepseek-chat',
                        help='OpenAI model for judging')
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    if args.doc_pool_path is None:
        data_dir = os.path.dirname(os.path.abspath(args.data_path))
        if "2wiki" in args.dataset_name.lower():
            corpus = "2wiki_input_rebuilt.json"
        else:
            corpus = "musique_input_rebuilt.json"
        doc_pool_path = os.path.join(data_dir, corpus)
    else:
        doc_pool_path = args.doc_pool_path

    print(f"Loading doc pool: {doc_pool_path}")
    doc_pool = DocumentPoolLoader(doc_pool_path)

    samples = load_dataset_samples(
        args.data_path,
        max_samples=args.max_samples,
        eval_level=args.eval_level,
        skip_untestable=args.skip_untestable,
        respect_llm_judge=args.respect_llm_judge,
        use_supported_docs=args.use_supported_docs,
    )
    if args.max_query_samples is not None and args.max_query_samples > 0:
        samples = samples[: args.max_query_samples]
    print(f"Loaded samples: {len(samples)}")

    build_preprocess = args.build_cache_source == "preprocess"
    if build_preprocess:
        build_cache_dir = os.path.join(args.cache_root, "preprocess_kv_cache")
    else:
        build_cache_dir = os.path.join(args.cache_root, "raw_kv_cache")

    os.makedirs(build_cache_dir, exist_ok=True)

    openai_client = None
    if args.enable_openai_judge:
        api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI judge enabled but no API key provided")
        openai_client = OpenAI(api_key=api_key, base_url=args.openai_base_url)
        _load_judge_cache(os.path.dirname(args.output_json))

    built_doc_ids = set()
    rows: List[Dict[str, Any]] = []
    total_f1 = 0.0
    total_em = 0.0
    output_acc = _derive_accuracy_path(args.output_json)

    for i, s in enumerate(samples, start=1):
        docs_text: List[str] = []
        
        # 1) Ensure caches for all docs in this question
        for doc_id in s.doc_ids:
            doc_text = doc_pool.get_document(doc_id)
            docs_text.append(doc_text)

            if doc_id in built_doc_ids:
                continue
            if has_doc_cache(build_cache_dir, doc_text):
                print(f"[SKIP] doc_id={doc_id} already has cache, skipping build")
                built_doc_ids.add(doc_id)
                continue

            print(f"[BUILD] Building cache for doc_id={doc_id} (main_id={s.main_id} sub_id={s.sub_id})")
            ok = build_cache_for_doc(
                base_url=base_url,
                model=args.model,
                doc_text=doc_text,
                preprocess=build_preprocess,
                preprocess_prefix=args.preprocess_prefix,
                max_tokens=args.cache_build_max_tokens,
            )
            print(f"[BUILD FIN] doc_id={doc_id} finished with status {ok}")
            if not ok:
                print(f"[WARN] cache build failed for doc_id={doc_id}")
            else:
                built_doc_ids.add(doc_id)

        # 2) Inference with cache
        try:
            pred = infer_answer(
                base_url=base_url,
                model=args.model,
                system_prompt=args.system_prompt,
                question=s.question,
                docs=docs_text,
                max_tokens=args.gen_max_tokens,
                cache_source=args.cache_source,
                match_prefix_mode=args.match_prefix_mode,
            )
        except Exception as e:
            pred = f"[INFER_ERROR] {e}"

        # 3) Evaluation
        gt = s.answer
        try:
            f1 = float(compute_f1(gt, pred, None))
        except Exception:
            f1 = 0.0
        em = 1.0 if _exact_match_score(gt, pred) else 0.0

        judge_correct = None
        judge_reason = ""
        if openai_client is not None:
            judge_correct, judge_reason = judge_answer_with_openai(
                openai_client=openai_client,
                openai_model=args.openai_model,
                question=s.question,
                predicted_answer=pred,
                ground_truth_answer=gt,
            )

        total_f1 += f1
        total_em += em

        row = {
            "sample_id": s.sample_id,
            "level": s.level,
            "main_id": s.main_id,
            "sub_id": s.sub_id,
            "should_test": s.should_test,
            "main_question": s.main_question,
            "question": s.question,
            "ground_truth": gt,
            "predicted": pred,
            "doc_ids": s.doc_ids,
            "f1": f1,
            "em": em,
            "judge_correct": judge_correct,
            "judge_reason": judge_reason,
        }
        rows.append(row)

        avg_f1 = total_f1 / len(rows) if rows else 0.0
        avg_em = total_em / len(rows) if rows else 0.0

        # Level-wise metrics
        main_rows = [r for r in rows if r["level"] == "main"]
        sub_rows = [r for r in rows if r["level"] == "sub"]
        main_avg_f1 = (sum(r["f1"] for r in main_rows) / len(main_rows)) if main_rows else 0.0
        main_avg_em = (sum(r["em"] for r in main_rows) / len(main_rows)) if main_rows else 0.0
        sub_avg_f1 = (sum(r["f1"] for r in sub_rows) / len(sub_rows)) if sub_rows else 0.0
        sub_avg_em = (sum(r["em"] for r in sub_rows) / len(sub_rows)) if sub_rows else 0.0

        # Main-question strict accuracy: all sub-questions under same main_id must be EM=1
        sub_by_main: Dict[int, List[Dict[str, Any]]] = {}
        for r in sub_rows:
            sub_by_main.setdefault(r["main_id"], []).append(r)
        strict_main_total = len(sub_by_main)
        strict_main_correct = sum(
            1 for _, rs in sub_by_main.items() if len(rs) > 0 and all(float(x["em"]) == 1.0 for x in rs)
        )
        strict_main_acc = (strict_main_correct / strict_main_total) if strict_main_total > 0 else 0.0

        summary = {
            "num_samples": len(rows),
            "avg_f1": avg_f1,
            "avg_em": avg_em,
            "main_avg_f1": main_avg_f1,
            "main_avg_em": main_avg_em,
            "sub_avg_f1": sub_avg_f1,
            "sub_avg_em": sub_avg_em,
            "strict_main_total": strict_main_total,
            "strict_main_correct": strict_main_correct,
            "strict_main_acc": strict_main_acc,
            "build_cache_source": args.build_cache_source,
            "inference_cache_source": args.cache_source,
            "eval_level": args.eval_level,
            "skip_untestable": args.skip_untestable,
            "respect_llm_judge": args.respect_llm_judge,
            "use_supported_docs": args.use_supported_docs,
            "match_prefix_mode": args.match_prefix_mode,
            "max_query_samples": args.max_query_samples,
            "cache_root": args.cache_root,
            "output_csv": args.output_csv,
            "output_json": args.output_json,
            "rows": rows,
            "progress": {"done": i, "total": len(samples)},
        }

        _write_outputs(rows, summary, args.output_csv, args.output_json, output_acc)

        if i % 10 == 0 or i == len(samples):
            print(f"Progress: {i}/{len(samples)} avg_f1={avg_f1:.4f} avg_em={avg_em:.4f}")

    avg_f1 = total_f1 / len(rows) if rows else 0.0
    avg_em = total_em / len(rows) if rows else 0.0

    # Level-wise metrics
    main_rows = [r for r in rows if r["level"] == "main"]
    sub_rows = [r for r in rows if r["level"] == "sub"]
    main_avg_f1 = (sum(r["f1"] for r in main_rows) / len(main_rows)) if main_rows else 0.0
    main_avg_em = (sum(r["em"] for r in main_rows) / len(main_rows)) if main_rows else 0.0
    sub_avg_f1 = (sum(r["f1"] for r in sub_rows) / len(sub_rows)) if sub_rows else 0.0
    sub_avg_em = (sum(r["em"] for r in sub_rows) / len(sub_rows)) if sub_rows else 0.0

    # Main-question strict accuracy: all sub-questions under same main_id must be EM=1
    sub_by_main: Dict[int, List[Dict[str, Any]]] = {}
    for r in sub_rows:
        sub_by_main.setdefault(r["main_id"], []).append(r)
    strict_main_total = len(sub_by_main)
    strict_main_correct = sum(
        1 for _, rs in sub_by_main.items() if len(rs) > 0 and all(float(x["em"]) == 1.0 for x in rs)
    )
    strict_main_acc = (strict_main_correct / strict_main_total) if strict_main_total > 0 else 0.0

    summary = {
        "num_samples": len(rows),
        "avg_f1": avg_f1,
        "avg_em": avg_em,
        "main_avg_f1": main_avg_f1,
        "main_avg_em": main_avg_em,
        "sub_avg_f1": sub_avg_f1,
        "sub_avg_em": sub_avg_em,
        "strict_main_total": strict_main_total,
        "strict_main_correct": strict_main_correct,
        "strict_main_acc": strict_main_acc,
        "build_cache_source": args.build_cache_source,
        "inference_cache_source": args.cache_source,
        "eval_level": args.eval_level,
        "skip_untestable": args.skip_untestable,
        "respect_llm_judge": args.respect_llm_judge,
        "use_supported_docs": args.use_supported_docs,
        "match_prefix_mode": args.match_prefix_mode,
        "max_query_samples": args.max_query_samples,
        "cache_root": args.cache_root,
        "output_csv": args.output_csv,
        "rows": rows,
    }
    _write_outputs(rows, summary, args.output_csv, args.output_json, output_acc)

    print("\n=== Done ===")
    print(f"samples={len(rows)} avg_f1={avg_f1:.4f} avg_em={avg_em:.4f}")
    print(
        f"main_avg_f1={main_avg_f1:.4f} main_avg_em={main_avg_em:.4f} "
        f"sub_avg_f1={sub_avg_f1:.4f} sub_avg_em={sub_avg_em:.4f}"
    )
    print(f"strict_main_acc={strict_main_acc:.4f} ({strict_main_correct}/{strict_main_total})")
    print(f"csv={args.output_csv}")
    print(f"json={args.output_json}")
    print(f"accuracy={output_acc}")


if __name__ == "__main__":
    main()
