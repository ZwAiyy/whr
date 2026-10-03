# -*- coding: utf-8 -*-
"""以大模型作为「评判者」：先给出它认为正确的切分（准金标准），再对四种方法打分。

注意：这是机器评判（LLM-as-a-judge），不是人工标注，报告中会明确标注其性质；
它评价的是「分词合理性」，与人工标注的分词 F1 只能做近似对照。
结果同样落盘缓存（outputs/cache/llm_judge_cache.jsonl），可断点续跑。
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request

import config as C
from seg_llm import RE_FENCE

JUDGE_SYSTEM = (
    "你是中文分词（CWS）评测专家。给定原文，你要：\n"
    "1) 先给出你认为最合理的切分（标准分词：功能词单字、专名不拆、"
    "数字/英文/标点各自成单位）；\n"
    "2) 再对给定的若干候选切分逐条打分（候选已统一按 \"/\" 连接，"
    "注意：**标点已被当作硬边界，各方法是否输出标点 token 不作为扣分点**）：\n"
    "   boundary_ok：词边界是否合理（0-10）\n"
    "   exp_complete：词是否表意完整、有无把固定搭配切碎（0-10）\n"
    "   prop_noun：专有名词/领域术语（如“龙谕酒庄”“贺兰山东麓”“橡木桶”）是否完整（0-10）\n"
    "   no_fragment：是否几乎没有无意义碎词（如“导/游”“解/员”）（0-10）\n"
    "   overall：总体切分质量（0-10）\n"
    "   comment：不超过 20 字的理由\n"
    "只输出 JSON，不要解释。格式：\n"
    '{"items":[{"id":1,"ideal":"词/词/词","scores":{"规则分词":{"boundary_ok":8,'
    '"exp_complete":7,"prop_noun":9,"no_fragment":8,"overall":8,"comment":"..."}, ...}}]}'
)

JUDGE_SCORES = ["boundary_ok", "exp_complete", "prop_noun", "no_fragment", "overall"]


class SegmentationJudge:
    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, cache_file=None):
        self.base_url = (base_url or C.LLM_BASE_URL).rstrip("/")
        self.model = model or C.LLM_MODEL
        self.api_key = api_key if api_key is not None else C.LLM_API_KEY
        self.cache_path = cache_file or (C.DIR_CACHE / "llm_judge_cache.jsonl")
        self.cache: dict[str, dict] = {}
        self.stats = {"requests": 0, "cache_hit": 0, "failed": 0}
        self._load()

    @staticmethod
    def key(text: str, methods: list[str]) -> str:
        raw = f"{C.JUDGE_PROMPT_VERSION}\u0001{text}\u0001{','.join(sorted(methods))}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _load(self):
        if not self.cache_path.exists():
            return
        try:
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["k"]] = rec["v"]
        except Exception:  # noqa: BLE001
            pass

    def _save(self, k: str, v: dict):
        self.cache[k] = v
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")

    def _chat(self, payload: list[dict]) -> dict:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": JUDGE_SYSTEM},
                         {"role": "user", "content": json.dumps({"items": payload},
                                                                 ensure_ascii=False)}],
            "temperature": 0.0,
            "max_tokens": C.LLM_MAX_TOKENS,
            "stream": False,
            "response_format": {"type": "json_object"},
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        last = None
        for attempt in range(C.LLM_RETRY):
            try:
                with urllib.request.urlopen(req, timeout=C.LLM_TIMEOUT) as r:
                    data = json.loads(r.read().decode("utf-8"))
                self.stats["requests"] += 1
                content = data["choices"][0]["message"]["content"]
                m = RE_FENCE.search(content)
                if m:
                    content = m.group(1)
                return json.loads(content)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"评判请求失败：{last}")

    def judge(self, texts: list[str], hyps: dict[str, list[list[str]]],
              log=print) -> dict[str, dict]:
        """返回 {text: {"ideal": [词...], "scores": {方法: {...}}}}。"""
        methods = list(hyps)
        out: dict[str, dict] = {}
        todo = []
        for i, t in enumerate(texts):
            k = self.key(t, methods)
            if k in self.cache:
                out[t] = self.cache[k]
                self.stats["cache_hit"] += 1
            else:
                todo.append((i, t, k))
        log(f"    LLM 评判：共 {len(texts)} 条，缓存命中 {self.stats['cache_hit']} 条，"
            f"待请求 {len(todo)} 条")
        for start in range(0, len(todo), C.JUDGE_BATCH_SIZE):
            chunk = todo[start:start + C.JUDGE_BATCH_SIZE]
            payload = []
            for local_id, (src_i, t, _) in enumerate(chunk, start=1):
                payload.append({"id": local_id, "text": t,
                                "candidates": {m: "/".join(hyps[m][src_i]) for m in methods}})
            try:
                res = self._chat(payload)
                items = res.get("items", []) if isinstance(res, dict) else res
                by_id = {int(it.get("id", 0)): it for it in items if isinstance(it, dict)}
                for local_id, (_, t, k) in enumerate(chunk, start=1):
                    it = by_id.get(local_id)
                    if not it:
                        self.stats["failed"] += 1
                        continue
                    ideal = [w.strip() for w in str(it.get("ideal", "")).split("/") if w.strip()]
                    rec = {"ideal": ideal, "scores": it.get("scores", {})}
                    out[t] = rec
                    self._save(k, rec)
            except Exception as e:  # noqa: BLE001
                log(f"    [warn] 评判批次失败：{e}")
                self.stats["failed"] += len(chunk)
            log(f"      进度 {min(start + C.JUDGE_BATCH_SIZE, len(todo))}/{len(todo)}")
            time.sleep(C.LLM_SLEEP)
        return out

    def stats_report(self) -> dict:
        return dict(self.stats)
