# -*- coding: utf-8 -*-
"""分词方法四：基于大模型（LLM）的分词。

做法：用 Few-shot 提示让大模型直接输出切分结果，要求以 "/" 连接词语，
      再用确定性解析器把返回文本还原为词序列；若拼接结果与原文不一致
      （字符级校验失败）则判为无效响应并重试，保证结果可验证。
工程要点：
  * 结果按 (提示版本, 文本) 的 md5 落盘缓存（outputs/cache/llm_seg_cache.jsonl），
    支持断点续跑、避免重复计费；
  * 批处理（每次请求 LLM_BATCH_SIZE 条），失败指数退避重试；
  * 记录每条文本是否命中缓存/是否退化为规则分词，便于在报告中说明。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request

import config as C

SYSTEM_PROMPT = (
    "你是一名中文分词引擎（CWS, Chinese Word Segmentation）。"
    "任务：把用户给出的每条中文文本切分成词，并用英文斜杠 \"/\" 连接。"
    "规则：\n"
    "1) 只输出切分结果，不要解释、不要翻译、不要改写原文；\n"
    "2) 数字、英文、标点各自作为独立单位输出，标点也保留；\n"
    "3) 不添加、不删除任何字符（去掉分隔符后必须与原文完全一致）；\n"
    "4) 领域专名（如“龙谕酒庄”“贺兰山东麓”“橡木桶”）应作为一个词；\n"
    "5) 输入是 JSON 数组，输出必须是 {\"results\": [...]} 形式的 JSON 对象，"
    "results 是与输入等长的字符串数组，每个元素是切分后的字符串。"
)

FEWSHOT = [
    {"role": "user", "content": json.dumps(
        ["龙谕酒庄真的超出预期！", "酒窖里一排排橡木桶氛围感拉满"], ensure_ascii=False)},
    {"role": "assistant", "content": json.dumps(
        {"results": ["龙谕酒庄/真的/超出/预期/！", "酒窖/里/一排排/橡木桶/氛围感/拉满"]},
        ensure_ascii=False)},
    {"role": "user", "content": json.dumps(
        ["门票80元，游玩1-2小时。", "讲解员专业细致，值得推荐"], ensure_ascii=False)},
    {"role": "assistant", "content": json.dumps(
        {"results": ["门票/80/元/，/游玩/1/-/2/小时/。", "讲解员/专业/细致/，/值得/推荐"]},
        ensure_ascii=False)},
]

RE_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class LLMSegmenter:
    """大模型分词器（DeepSeek Chat API，OpenAI 兼容协议）。"""

    name = "llm"
    display = "大模型分词"

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, cache_file=None, use_cache: bool = True,
                 fallback=None):
        self.base_url = (base_url or C.LLM_BASE_URL).rstrip("/")
        self.model = model or C.LLM_MODEL
        self.api_key = api_key if api_key is not None else C.LLM_API_KEY
        self.cache_path = cache_file or (C.DIR_CACHE / "llm_seg_cache.jsonl")
        self.use_cache = use_cache
        self.fallback = fallback                 # 解析失败时的兜底分词器
        self.cache: dict[str, list[str]] = {}
        self.stats = {"requests": 0, "cache_hit": 0, "retry": 0, "invalid": 0,
                      "fallback": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._load_cache()

    # ------------------------------------------------------------ 缓存
    @staticmethod
    def key(text: str) -> str:
        return hashlib.md5(f"{C.LLM_PROMPT_VERSION}\u0001{text}".encode("utf-8")).hexdigest()

    def _load_cache(self):
        if not self.cache_path.exists():
            return
        try:
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                self.cache[rec["k"]] = rec["t"]
        except Exception:  # noqa: BLE001
            pass

    def _append_cache(self, k: str, toks: list[str]):
        self.cache[k] = toks
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": k, "t": toks}, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ 调用
    def _chat(self, texts: list[str]) -> list[str]:
        """一次 API 调用，返回与输入等长的切分结果字符串列表。"""
        user = json.dumps(texts, ensure_ascii=False)
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + FEWSHOT
                        + [{"role": "user", "content": user}],
            "temperature": C.LLM_TEMPERATURE,
            "max_tokens": C.LLM_MAX_TOKENS,
            "stream": False,
            "response_format": {"type": "json_object"},
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        last_err = None
        for attempt in range(C.LLM_RETRY):
            try:
                with urllib.request.urlopen(req, timeout=C.LLM_TIMEOUT) as r:
                    data = json.loads(r.read().decode("utf-8"))
                self.stats["requests"] += 1
                usage = data.get("usage") or {}
                self.stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
                self.stats["completion_tokens"] += int(usage.get("completion_tokens", 0))
                content = data["choices"][0]["message"]["content"]
                return self._parse(content, len(texts))
            except Exception as e:  # noqa: BLE001
                last_err = e
                if hasattr(e, "read"):
                    try:
                        last_err = f"{e} {e.read().decode('utf-8', 'ignore')[:200]}"
                    except Exception:  # noqa: BLE001
                        pass
                self.stats["retry"] += 1
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM 请求失败：{last_err}")

    def _parse(self, content: str, n_expected: int) -> list[str]:
        """解析模型输出为 n_expected 条「以 / 连接的字符串」。"""
        txt = content.strip()
        m = RE_FENCE.search(txt)
        if m:
            txt = m.group(1).strip()
        try:
            obj = json.loads(txt)
        except Exception:  # noqa: BLE001
            raise ValueError(f"返回不是合法 JSON：{txt[:120]}")
        if isinstance(obj, dict):
            # 兼容 {"results": [...]} / {"segments": [...]} 等包裹形式
            for key in ("results", "segments", "result", "data", "output"):
                if isinstance(obj.get(key), list):
                    obj = obj[key]
                    break
            else:
                for v in obj.values():
                    if isinstance(v, list):
                        obj = v
                        break
        if not isinstance(obj, list) or len(obj) != n_expected:
            raise ValueError(f"返回条数不符：{type(obj)} len={len(obj) if isinstance(obj, list) else '-'}")
        return [str(x) for x in obj]

    # ------------------------------------------------------------ 解析成词
    @staticmethod
    def split_tokens(s: str) -> list[str]:
        """把 "/" 连接的字符串还原成词序列。"""
        s = s.strip()
        if not s:
            return []
        if C.LLM_SEG_DELIMITER in s:
            parts = [p for p in s.split(C.LLM_SEG_DELIMITER)]
        else:
            parts = s.split()          # 兜底：模型用了空格分隔
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _check(text: str, toks: list[str]) -> bool:
        """字符级一致性校验：去掉空白后必须与原文相同。"""
        joined = "".join(toks).replace(" ", "").replace("\u3000", "")
        ref = text.replace(" ", "").replace("\u3000", "")
        return joined == ref

    # ------------------------------------------------------------ 主入口
    def cut_batch(self, texts, log=print) -> list[list[str]]:
        results: list[list[str] | None] = [None] * len(texts)
        todo: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            k = self.key(t)
            if self.use_cache and k in self.cache:
                results[i] = self.cache[k]
                self.stats["cache_hit"] += 1
            else:
                todo.append((i, t))
        log(f"    LLM 分词：共 {len(texts)} 条，缓存命中 {self.stats['cache_hit']} 条，"
            f"待请求 {len(todo)} 条")
        for start in range(0, len(todo), C.LLM_BATCH_SIZE):
            chunk = todo[start:start + C.LLM_BATCH_SIZE]
            batch = [t for _, t in chunk]
            try:
                outs = self._chat(batch)
                for (i, t), s in zip(chunk, outs):
                    toks = self.split_tokens(s)
                    if self._check(t, toks):
                        results[i] = toks
                        self._append_cache(self.key(t), toks)
                    else:
                        # 单条重试一次，仍不一致则记为无效
                        try:
                            single = self._chat([t])[0]
                            toks2 = self.split_tokens(single)
                        except Exception:  # noqa: BLE001
                            toks2 = []
                        if self._check(t, toks2):
                            results[i] = toks2
                            self._append_cache(self.key(t), toks2)
                        else:
                            self.stats["invalid"] += 1
                            results[i] = None
            except Exception as e:  # noqa: BLE001
                log(f"    [warn] 批 {start // C.LLM_BATCH_SIZE + 1} 请求失败：{e}")
                for i, _ in chunk:
                    results[i] = None
            done = min(start + C.LLM_BATCH_SIZE, len(todo))
            log(f"      进度 {done}/{len(todo)}")
            time.sleep(C.LLM_SLEEP)
        # 兜底
        for i, t in enumerate(texts):
            if results[i] is None:
                self.stats["fallback"] += 1
                if self.fallback is not None:
                    results[i] = self.fallback.cut(t)
                else:
                    results[i] = self.split_tokens(t.replace("", "")) or list(t)
        return [r or [] for r in results]

    def cut(self, text: str) -> list[str]:
        return self.cut_batch([text], log=lambda *_: None)[0]

    def stats_report(self) -> dict:
        return dict(self.stats)
