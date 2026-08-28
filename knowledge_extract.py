import csv
import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

CSV_FILE = os.path.join(os.path.dirname(__file__), "weather_data.csv")

# ── 大模型配置 ──
MODELS = {
    "deepseek": {
        "name": "DeepSeek-V3",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "model": "deepseek-chat",
    },
    "qwen": {
        "name": "通义千问-Max",
        "api_key": os.getenv("QWEN_API_KEY", ""),
        "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "model": "qwen-max",
    },
    "mimo": {
        "name": "MiMo",
        "api_key": os.getenv("MIMO_API_KEY", ""),
        "base_url": os.getenv("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1"),
        "model": "mimo-v2.5-pro",
    },
}

# ── 知识抽取 Prompt ──
SYSTEM_PROMPT = """你是一个气象知识图谱专家。你的任务是从天气数据中抽取结构化知识。

请从给定的天气 CSV 数据中抽取以下内容，以 JSON 格式返回：

1. **entities** (实体列表): 每个实体包含
   - id: 唯一标识
   - type: 实体类型 (WeatherRecord / WeatherEvent / Location / TimePeriod)
   - name: 实体名称
   - properties: 属性字典

2. **relationships** (关系列表): 每个关系包含
   - source: 源实体 id
   - target: 目标实体 id
   - type: 关系类型 (如 OCCURRED_ON, HAS_EVENT, FOLLOWED_BY, SIMILAR_TO)
   - properties: 关系属性

3. **rules** (气象规律): 从数据中发现的有意义的规律，如
   - "连续干旱超过 N 天后出现降雨的概率"
   - "高温日与大风日的共现关系"
   - "季节性温度变化趋势"

请返回纯 JSON，不要包含 markdown 代码块标记。"""


def extract_sample_data(num_rows=50):
    """读取 CSV 前 N 行作为样本数据"""
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= num_rows:
                break
            rows.append(row)
    return rows


def call_llm(config, user_prompt):
    """调用大模型 API"""
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    )
    print(f"  调用 {config['name']} ...")
    start = time.time()
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=8000,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    print(f"  {config['name']} 完成，耗时 {elapsed:.1f}s，"
          f"输入 tokens: {response.usage.prompt_tokens}, "
          f"输出 tokens: {response.usage.completion_tokens}")
    return content, elapsed, response.usage


def parse_json_response(text):
    import re
    text = text.strip()
    # 去掉 markdown 代码块标记
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    # 第一次尝试：直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 修复常见 LLM JSON 错误
    fixed = text
    # 1. 去掉尾部逗号 (}, ] 前的逗号)
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    # 2. 把裸值如 1-5, 30+ 等替换为字符串
    fixed = re.sub(r':\s*(-?\d+[\-~+]\d+)\s*([,}\n])', r': "\1"\2', fixed)
    # 3. 把单引号替换为双引号（简单场景）
    fixed = fixed.replace("'", '"')
    # 4. 去掉注释
    fixed = re.sub(r'//[^\n]*', '', fixed)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 5. 尝试截断修复：移除末尾不完整字段并补全括号
    last_comma = fixed.rfind(",")
    last_brace = fixed.rfind("}")
    if last_comma > last_brace:
        fixed = fixed[:last_comma]
    open_braces = fixed.count("{") - fixed.count("}")
    open_brackets = fixed.count("[") - fixed.count("]")
    fixed += "]" * max(0, open_brackets) + "}" * max(0, open_braces)

    try:
        parsed = json.loads(fixed)
        print("  (JSON 已自动修复)")
        return parsed
    except json.JSONDecodeError as e:
        print(f"  JSON 解析失败: {e}")
        print(f"  原文前 500 字: {text[:500]}")
        return None


def main():
    # 检查 API Key
    available_models = {}
    for key, config in MODELS.items():
        if config["api_key"] and config["api_key"] != f"your_{key}_key_here":
            available_models[key] = config
        else:
            print(f"跳过 {config['name']}: 未配置 API Key")

    if not available_models:
        print("\n请在 .env 文件中配置至少一个大模型的 API Key！")
        print("编辑: C:\\Users\\Windy\\Desktop\\whr\\.env")
        return

    # 读取样本数据
    rows = extract_sample_data(50)
    sample_csv = "\n".join([
        ",".join(rows[0].keys()),  # 表头
        *[",".join(row.values()) for row in rows[:20]],  # 前 20 行
    ])

    # 统计摘要
    summary_parts = [
        f"数据总量: {len(extract_sample_data(10000))} 条记录",
        f"时间范围: {rows[0]['date']} ~ {rows[-1]['date']}",
        f"字段: {', '.join(rows[0].keys())}",
    ]

    user_prompt = f"""以下是天气数据的基本信息：
{chr(10).join(summary_parts)}

前 20 行数据样本：
{sample_csv}

请从这些数据中抽取知识图谱的实体、关系和气象规律。
注意：
1. 不需要逐条抽取所有记录，而是抽取有代表性的模式和规律
2. 实体应包含关键的天气事件和时间段
3. 关系要体现天气事件之间的关联
4. 规律要基于数据中的统计特征"""

    # 逐个调用大模型
    results = {}
    for key, config in available_models.items():
        print(f"\n{'='*50}")
        print(f"抽取模型: {config['name']}")
        print(f"{'='*50}")

        raw_text, elapsed, usage = call_llm(config, user_prompt)
        parsed = parse_json_response(raw_text)

        output = {
            "model": config["name"],
            "elapsed_seconds": elapsed,
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "raw_response": raw_text,
            "parsed": parsed,
        }

        output_file = os.path.join(os.path.dirname(__file__), f"knowledge_{key}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"  结果已保存: {output_file}")

        results[key] = output

    # ── 对比分析 ──
    print(f"\n{'='*70}")
    print("  大模型知识抽取对比")
    print(f"{'='*70}")

    model_names = {key: results[key]["model"] for key in results}
    header = f"{'指标':<20}" + "".join(f"{name:<20}" for name in model_names.values())
    print(f"\n{header}")
    print("-" * (20 + 20 * len(model_names)))

    for metric_name, metric_fn in [
        ("耗时(秒)", lambda r: f"{r['elapsed_seconds']:.1f}"),
        ("输入tokens", lambda r: str(r['input_tokens'])),
        ("输出tokens", lambda r: str(r['output_tokens'])),
        ("实体数量", lambda r: str(len(r['parsed'].get('entities', []))) if r['parsed'] else "解析失败"),
        ("关系数量", lambda r: str(len(r['parsed'].get('relationships', []))) if r['parsed'] else "解析失败"),
        ("规律数量", lambda r: str(len(r['parsed'].get('rules', []))) if r['parsed'] else "解析失败"),
    ]:
        vals = []
        for key in results:
            vals.append(metric_fn(results[key]))
        print(f"{metric_name:<20}" + "".join(f"{v:<20}" for v in vals))

    # 保存对比结果
    comparison = {
        "models_compared": list(results.keys()),
        "metrics": {}
    }
    for key, r in results.items():
        comparison["metrics"][r["model"]] = {
            "elapsed_seconds": r["elapsed_seconds"],
            "tokens": {"input": r['input_tokens'], "output": r['output_tokens']},
            "entities_count": len(r['parsed'].get('entities', [])) if r['parsed'] else 0,
            "relationships_count": len(r['parsed'].get('relationships', [])) if r['parsed'] else 0,
            "rules_count": len(r['parsed'].get('rules', [])) if r['parsed'] else 0,
        }

    comp_file = os.path.join(os.path.dirname(__file__), "llm_comparison.json")
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存: {comp_file}")


if __name__ == "__main__":
    main()
