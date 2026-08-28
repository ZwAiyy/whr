// ============================================
// Neo4j 可视化查询 — 复制到 Neo4j Browser 执行
// ============================================

// ── 1. 查看所有节点和关系（全貌）──
MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 200

// ── 2. 查看某一天的天气记录及其事件 ──
MATCH (r:WeatherRecord {date: "2024-08-24"})-[rel]->(target)
RETURN r, rel, target

// ── 3. 查看高温日关联的所有记录 ──
MATCH (r:WeatherRecord)-[:HAS_EVENT]->(e:WeatherEvent {name: "HeatDay"})
RETURN r, e
ORDER BY r.maxTemp DESC

// ── 4. 查看连续干旱的记录链 ──
MATCH path = (r1:WeatherRecord)-[:NEXT_DAY*2..10]->(r2:WeatherRecord)
WHERE r1.dryStreak > 1 AND ALL(n IN nodes(path) WHERE n.dryStreak > 0)
RETURN path
LIMIT 20

// ── 5. 月度温度趋势 ──
MATCH (m:Month)
RETURN m.id AS month, m.avgMaxTemp, m.avgMinTemp, m.totalPrecipitation
ORDER BY m.id

// ── 6. 查看大模型抽取的规律 ──
MATCH (rule:Rule)
RETURN rule.model AS model, rule.description AS 规律
ORDER BY rule.model

// ── 7. 查看大模型抽取的实体和关系 ──
MATCH (n)
WHERE n.source_model IS NOT NULL
RETURN n, labels(n) AS 类型
LIMIT 100

// ── 8. 某月所有天气事件统计 ──
MATCH (r:WeatherRecord)-[:IN_MONTH]->(m:Month {id: "2024-07"})
MATCH (r)-[:HAS_EVENT]->(e:WeatherEvent)
RETURN e.name AS event, count(*) AS cnt
ORDER BY cnt DESC

// ── 9. 同一天出现多种极端天气的记录 ──
MATCH (r:WeatherRecord)-[:HAS_EVENT]->(e:WeatherEvent)
WITH r, collect(e.name) AS events, count(e) AS eventCount
WHERE eventCount >= 2
RETURN r.date, r.maxTemp, r.minTemp, events, eventCount
ORDER BY eventCount DESC
LIMIT 20

// ── 10. 大模型对比：DeepSeek vs 通义千问抽取的知识 ──
MATCH (n)
WHERE n.source_model = "DeepSeek-V3"
WITH count(n) AS deepseekCount
MATCH (m)
WHERE m.source_model = "通义千问-Max"
RETURN deepseekCount, count(m) AS qwenCount
