import csv
import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

CSV_FILE = os.path.join(os.path.dirname(__file__), "weather_data.csv")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

# 布尔列 -> 事件类
BOOLEAN_COLUMNS = {
    "frost_day": "FrostDay",
    "heat_day": "HeatDay",
    "severe_heat_day": "SevereHeatDay",
    "dry_day": "DryDay",
    "strong_wind_day": "StrongWindDay",
    "dust_storm_risk": "DustStormRisk",
    "rainy_day": "RainyDay",
}


class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print(f"已连接 Neo4j: {uri}")

    def close(self):
        self.driver.close()

    def clear_database(self):
        """清空数据库"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("已清空数据库")

    def create_constraints(self):
        """创建唯一性约束和索引"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:WeatherRecord) REQUIRE r.date IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:WeatherEvent) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (r:WeatherRecord) ON (r.maxTemp)",
            "CREATE INDEX IF NOT EXISTS FOR (r:WeatherRecord) ON (r.month)",
        ]
        with self.driver.session() as session:
            for c in constraints:
                session.run(c)
        print("已创建约束和索引")

    def import_weather_records(self, rows):
        """批量导入天气记录节点"""
        query = """
        UNWIND $rows AS row
        CREATE (r:WeatherRecord {
            date: row.date,
            maxTemp: toFloat(row.maxTemp),
            minTemp: toFloat(row.minTemp),
            precipitation: toFloat(row.precipitation),
            windSpeed: toFloat(row.windSpeed),
            humidity: toInteger(row.humidity),
            dryStreak: toInteger(row.dryStreak),
            rainStreak: toInteger(row.rainStreak),
            month: row.month,
            year: row.year
        })
        """

        # 准备数据
        data = []
        for row in rows:
            date = row["date"].strip()
            if not date:
                continue
            parts = date.split("-")
            data.append({
                "date": date,
                "maxTemp": row["temperature_2m_max"].strip(),
                "minTemp": row["temperature_2m_min"].strip(),
                "precipitation": row["precipitation_sum"].strip(),
                "windSpeed": row["wind_speed_10m_max"].strip(),
                "humidity": row["relative_humidity_2m_mean"].strip(),
                "dryStreak": row.get("dry_streak", "0").strip(),
                "rainStreak": row.get("continuous_rain_streak", "0").strip(),
                "month": f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else "",
                "year": parts[0] if len(parts) >= 1 else "",
            })

        with self.driver.session() as session:
            # 分批导入（每批 500 条）
            batch_size = 500
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                session.run(query, rows=batch)
                print(f"  已导入 {min(i + batch_size, len(data))}/{len(data)} 条天气记录")

    def import_weather_events(self, rows):
        """创建天气事件节点，并建立与天气记录的关系"""
        # 先创建事件节点
        event_names = list(BOOLEAN_COLUMNS.values())
        with self.driver.session() as session:
            for event_name in event_names:
                session.run(
                    "MERGE (e:WeatherEvent {name: $name})",
                    name=event_name,
                )
        print(f"已创建 {len(event_names)} 个天气事件节点")

        # 创建 RECORD -> EVENT 关系
        rel_data = []
        for row in rows:
            date = row["date"].strip()
            if not date:
                continue
            for bool_col, event_name in BOOLEAN_COLUMNS.items():
                val = row.get(bool_col, "").strip().lower()
                if val == "true":
                    rel_data.append({"date": date, "event": event_name})

        query = """
        UNWIND $rels AS rel
        MATCH (r:WeatherRecord {date: rel.date})
        MATCH (e:WeatherEvent {name: rel.event})
        MERGE (r)-[:HAS_EVENT]->(e)
        """

        with self.driver.session() as session:
            batch_size = 1000
            for i in range(0, len(rel_data), batch_size):
                batch = rel_data[i:i + batch_size]
                session.run(query, rels=batch)
            print(f"已创建 {len(rel_data)} 条 HAS_EVENT 关系")

    def create_temporal_relations(self):
        """创建时间序列关系：前一天 -> 后一天"""
        query = """
        MATCH (r1:WeatherRecord), (r2:WeatherRecord)
        WHERE r2.date = toString(date(r1.date) + duration('P1D'))
        MERGE (r1)-[:NEXT_DAY]->(r2)
        """
        with self.driver.session() as session:
            result = session.run(query)
            print("已创建时间序列关系 (NEXT_DAY)")

    def create_monthly_nodes(self):
        """创建月份聚合节点"""
        query = """
        MATCH (r:WeatherRecord)
        WITH r.month AS month,
             avg(r.maxTemp) AS avgMax,
             avg(r.minTemp) AS avgMin,
             sum(r.precipitation) AS totalPrecip,
             avg(r.windSpeed) AS avgWind,
             count(r) AS days
        MERGE (m:Month {id: month})
        SET m.avgMaxTemp = round(avgMax, 1),
            m.avgMinTemp = round(avgMin, 1),
            m.totalPrecipitation = round(totalPrecip, 1),
            m.avgWindSpeed = round(avgWind, 1),
            m.recordCount = days
        """
        with self.driver.session() as session:
            session.run(query)
        print("已创建月份聚合节点")

        # 创建记录属于某月的关系
        query2 = """
        MATCH (r:WeatherRecord), (m:Month {id: r.month})
        MERGE (r)-[:IN_MONTH]->(m)
        """
        with self.driver.session() as session:
            session.run(query2)
        print("已创建记录-月份关系")

    def import_llm_knowledge(self, model_key):
        """导入大模型抽取的知识"""
        filepath = os.path.join(os.path.dirname(__file__), f"knowledge_{model_key}.json")
        if not os.path.exists(filepath):
            print(f"未找到 {filepath}，跳过")
            return

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        parsed = data.get("parsed")
        if not parsed:
            print(f"{model_key} 的抽取结果解析失败，跳过")
            return

        # 导入实体（跳过与已有节点冲突的）
        entities = parsed.get("entities", [])
        imported = 0
        for ent in entities:
            ent_type = ent.get("type", "Unknown").replace(" ", "")
            props = ent.get("properties", {})
            props["name"] = ent.get("name", ent.get("id", ""))
            props["source_model"] = data["model"]
            try:
                with self.driver.session() as session:
                    session.run(
                        f"MERGE (e:{ent_type} {{name: $name}}) "
                        f"SET e += $props",
                        name=props["name"],
                        props=props,
                    )
                imported += 1
            except Exception:
                pass  # 跳过与已有约束冲突的实体
        print(f"  {model_key}: 导入 {imported}/{len(entities)} 个实体")

        # 导入关系
        relationships = parsed.get("relationships", [])
        rel_imported = 0
        for rel in relationships:
            rel_type = rel.get("type", "RELATED_TO").upper().replace(" ", "_")
            props = rel.get("properties", {})
            props["source_model"] = data["model"]
            try:
                with self.driver.session() as session:
                    session.run(
                        f"MATCH (a {{name: $src}}), (b {{name: $tgt}}) "
                        f"MERGE (a)-[r:{rel_type}]->(b) "
                        f"SET r += $props",
                        src=rel.get("source", ""),
                        tgt=rel.get("target", ""),
                        props=props,
                    )
                rel_imported += 1
            except Exception:
                pass
        print(f"  {model_key}: 导入 {rel_imported}/{len(relationships)} 条关系")

        # 存储抽取的规律
        rules = parsed.get("rules", [])
        for i, rule in enumerate(rules):
            with self.driver.session() as session:
                rule_text = rule if isinstance(rule, str) else rule.get("description", str(rule))
                session.run(
                    "MERGE (rule:Rule {id: $id}) "
                    "SET rule.description = $desc, rule.source = $src, rule.model = $model",
                    id=f"{model_key}_rule_{i}",
                    desc=rule_text,
                    src=model_key,
                    model=data["model"],
                )
        print(f"  {model_key}: 导入 {len(rules)} 条规律")

    def get_stats(self):
        """获取数据库统计"""
        with self.driver.session() as session:
            result = session.run("MATCH (n) RETURN labels(n)[0] AS label, count(n) AS cnt ORDER BY cnt DESC")
            print(f"\n{'='*40}")
            print("  Neo4j 数据库统计")
            print(f"{'='*40}")
            for record in result:
                print(f"  {record['label']:<20} {record['cnt']:>6} 个节点")

            result = session.run("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS cnt ORDER BY cnt DESC")
            print()
            for record in result:
                print(f"  {record['type']:<20} {record['cnt']:>6} 条关系")


def main():
    # 读取 CSV
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"读取 {len(rows)} 条天气记录")

    importer = Neo4jImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # 1. 清空并初始化
        importer.clear_database()
        importer.create_constraints()

        # 2. 导入天气记录
        print("\n--- 导入天气数据 ---")
        importer.import_weather_records(rows)
        importer.import_weather_events(rows)

        # 3. 创建时间关系
        print("\n--- 创建时间关系 ---")
        importer.create_temporal_relations()
        importer.create_monthly_nodes()

        # 4. 导入大模型抽取的知识
        print("\n--- 导入大模型知识 ---")
        for model_key in ["deepseek", "qwen", "mimo"]:
            importer.import_llm_knowledge(model_key)

        # 5. 统计
        importer.get_stats()

        print(f"\n导入完成！打开 Neo4j Browser 查看: http://localhost:7474")

    finally:
        importer.close()


if __name__ == "__main__":
    main()
