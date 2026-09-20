import csv
import os
from xml.sax.saxutils import escape

CSV_FILE = os.path.join(os.path.dirname(__file__), "weather_data.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "weather_ontology.owl")
NAMESPACE = "http://example.org/weather-ontology#"
XSD = "http://www.w3.org/2001/XMLSchema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"

# 布尔列 -> 事件类名
BOOLEAN_COLUMNS = {
    "frost_day": "FrostDay",
    "heat_day": "HeatDay",
    "severe_heat_day": "SevereHeatDay",
    "dry_day": "DryDay",
    "strong_wind_day": "StrongWindDay",
    "dust_storm_risk": "DustStormRisk",
    "rainy_day": "RainyDay",
}

# 数据属性: (属性名, CSV列名, XSD类型)
DATA_PROPERTIES = [
    ("hasMaxTemp", "temperature_2m_max", "float"),
    ("hasMinTemp", "temperature_2m_min", "float"),
    ("hasPrecipitation", "precipitation_sum", "float"),
    ("hasWindSpeed", "wind_speed_10m_max", "float"),
    ("hasHumidity", "relative_humidity_2m_mean", "integer"),
    ("hasDate", "date", "date"),
    ("hasDryStreak", "dry_streak", "integer"),
    ("hasRainStreak", "continuous_rain_streak", "integer"),
]


def iri(local_name):
    return f"{NAMESPACE}{local_name}"


def generate_owl():
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"读取到 {len(rows)} 条天气记录")

    lines = []

    # === XML 头部 ===
    lines.append('<?xml version="1.0"?>')
    lines.append('<rdf:RDF')
    lines.append(f'    xmlns:rdf="{RDF_NS}"')
    lines.append(f'    xmlns:rdfs="{RDFS_NS}"')
    lines.append(f'    xmlns:owl="{OWL_NS}"')
    lines.append(f'    xmlns:xsd="{XSD}"')
    lines.append(f'    xmlns:ns="{NAMESPACE}">')
    lines.append("")
    lines.append("  <!-- 本体声明 -->")
    lines.append(f'  <owl:Ontology rdf:about="{NAMESPACE.rstrip("#")}">')
    lines.append("    <rdfs:comment>从 weather_data.csv 自动生成的天气本体</rdfs:comment>")
    lines.append("  </owl:Ontology>")
    lines.append("")

    # === 类声明 ===
    lines.append("  <!-- ========== 类定义 ========== -->")
    all_classes = ["WeatherRecord", "WeatherEvent"] + list(BOOLEAN_COLUMNS.values())
    for cls in all_classes:
        parent = "owl:Thing" if cls in ("WeatherRecord", "WeatherEvent") else "WeatherEvent"
        if cls == "WeatherRecord":
            parent = "owl:Thing"
        lines.append(f'  <owl:Class rdf:about="{iri(cls)}">')
        if parent != "owl:Thing":
            lines.append(f'    <rdfs:subClassOf rdf:resource="{iri(parent)}"/>')
        lines.append(f"    <rdfs:label>{cls}</rdfs:label>")
        lines.append("  </owl:Class>")
    lines.append("")

    # === 数据属性 ===
    lines.append("  <!-- ========== 数据属性 ========== -->")
    for prop_name, _, xsd_type in DATA_PROPERTIES:
        lines.append(f'  <owl:DatatypeProperty rdf:about="{iri(prop_name)}">')
        lines.append(f'    <rdfs:domain rdf:resource="{iri("WeatherRecord")}"/>')
        lines.append(f'    <rdfs:range rdf:resource="{XSD}{xsd_type}"/>')
        lines.append(f"    <rdfs:label>{prop_name}</rdfs:label>")
        lines.append("  </owl:DatatypeProperty>")
    lines.append("")

    # === 对象属性 ===
    lines.append("  <!-- ========== 对象属性 ========== -->")
    lines.append(f'  <owl:ObjectProperty rdf:about="{iri("hasWeatherEvent")}">')
    lines.append(f'    <rdfs:domain rdf:resource="{iri("WeatherRecord")}"/>')
    lines.append(f'    <rdfs:range rdf:resource="{iri("WeatherEvent")}"/>')
    lines.append("    <rdfs:label>hasWeatherEvent</rdfs:label>")
    lines.append("  </owl:ObjectProperty>")
    lines.append("")

    # === 天气事件实例（每个事件类一个共享实例）===
    lines.append("  <!-- ========== 事件类实例 ========== -->")
    for _, event_cls in BOOLEAN_COLUMNS.items():
        inst_name = f"{event_cls}_instance"
        lines.append(f'  <owl:NamedIndividual rdf:about="{iri(inst_name)}">')
        lines.append(f'    <rdf:type rdf:resource="{iri(event_cls)}"/>')
        lines.append(f"    <rdfs:label>{inst_name}</rdfs:label>")
        lines.append("  </owl:NamedIndividual>")
    lines.append("")

    # === 天气记录实例 ===
    lines.append("  <!-- ========== 天气记录实例 (每行一条) ========== -->")
    for i, row in enumerate(rows):
        date_val = row["date"].strip()
        inst_name = f"Record_{date_val}"

        lines.append(f'  <owl:NamedIndividual rdf:about="{iri(inst_name)}">')
        lines.append(f'    <rdf:type rdf:resource="{iri("WeatherRecord")}"/>')
        lines.append(f"    <rdfs:label>{inst_name}</rdfs:label>")

        # 数据属性
        for prop_name, csv_col, prop_xsd_type in DATA_PROPERTIES:
            val = row.get(csv_col, "").strip()
            if val == "":
                continue
            lines.append(f'    <ns:{prop_name} rdf:datatype="{XSD}{prop_xsd_type}">{escape(val)}</ns:{prop_name}>')

        # 布尔列 -> 对象属性关联事件
        for bool_col, event_cls in BOOLEAN_COLUMNS.items():
            val = row.get(bool_col, "").strip().lower()
            if val == "true":
                lines.append(
                    f'    <ns:hasWeatherEvent rdf:resource="{iri(event_cls + "_instance")}"/>'
                )

        lines.append("  </owl:NamedIndividual>")

    lines.append("")
    lines.append("</rdf:RDF>")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"已生成 OWL 文件: {OUTPUT_FILE}")
    print(f"  - 类: {len(all_classes)} 个")
    print(f"  - 数据属性: {len(DATA_PROPERTIES)} 个")
    print(f"  - 对象属性: 1 个")
    print(f"  - 天气记录实例: {len(rows)} 个")
    print(f"  - 事件实例: {len(BOOLEAN_COLUMNS)} 个")


if __name__ == "__main__":
    generate_owl()
