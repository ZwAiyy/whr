import rdflib

NAMESPACE = "http://example.org/weather-ontology#"
XSD = "http://www.w3.org/2001/XMLSchema#"

def run_query(g, title, query):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    results = g.query(query)
    for row in results:
        print("  ", " | ".join(str(v) for v in row))

def main():
    g = rdflib.Graph()
    g.parse(".\protege\weather_ontology.owl", format="xml")
    print(f"加载本体: {len(g)} 条三元组")

    # 1. 最高温超过 35°C 的日期
    run_query(g, "最高温超过 35°C 的日期", f"""
        PREFIX ns: <{NAMESPACE}>
        SELECT ?date ?maxTemp
        WHERE {{
            ?r a ns:WeatherRecord .
            ?r ns:hasDate ?date .
            ?r ns:hasMaxTemp ?maxTemp .
            FILTER(?maxTemp > 35.0)
        }}
        ORDER BY DESC(?maxTemp)
    """)

    # 2. 统计雨天数量
    run_query(g, "雨天总数", f"""
        PREFIX ns: <{NAMESPACE}>
        SELECT (COUNT(?r) AS ?count)
        WHERE {{
            ?r a ns:WeatherRecord .
            ?r ns:hasWeatherEvent ns:RainyDay_instance .
        }}
    """)

    # 3. 统计各类极端天气事件次数
    run_query(g, "各类极端天气事件发生次数", f"""
        PREFIX ns: <{NAMESPACE}>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?eventName (COUNT(?r) AS ?count)
        WHERE {{
            ?r a ns:WeatherRecord .
            ?r ns:hasWeatherEvent ?event .
            ?event rdfs:label ?eventName .
        }}
        GROUP BY ?eventName
        ORDER BY DESC(?count)
    """)

    # 4. 连续干旱天数最长的记录
    run_query(g, "连续干旱天数 TOP 5", f"""
        PREFIX ns: <{NAMESPACE}>
        SELECT ?date ?dryStreak
        WHERE {{
            ?r a ns:WeatherRecord .
            ?r ns:hasDate ?date .
            ?r ns:hasDryStreak ?dryStreak .
        }}
        ORDER BY DESC(?dryStreak)
        LIMIT 5
    """)

    # 5. 降水量最大的日期
    run_query(g, "降水量 TOP 5", f"""
        PREFIX ns: <{NAMESPACE}>
        SELECT ?date ?precipitation
        WHERE {{
            ?r a ns:WeatherRecord .
            ?r ns:hasDate ?date .
            ?r ns:hasPrecipitation ?precipitation .
            FILTER(?precipitation > 0)
        }}
        ORDER BY DESC(?precipitation)
        LIMIT 5
    """)

if __name__ == "__main__":
    main()
