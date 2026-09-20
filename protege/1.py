import pandas as pd
from owlready2 import *
import datetime

# 1. 创建本体（使用 get_ontology，不是 Ontology）
onto = get_ontology("http://example.org/weather-ontology")

# 2. 定义类
class Record(Thing):
    namespace = onto

# 3. 定义数据属性（指定数据类型范围）
class has_temperature_max(DataProperty, FunctionalProperty):
    namespace = onto
    range = [float]

class has_temperature_min(DataProperty, FunctionalProperty):
    namespace = onto
    range = [float]

class has_precipitation(DataProperty, FunctionalProperty):
    namespace = onto
    range = [float]

class has_wind_speed_max(DataProperty, FunctionalProperty):
    namespace = onto
    range = [float]

class has_humidity(DataProperty, FunctionalProperty):
    namespace = onto
    range = [int]

class has_date(DataProperty, FunctionalProperty):
    namespace = onto
    range = [datetime.date]

class has_frost(DataProperty, FunctionalProperty):
    namespace = onto
    range = [bool]

class has_heat(DataProperty, FunctionalProperty):
    namespace = onto
    range = [bool]

class has_severe_heat(DataProperty, FunctionalProperty):
    namespace = onto
    range = [bool]

class has_dry(DataProperty, FunctionalProperty):
    namespace = onto
    range = [bool]

class has_dry_streak(DataProperty, FunctionalProperty):
    namespace = onto
    range = [int]

class has_strong_wind(DataProperty, FunctionalProperty):
    namespace = onto
    range = [bool]

class has_dust_storm(DataProperty, FunctionalProperty):
    namespace = onto
    range = [bool]

class has_rainy(DataProperty, FunctionalProperty):
    namespace = onto
    range = [bool]

class has_continuous_rain_streak(DataProperty, FunctionalProperty):
    namespace = onto
    range = [int]

# 4. 读取CSV
df = pd.read_csv("weather_data.csv")

# 5. 遍历每一行，创建个体
for index, row in df.iterrows():
    date_str = row["date"]
    individual_name = f"Record_{date_str}"
    ind = Record(individual_name)

    ind.has_temperature_max = float(row["temperature_2m_max"])
    ind.has_temperature_min = float(row["temperature_2m_min"])
    ind.has_precipitation = float(row["precipitation_sum"])
    ind.has_wind_speed_max = float(row["wind_speed_10m_max"])
    ind.has_humidity = int(row["relative_humidity_2m_mean"])
    ind.has_date = datetime.datetime.strptime(row["date"], "%Y-%m-%d").date()

    ind.has_frost = row["frost_day"] == "True"
    ind.has_heat = row["heat_day"] == "True"
    ind.has_severe_heat = row["severe_heat_day"] == "True"
    ind.has_dry = row["dry_day"] == "True"
    ind.has_dry_streak = int(row["dry_streak"])
    ind.has_strong_wind = row["strong_wind_day"] == "True"
    ind.has_dust_storm = row["dust_storm_risk"] == "True"
    ind.has_rainy = row["rainy_day"] == "True"
    ind.has_continuous_rain_streak = int(row["continuous_rain_streak"])

# 6. 保存为OWL文件
onto.save("weather_ontology.owl", format="rdfxml")

print("OWL文件已生成：weather_ontology.owl")
