import requests
import pandas as pd

def fetch_weather_data(latitude=38.5, longitude=106.2, start_date="2025-06-08", end_date="2026-06-08"):
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    daily_params = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",#日降水总合
        "wind_speed_10m_max",
        "relative_humidity_2m_mean",#相对湿度
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": daily_params,
        "timezone": "auto",
    }

    print(f"正在请求API: {url}")
    print(f"参数: latitude={latitude}, longitude={longitude}")
    print(f"日期范围: {start_date} 至 {end_date}")
    print(f"请求的气象参数: {', '.join(daily_params)}")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()  # 检查HTTP错误

        data = response.json()

        #打印API响应结构
        print(f"\nAPI响应状态码: {response.status_code}")
        print(f"响应键名: {list(data.keys())}")

        # 检查是否有错误
        if 'error' in data:
            print(f"API错误: {data.get('error_message', 'Unknown error')}")
            return None

        # 检查是否有daily数据
        if 'daily' not in data:
            print("错误: API响应中没有'daily'数据")
            print(f"完整响应: {data}")
            return None

        # 转换为DataFrame
        daily = data['daily']
        df = pd.DataFrame(daily)
        df['date'] = pd.to_datetime(df['time'])
        df.drop(columns=['time'], inplace=True)

        # 计算衍生指标
        df = calculate_disaster_indicators(df)

        return df

    except requests.exceptions.RequestException as e:
        print(f"网络请求错误: {e}")
        return None
    except Exception as e:
        print(f"处理数据时出错: {e}")
        return None

def calculate_disaster_indicators(df):
    # 霜冻 (温度 ≤ 0°C)
    df['frost_day'] = df['temperature_2m_min'] <= 0

    # 高温灾害 (温度 ≥ 35°C)
    df['heat_day'] = df['temperature_2m_max'] >= 35

    # 严重高温 (温度 ≥ 38°C)
    df['severe_heat_day'] = df['temperature_2m_max'] >= 38

    # 干旱 (日降水 < 1mm)
    df['dry_day'] = df['precipitation_sum'] < 1

    df['dry_streak'] = df['dry_day'].groupby((~df['dry_day']).cumsum()).cumcount() + 1

    # 大风指标 (风速 ≥ 17m/s)
    df['strong_wind_day'] = df['wind_speed_10m_max'] >= 17

    # 沙尘暴风险 (大风 + 低湿度)
    df['dust_storm_risk'] = (df['wind_speed_10m_max'] >= 15) & (df['relative_humidity_2m_mean'] <= 30)

    # 连阴雨 (连续降水 ≥ 3天)
    df['rainy_day'] = df['precipitation_sum'] >= 1
    df['continuous_rain_streak'] = df['rainy_day'].groupby((~df['rainy_day']).cumsum()).cumcount() + 1
    df.loc[~df['rainy_day'], 'continuous_rain_streak'] = 0
    return df

def save_to_csv(df, filename='weather_data_orign.csv'):
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    start_date = "2023-06-08"
    end_date = "2026-06-08"

    print(f"正在获取数据: {start_date} 到 {end_date}")
    df = fetch_weather_data(start_date=start_date, end_date=end_date)

    if df is not None and len(df) > 0:
        save_to_csv(df, 'weather_data.csv')
        print(df[['date', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum', 'wind_speed_10m_max']].head(10))
    else:
        print("\n无法获取或处理数据")
