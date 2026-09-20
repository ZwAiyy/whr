import pandas as pd
from datetime import datetime

def analyze_grape_disasters(df):
    """
    分析葡萄气象灾害

    葡萄主要气象灾害：
    1. 霜冻灾害 (春季和秋季)
    2. 高温热害 (夏季)
    3. 干旱灾害
    4. 连阴雨灾害
    5. 大风灾害
    """

    print("=" * 60)
    print("葡萄气象灾害分析报告")
    print("=" * 60)

    # 1. 霜冻灾害分析 (春季: 3-5月, 秋季: 9-10月)
    spring_frost = df[(df['date'].dt.month.isin([3, 4, 5])) & (df['frost_day'] == True)]
    autumn_frost = df[(df['date'].dt.month.isin([9, 10])) & (df['frost_day'] == True)]

    print("\n1. 霜冻灾害分析")
    print(f"   春季霜冻天数 (3-5月): {len(spring_frost)} 天")
    if len(spring_frost) > 0:
        print(f"   春季霜冻日期: {spring_frost['date'].dt.strftime('%Y-%m-%d').tolist()}")
    print(f"   秋季霜冻天数 (9-10月): {len(autumn_frost)} 天")
    if len(autumn_frost) > 0:
        print(f"   秋季霜冻日期: {autumn_frost['date'].dt.strftime('%Y-%m-%d').tolist()}")

    # 2. 高温热害分析 (夏季: 6-8月)
    summer_heat = df[(df['date'].dt.month.isin([6, 7, 8])) & (df['heat_day'] == True)]
    severe_summer_heat = df[(df['date'].dt.month.isin([6, 7, 8])) & (df['severe_heat_day'] == True)]

    print("\n2. 高温热害分析 (夏季)")
    print(f"   高温天数 (≥35°C): {len(summer_heat)} 天")
    if len(summer_heat) > 0:
        print(f"   高温日期: {summer_heat['date'].dt.strftime('%Y-%m-%d').tolist()}")
    print(f"   严重高温天数 (≥38°C): {len(severe_summer_heat)} 天")

    # 3. 干旱灾害分析
    total_dry_days = df['dry_day'].sum()
    max_drought_streak = df['dry_streak'].max()

    print("\n3. 干旱灾害分析")
    print(f"   干旱天数 (日降水 < 1mm): {total_dry_days} 天")
    print(f"   最长连续干旱天数: {max_drought_streak} 天")

    # 找出最长干旱期
    if max_drought_streak >= 7:
        drought_periods = df[df['dry_streak'] >= 7]
        if not drought_periods.empty:
            print(f"   长期干旱期 (≥7天):")
            for idx, row in drought_periods.iterrows():
                if row['dry_streak'] == 1:  # 干旱期开始
                    start_date = row['date']
                    end_date = start_date + pd.Timedelta(days=int(row['dry_streak']) - 1)
                    print(f"     {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")

    # 4. 连阴雨灾害分析
    continuous_rain = df[df['continuous_rain_streak'] >= 3]
    max_rain_streak = df['continuous_rain_streak'].max()

    print("\n4. 连阴雨灾害分析")
    print(f"   连续降水天数 (≥3天): {len(continuous_rain)} 天")
    print(f"   最长连续降水天数: {max_rain_streak} 天")

    # 5. 大风灾害分析
    strong_wind_days = df['strong_wind_day'].sum()

    print("\n5. 大风灾害分析")
    print(f"   大风天数 (≥17m/s): {strong_wind_days} 天")

    # 6. 综合灾害风险评估
    print("\n6. 综合灾害风险评估")
    high_risk_days = len(df[
        (df['frost_day']) |
        (df['heat_day']) |
        (df['dry_streak'] >= 5) |
        (df['strong_wind_day'])
    ])
    print(f"   高风险天数 (霜冻/高温/干旱/大风): {high_risk_days} 天")
    print(f"   高风险天数占比: {high_risk_days / len(df) * 100:.1f}%")

    # 7. 月份统计
    print("\n7. 各月灾害统计")
    df['month'] = df['date'].dt.month
    monthly_stats = df.groupby('month').agg({
        'frost_day': 'sum',
        'heat_day': 'sum',
        'dry_day': 'sum',
        'strong_wind_day': 'sum'
    }).round(0)

    month_names = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
    for month in range(1, 13):
        if month in monthly_stats.index:
            stats = monthly_stats.loc[month]
            print(f"   {month_names[month-1]}: 霜冻={int(stats['frost_day'])}天, 高温={int(stats['heat_day'])}天, 干旱={int(stats['dry_day'])}天, 大风={int(stats['strong_wind_day'])}天")
        else:
            print(f"   {month_names[month-1]}: 霜冻=0天, 高温=0天, 干旱=0天, 大风=0天")

    return {
        'spring_frost_days': len(spring_frost),
        'autumn_frost_days': len(autumn_frost),
        'summer_heat_days': len(summer_heat),
        'severe_heat_days': len(severe_summer_heat),
        'total_dry_days': total_dry_days,
        'max_drought_streak': max_drought_streak,
        'max_rain_streak': max_rain_streak,
        'strong_wind_days': strong_wind_days,
        'high_risk_days': high_risk_days
    }

if __name__ == "__main__":
    # 读取数据
    try:
        df = pd.read_csv('weather_data.csv', parse_dates=['date'])
        results = analyze_grape_disasters(df)
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
    except FileNotFoundError:
        print("错误: 未找到 weather_data.csv 文件")
        print("请先运行 API.py 获取气象数据")
