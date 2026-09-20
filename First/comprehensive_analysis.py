import pandas as pd
from datetime import datetime

def comprehensive_disaster_report(df):
    # 1. 霜冻灾害
    spring_frost = df[(df['date'].dt.month.isin([3, 4, 5])) & (df['frost_day'] == True)]
    autumn_frost = df[(df['date'].dt.month.isin([9, 10])) & (df['frost_day'] == True)]

    print(f"\n1. 霜冻灾害")
    print(f"   春季霜冻 (3-5月): {len(spring_frost)} 天")
    print(f"   秋季霜冻 (9-10月): {len(autumn_frost)} 天")
    print(f"   全年霜冻总计: {len(spring_frost) + len(autumn_frost)} 天")

    # 2. 高温灾害
    summer_heat = df[(df['date'].dt.month.isin([6, 7, 8])) & (df['heat_day'] == True)]
    severe_heat = df[(df['date'].dt.month.isin([6, 7, 8])) & (df['severe_heat_day'] == True)]

    print(f"\n2. 高温灾害 (夏季)")
    print(f"   高温天数 (≥35°C): {len(summer_heat)} 天")
    print(f"   严重高温天数 (≥38°C): {len(severe_heat)} 天")

    # 3. 干旱灾害
    total_dry_days = df['dry_day'].sum()
    max_drought = df['dry_streak'].max()

    print(f"\n3. 干旱灾害")
    print(f"   干旱天数 (日降水 < 1mm): {total_dry_days} 天")
    print(f"   最长连续干旱: {max_drought} 天")

    # 4. 大风灾害
    strong_wind = df['strong_wind_day'].sum()

    print(f"\n4. 大风灾害")
    print(f"   大风天数 (≥17m/s): {strong_wind} 天")

    # 葡萄关键期灾害
    grape_key_periods = {
        '萌芽期 (3-4月)': df[(df['date'].dt.month.isin([3, 4]))],
        '开花期 (5-6月)': df[(df['date'].dt.month.isin([5, 6]))],
        '果实膨大期 (7-8月)': df[(df['date'].dt.month.isin([7, 8]))],
        '成熟期 (9-10月)': df[(df['date'].dt.month.isin([9, 10]))]
    }

    for period_name, period_data in grape_key_periods.items():
        frost = period_data['frost_day'].sum()
        heat = period_data['heat_day'].sum()
        dry = period_data['dry_day'].sum()
        wind = period_data['strong_wind_day'].sum()

        print(f"\n{period_name}:")
        print(f"  霜冻: {frost}天, 高温: {heat}天, 干旱: {dry}天, 大风: {wind}天")


    # 枸杞关键期灾害
    goji_key_periods = {
        '萌芽期 (3-4月)': df[(df['date'].dt.month.isin([3, 4]))],
        '花期 (5-6月)': df[(df['date'].dt.month.isin([5, 6]))],
        '幼果期 (7-8月)': df[(df['date'].dt.month.isin([7, 8]))],
        '成熟期 (9-10月)': df[(df['date'].dt.month.isin([9, 10]))]
    }

    for period_name, period_data in goji_key_periods.items():
        frost = period_data['frost_day'].sum()
        heat = period_data['heat_day'].sum()
        dry = period_data['dry_day'].sum()
        wind = period_data['strong_wind_day'].sum()
        dust = period_data['dust_storm_risk'].sum()

        print(f"\n{period_name}:")
        print(f"  霜冻: {frost}天, 高温: {heat}天, 干旱: {dry}天, 大风: {wind}天, 沙尘: {dust}天")

    # 沙尘暴专项分析
    spring_dust = df[(df['date'].dt.month.isin([3, 4, 5])) & (df['dust_storm_risk'] == True)]
    print(f"\n沙尘暴风险 (春季3-5月): {len(spring_dust)} 天")

    # 综合风险评估
    print("\n" + "=" * 70)
    print("【综合风险评估】")
    print("-" * 70)

    # 高风险天数统计
    high_risk_criteria = (
        (df['frost_day']) |
        (df['heat_day']) |
        (df['dry_streak'] >= 5) |
        (df['strong_wind_day']) |
        (df['dust_storm_risk'])
    )
    high_risk_days = len(df[high_risk_criteria])

    print(f"\n高风险天数总计: {high_risk_days} 天 ({high_risk_days / len(df) * 100:.1f}%)")

    # 各月灾害分布
    print("\n" + "=" * 70)
    print("【各月灾害分布】")
    print("-" * 70)
    print("\n月份 | 霜冻 | 高温 | 干旱 | 大风 | 沙尘 |")
    print("-" * 50)

    df['month'] = df['date'].dt.month
    monthly = df.groupby('month').agg({
        'frost_day': 'sum',
        'heat_day': 'sum',
        'dry_day': 'sum',
        'strong_wind_day': 'sum',
        'dust_storm_risk': 'sum'
    })

    month_names = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
    for month in range(1, 13):
        if month in monthly.index:
            stats = monthly.loc[month]
            print(f"{month_names[month-1]:4} | {int(stats['frost_day']):4} | {int(stats['heat_day']):4} | {int(stats['dry_day']):4} | {int(stats['strong_wind_day']):4} | {int(stats['dust_storm_risk']):4} |")
        else:
            print(f"{month_names[month-1]:4} | {'0':4} | {'0':4} | {'0':4} | {'0':4} | {'0':4} |")

if __name__ == "__main__":
    try:
        df = pd.read_csv('weather_data.csv', parse_dates=['date'])
        comprehensive_disaster_report(df)
    except FileNotFoundError:
        print("错误: 未找到文件")
