import pandas as pd
from datetime import datetime

def analyze_goji_disasters(df):
    """
    分析枸杞气象灾害

    枸杞主要气象灾害：
    1. 霜冻灾害 (春季和秋季)
    2. 高温热害 (夏季)
    3. 干旱灾害
    4. 大风灾害
    5. 沙尘暴灾害
    """

    print("=" * 60)
    print("枸杞气象灾害分析报告")
    print("=" * 60)

    # 1. 霜冻灾害分析 (春季: 3-5月, 秋季: 9-10月)
    # 枸杞萌芽期 (3-4月) 和果实成熟期 (9-10月) 对霜冻敏感
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
    # 枸杞花期和幼果期对高温敏感
    summer_heat = df[(df['date'].dt.month.isin([6, 7, 8])) & (df['heat_day'] == True)]
    severe_summer_heat = df[(df['date'].dt.month.isin([6, 7, 8])) & (df['severe_heat_day'] == True)]

    print("\n2. 高温热害分析 (夏季)")
    print(f"   高温天数 (≥35°C): {len(summer_heat)} 天")
    if len(summer_heat) > 0:
        print(f"   高温日期: {summer_heat['date'].dt.strftime('%Y-%m-%d').tolist()}")
    print(f"   严重高温天数 (≥38°C): {len(severe_summer_heat)} 天")

    # 3. 干旱灾害分析
    # 枸杞耐旱但花期和幼果期需要适量水分
    total_dry_days = df['dry_day'].sum()
    max_drought_streak = df['dry_streak'].max()

    print("\n3. 干旱灾害分析")
    print(f"   干旱天数 (日降水 < 1mm): {total_dry_days} 天")
    print(f"   最长连续干旱天数: {max_drought_streak} 天")

    # 找出花期干旱 (5-6月)
    flowering_dry = df[(df['date'].dt.month.isin([5, 6])) & (df['dry_day'] == True)]
    print(f"   花期干旱天数 (5-6月): {len(flowering_dry)} 天")

    # 4. 大风灾害分析
    # 枸杞枝条脆弱，大风易造成机械损伤
    strong_wind_days = df['strong_wind_day'].sum()

    print("\n4. 大风灾害分析")
    print(f"   大风天数 (≥17m/s): {strong_wind_days} 天")

    # 5. 沙尘暴灾害分析
    # 宁夏地区春季沙尘暴频发
    dust_storm_days = df['dust_storm_risk'].sum()
    spring_dust = df[(df['date'].dt.month.isin([3, 4, 5])) & (df['dust_storm_risk'] == True)]

    print("\n5. 沙尘暴灾害分析")
    print(f"   沙尘暴风险天数: {dust_storm_days} 天")
    print(f"   春季沙尘暴风险天数 (3-5月): {len(spring_dust)} 天")
    if len(spring_dust) > 0:
        print(f"   沙尘暴风险日期: {spring_dust['date'].dt.strftime('%Y-%m-%d').tolist()}")

    # 6. 连阴雨灾害分析
    # 枸杞成熟期需要晴朗天气
    continuous_rain = df[df['continuous_rain_streak'] >= 3]
    max_rain_streak = df['continuous_rain_streak'].max()

    print("\n6. 连阴雨灾害分析")
    print(f"   连续降水天数 (≥3天): {len(continuous_rain)} 天")
    print(f"   最长连续降水天数: {max_rain_streak} 天")

    # 7. 综合灾害风险评估
    print("\n7. 综合灾害风险评估")
    high_risk_days = len(df[
        (df['frost_day']) |
        (df['heat_day']) |
        (df['dry_streak'] >= 5) |
        (df['strong_wind_day']) |
        (df['dust_storm_risk'])
    ])
    print(f"   高风险天数 (霜冻/高温/干旱/大风/沙尘): {high_risk_days} 天")
    print(f"   高风险天数占比: {high_risk_days / len(df) * 100:.1f}%")

    # 8. 生长季灾害统计 (4-10月)
    growing_season = df[(df['date'].dt.month.isin([4, 5, 6, 7, 8, 9, 10]))]
    print("\n8. 生长季灾害统计 (4-10月)")
    print(f"   生长季总天数: {len(growing_season)} 天")
    print(f"   生长季霜冻天数: {growing_season['frost_day'].sum()} 天")
    print(f"   生长季高温天数: {growing_season['heat_day'].sum()} 天")
    print(f"   生长季干旱天数: {growing_season['dry_day'].sum()} 天")
    print(f"   生长季大风天数: {growing_season['strong_wind_day'].sum()} 天")

    # 9. 月份统计
    print("\n9. 各月灾害统计")
    df['month'] = df['date'].dt.month
    monthly_stats = df.groupby('month').agg({
        'frost_day': 'sum',
        'heat_day': 'sum',
        'dry_day': 'sum',
        'strong_wind_day': 'sum',
        'dust_storm_risk': 'sum'
    }).round(0)

    month_names = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
    for month in range(1, 13):
        if month in monthly_stats.index:
            stats = monthly_stats.loc[month]
            print(f"   {month_names[month-1]}: 霜冻={int(stats['frost_day'])}天, 高温={int(stats['heat_day'])}天, 干旱={int(stats['dry_day'])}天, 大风={int(stats['strong_wind_day'])}天, 沙尘={int(stats['dust_storm_risk'])}天")
        else:
            print(f"   {month_names[month-1]}: 霜冻=0天, 高温=0天, 干旱=0天, 大风=0天, 沙尘=0天")

    return {
        'spring_frost_days': len(spring_frost),
        'autumn_frost_days': len(autumn_frost),
        'summer_heat_days': len(summer_heat),
        'severe_heat_days': len(severe_summer_heat),
        'total_dry_days': total_dry_days,
        'flowering_dry_days': len(flowering_dry),
        'max_drought_streak': max_drought_streak,
        'strong_wind_days': strong_wind_days,
        'dust_storm_days': dust_storm_days,
        'spring_dust_days': len(spring_dust),
        'max_rain_streak': max_rain_streak,
        'high_risk_days': high_risk_days
    }

if __name__ == "__main__":
    # 读取数据
    try:
        df = pd.read_csv('weather_data.csv', parse_dates=['date'])
        results = analyze_goji_disasters(df)
        print("\n" + "=" * 60)
        print("分析完成！")
        print("=" * 60)
    except FileNotFoundError:
        print("错误: 未找到 weather_data.csv 文件")
        print("请先运行 API.py 获取气象数据")
