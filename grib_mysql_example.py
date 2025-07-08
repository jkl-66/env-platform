#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRIB文件到MySQL数据库存储 - 使用示例
"""

import json
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()
from grib_to_mysql import GRIBToMySQLProcessor

def load_config(config_file='mysql_config.json'):
    """
    加载配置文件
    """
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return None
        
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    """
    主函数 - 演示如何使用GRIB到MySQL处理器
    """
    print("🌡️ GRIB文件到MySQL数据库存储示例")
    print("=" * 50)
    
    # 1. 加载配置
    config = load_config()
    if not config:
        print("❌ 无法加载配置文件")
        return
    
    mysql_config = config['mysql']

    # 优先从环境变量加载数据库配置，以覆盖json文件中的配置
    mysql_config['user'] = os.getenv('DB_USER', mysql_config.get('user'))
    mysql_config['password'] = os.getenv('DB_PASSWORD', mysql_config.get('password'))
    mysql_config['host'] = os.getenv('DB_HOST', mysql_config.get('host'))
    mysql_config['port'] = int(os.getenv('DB_PORT', mysql_config.get('port')))
    mysql_config['database'] = os.getenv('DB_NAME', 'summer_predict') # 强制指定数据库
    grib_settings = config['grib_settings']
    
    # 2. 设置GRIB文件路径
    # 重要提示: 请将您的GRIB文件移动到项目根目录并重命名为 data.grib
    grib_file_path = "data.grib"
    
    print(f"📂 GRIB文件路径: {grib_file_path}")
    print(f"🗄️ 数据库: {mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}")
    
    try:
        # 3. 创建处理器
        print("\n🔧 初始化处理器...")
        processor = GRIBToMySQLProcessor(mysql_config)
        
        # 创建数据库表 (如果不存在)
        processor.create_database_tables()
        
        # 4. 加载GRIB文件
        print("\n📂 加载GRIB文件...")
        if not processor.load_grib_file(grib_file_path):
            print("❌ GRIB文件加载失败")
            return
        
        # 5. 处理数据
        print("\n🔧 处理GRIB数据...")
        processed_data = processor.process_grib_data()
        
        print(f"\n📊 数据处理完成:")
        print(f"   - 总记录数: {len(processed_data):,}")
        print(f"   - 变量数量: {processed_data['variable'].nunique()}")
        print(f"   - 变量列表: {', '.join(processed_data['variable'].unique())}")
        print(f"   - 时间范围: {processed_data['time'].min()} 到 {processed_data['time'].max()}")
        
        # 6. 显示数据预览
        print("\n📋 数据预览:")
        print(processed_data.head(10))
        
        # 7. 保存到数据库
        print("\n💾 保存到MySQL数据库...")
        batch_size = grib_settings.get('batch_size', 5000)
        
        if processor.save_to_database(batch_size=batch_size):
            print("✅ 数据保存成功!")
            
            # 8. 获取数据库统计信息
            print("\n📊 获取数据库统计信息...")
            stats = processor.get_statistics()
            
            print("\n📈 数据库统计:")
            print(f"   - 总记录数: {stats.get('total_records', 0):,}")
            
            if 'variables' in stats:
                print("   - 变量统计:")
                for var_stat in stats['variables']:
                    print(f"     * {var_stat['variable']}: {var_stat['count']:,} 条记录")
            
            if 'time_range' in stats:
                time_range = stats['time_range']
                print(f"   - 时间范围: {time_range['start']} 到 {time_range['end']}")
            
            if 'spatial_range' in stats:
                spatial = stats['spatial_range']
                print(f"   - 空间范围: 纬度 {spatial['lat_min']:.2f}°~{spatial['lat_max']:.2f}°, 经度 {spatial['lon_min']:.2f}°~{spatial['lon_max']:.2f}°")
            
            # 9. 示例查询
            print("\n🔍 示例查询...")
            
            # 查询温度数据
            temp_data = processor.query_data(
                variables=['t2m'],
                limit=5
            )
            
            if not temp_data.empty:
                print("\n🌡️ 温度数据示例:")
                print(temp_data)
            
            # 查询特定区域数据
            region_data = processor.query_data(
                lat_range=(30.0, 32.0),  # 上海地区
                lon_range=(120.0, 122.0),
                limit=5
            )
            
            if not region_data.empty:
                print("\n🗺️ 上海地区数据示例:")
                print(region_data)
                
        else:
            print("❌ 数据保存失败")
        
        # 10. 关闭连接
        processor.close()
        print("\n✅ 处理完成!")
        
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def query_example():
    """
    查询示例函数
    """
    print("\n🔍 数据查询示例")
    print("=" * 30)
    
    # 加载配置
    config = load_config()
    if not config:
        return
    
    try:
        # 创建处理器（仅用于查询）
        processor = GRIBToMySQLProcessor(config['mysql'])
        
        # 示例1: 查询最新数据
        print("\n📅 查询最新10条记录:")
        latest_data = processor.query_data(limit=10)
        print(latest_data[['time', 'latitude', 'longitude', 'variable', 'value']].head())
        
        # 示例2: 查询特定变量
        print("\n🌡️ 查询温度数据:")
        temp_data = processor.query_data(
            variables=['t2m'],
            limit=5
        )
        if not temp_data.empty:
            print(temp_data[['time', 'latitude', 'longitude', 'value']].head())
        
        # 示例3: 查询特定时间范围
        print("\n📆 查询2024年数据:")
        time_range_data = processor.query_data(
            start_time='2024-01-01',
            end_time='2024-12-31',
            limit=5
        )
        if not time_range_data.empty:
            print(time_range_data[['time', 'variable', 'value']].head())
        
        # 示例4: 查询特定区域
        print("\n🗺️ 查询北京地区数据:")
        beijing_data = processor.query_data(
            lat_range=(39.5, 40.5),
            lon_range=(116.0, 117.0),
            limit=5
        )
        if not beijing_data.empty:
            print(beijing_data[['latitude', 'longitude', 'variable', 'value']].head())
        
        processor.close()
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")

if __name__ == "__main__":
    # 运行主程序
    main()
    
    # 运行查询示例
    # query_example()