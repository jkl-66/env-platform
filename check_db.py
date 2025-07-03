import asyncio
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data_processing.data_storage import DataStorage

async def check_database():
    storage = DataStorage()
    await storage.initialize()
    
    try:
        df = await storage.fetch_data_as_dataframe('SELECT * FROM climate_data_records')
        print('数据库中的记录:')
        print(df.to_string())
        print('\n记录详情:')
        for index, row in df.iterrows():
            print(f"记录 {index + 1}:")
            print(f"  来源: {row['source']}")
            print(f"  数据类型: {row['data_type']}")
            print(f"  文件路径: {row['file_path']}")
            print(f"  文件格式: {row['file_format']}")
            print(f"  文件大小: {row['file_size']}")
            print(f"  变量: {row['variables']}")
            print(f"  创建时间: {row['created_at']}")
            print("---")
    finally:
        await storage.close()

if __name__ == "__main__":
    asyncio.run(check_database())