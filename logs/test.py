import re
import pandas as pd

# 读取日志文件
with open('ehr/logs/search.log', 'r') as file:
    lines = file.readlines()

# 初始化数据列表
data = []

# 定义处理每行的函数
for line in lines:
    # 使用正则表达式提取所有键值对、score和time
    kv_pattern = r'(\w+)=([^,;]+)'
    score_pattern = r'score=([\d.]+)'
    time_pattern = r'total time=([\d.]+[a-z]+)'

    # 查找所有匹配项
    kv_matches = re.findall(kv_pattern, line)
    score_match = re.search(score_pattern, line)
    time_match = re.search(time_pattern, line)

    # 构建当前行的字典
    row_dict = {}
    for key, value in kv_matches:
        # 尝试将数值字符串转换为浮点数或整数
        try:
            if '.' in value:
                row_dict[key] = float(value)
            else:
                row_dict[key] = int(value)
        except:
            row_dict[key] = value.strip()

    # 添加score和time到字典
    if score_match:
        row_dict['score'] = float(score_match.group(1))
    if time_match:
        row_dict['total_time'] = time_match.group(1).strip()

    # 将当前行的字典添加到数据列表
    data.append(row_dict)

# 将数据列表转换为DataFrame
df = pd.DataFrame(data)

# 将DataFrame保存为Excel文件
df.to_excel('output.xlsx', index=False)

print("Excel 文件已保存！")