import os
import yaml

def load_config():
    # 获取当前脚本的绝对路径
    current_script_path = os.path.abspath(__file__)
    print(current_script_path)
    # 获取项目根目录（假设 D.yaml 在项目根目录下）
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    print(project_root)
    # 构建配置文件路径
    config_path = os.path.join(project_root, "config.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config