import logging
from logging.handlers import RotatingFileHandler
import os
import sys

# 配置日志
def setup_logging(name="main"):
    # 创建日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG

    # 创建日志格式器
    formatter = logging.Formatter(
        fmt=
        "[ %(asctime)s - %(levelname)s - %(filename)s:%(lineno)d ] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # 检查是否已经添加了处理器
    if not logger.handlers:
        # 创建控制台日志处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # 控制台输出日志级别为INFO
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 创建文件日志处理器（支持日志文件按大小滚动）
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, f"{name}.log"),
            maxBytes=1024 * 1024 * 5,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # 文件输出日志级别为DEBUG
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("\n\n\n-----")#空几行用于区分不同的运行
        log_file = os.path.join(log_dir, f"{name}.log")
        logger.info(f"日志文件保存在：{os.path.abspath(log_file)}")
        logger.info("Logging setup complete.")

    # 捕获异常信息
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    
    return logger


if __name__ == "__main__":
    logger = setup_logging("test")
    logger1 = setup_logging("test")
    logger.info("测试Logging setup complete.")
    # 这里可以添加其他代码，例如调用train_model等函数进行模型训练和评估
