import logging

from preprocessing.temporal.temporal_data_process import temporal_data_process


# 日志信息常量类
class LogInfo:
    # 日志级别
    level = logging.DEBUG
    # 日志输出格式
    format = "%(asctime)s - %(levelname)s - %(message)s"


# 配置日志基本信息
logging.basicConfig(
    filename="./log/app.log",
    level=LogInfo.level,
    format=LogInfo.format
)

# 配置日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(LogInfo.level)
console_handler.setFormatter(logging.Formatter(LogInfo.format))

# 初始化日志记录仪
logger = logging.getLogger()
logger.addHandler(console_handler)


if __name__ == '__main__':
    temporal_data_process()
