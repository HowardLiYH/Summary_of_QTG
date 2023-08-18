import configparser
import os

"""
os.path.realpath(): 获取当前文件的全路径
os.path.split()：按照路径将文件名和路径分割开
os.path.join(): 将多个路径组合后返回
"""


# basePath = os.path.split(os.path.realpath(__file__))[0]
# configPath = os.path.join(basePath + '\config.ini')


class ReadConfig():
    """
    初始化ConfigParser实例，使用ConfigParser模块读取配置文件的section节点，section节点就是config.ini中[]的内容
    """

    def __init__(self, path_ini):
        basePath = os.path.split(os.path.realpath(__file__))[0]
        configPath = os.path.join(basePath + "/" + path_ini)
        self.conf = configparser.ConfigParser()
        self.conf.read(configPath, encoding='utf-8')

    def get_str(self, section, option):
        return self.conf.get(section, option)

    def get_boolean(self, section, option):
        return self.conf.getboolean(section, option)

    def get_int(self, section, option):
        return self.conf.getint(section, option)

    def get_float(self, section, option):
        return self.conf.getfloat(section, option)
