import re


class RETool:
    """ 列名的处理涉及很多正则匹配。统一放在 Tool 中便于维护
    """
    @staticmethod
    def get_last_op_params(data_full_name):
        """
        得到最后一个操作的 param list

        :return:
        """
        return re.match(r'[\dA-Za-z]+(?:_[\dA-Za-z]+)*?((?:_\d+)*)', data_full_name.split('.')[-1]).groups()[0]
