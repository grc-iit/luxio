from luxio.common.error_codes import *
from typing import Tuple, List

class Daemon:
    def __init__(self, fun):
        self.fun = fun
        while True:
            self.fun()
