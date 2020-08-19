from common.error_codes import *

class MapperManager:
    def __init__(self):
        pass

    def _run_expr(self, key):
        element = self.output[key]
        exec(element["include"])
        exec(element["expr"])
        self.output[key]["executed"]=True

    def _resolve_dependency(self, key):
        element = self.output[key]
        if "executed" not in element or not element["executed"]:
            for variable in element["dependencies"]:
                self._resolve_dependency(variable)
            self._run_expr(key)

    def run(self, input, output):
        self.input = input
        self.output = output
        for key in output:
            element = self.output[key]
            element["executed"]=False
            for variable in element["dependencies"]:
                try:
                    self._resolve_dependency(variable)
                except:
                    raise Error(ErrorCode.MAPPER_EXEC_ERROR).format(variable)

            try:
                self._run_expr(key)
            except:
                raise Error(ErrorCode.MAPPER_EXEC_ERROR).format(key)