from common.error_codes import *

class MapperManager:
    def __init__(self):
        pass

    def _run_expr(self, key):
        element = self.output[key]
        try:
            exec(element["include"])
        except:
            raise Error(ErrorCode.MAPPER_EXEC_ERROR).format("include", key)

        try:
            if not eval(element["guard"]):
                self.output[key]["val"] = element['val']
                self.output[key]["executed"] = True
                return
        except:
            raise Error(ErrorCode.MAPPER_EXEC_ERROR).format("guard", key)

        try:
            exec(element["expr"])
        except:
            raise Error(ErrorCode.MAPPER_EXEC_ERROR).format("expr", key)
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
                self._resolve_dependency(variable)
            self._run_expr(key)