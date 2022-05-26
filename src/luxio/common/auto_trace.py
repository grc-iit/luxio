from .time import Timer

class AutoTrace:
    anatomy = {}
    verbose = False

    def __init__(self, fun_id, **params):
        if AutoTrace.verbose:
            print(f"ENTERING {fun_id} with params: {params}")
        self.fun_id = fun_id
        if fun_id not in AutoTrace.anatomy:
            AutoTrace.anatomy[fun_id] = 0
        self.timer = Timer()
        self.timer.resume()

    def __del__(self):
        self.entry[self.fun_id] += self.timer.pause().msec()
        if AutoTrace.verbose:
            print(f"LEAVING {fun_id} (time: {self.timer.msec()})")

    @staticmethod
    def verbose():
        AutoTrace.verbose = True

    @staticmethod
    def save(path, agg=False):
        if path is not None:
            pd.Series(AutoTrace.anatomy).to_csv(path)
