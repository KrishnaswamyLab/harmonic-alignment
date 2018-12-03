import joblib


class ParallelQueue(object):

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs
        self.parallel = joblib.Parallel(n_jobs=self.n_jobs)
        self.reset()

    def __enter__(self):
        self.parallel.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self.parallel.__exit__(exc_type, exc_value, traceback)

    def reset(self):
        self.functions = []
        self.args = []
        self.kwargs = []

    def queue(self, function, *args, **kwargs):
        self.functions.append(function)
        self.args.append(args)
        self.kwargs.append(kwargs)

    def run(self):
        result = joblib.Parallel(self.n_jobs)(
            joblib.delayed(self.functions[i])(
                *(self.args[i]), **(self.kwargs[i]))
            for i in range(len(self.functions)))
        self.reset()
        return result
