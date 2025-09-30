from timeit import default_timer as timer
from datetime import timedelta


def time_function_call(f, *args, **kwargs):
    start = timer()
    result = f(*args, **kwargs)
    end = timer()
    return result, timedelta(seconds=end-start)
