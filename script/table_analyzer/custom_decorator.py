from functools import wraps
import time


def stop_watch(data_dict):
    def _stop_watch(func):
        @wraps(func)
        def wrapper(*args, **kargs) :
            start = time.time()
            result = func(*args,**kargs)
            process_time =  time.time() - start
            data_dict["process_time"] = process_time
            return result
        return wrapper
    return _stop_watch

def get_process_time(data_dict):
    return data_dict["process_time"]
