def exists(val):
    return val is not None

def default(val, default_val):
    if val is None:
        return default_val
    return val