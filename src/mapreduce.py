"""
Code from book "Annotated Algorithms in Python3"
Written by Massimo Di Pierro - BSD License
"""


def mapreduce(mapper, reducer, data):
    """
    >>> def mapfn(x): return x%2, 1
    >>> def reducefn(key, values): return len(values)
    >>> data = list(range(100))
    >>> mapreduce(mapfn, reducefn, data)
    {0: 50, 1: 50}
    """
    partials = {}
    results = {}
    for item in data:
        key, value = mapper(item)
        if not key in partials:
            partials[key] = [value]
        else:
            partials[key].append(value)
    for key, values in partials.items():
        results[key] = reducer(key, values)
    return results
