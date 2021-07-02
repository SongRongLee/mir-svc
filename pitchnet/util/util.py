import functools


def rgetattr(obj, attr, *args):
    """
    Recursive version of getattr().
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
