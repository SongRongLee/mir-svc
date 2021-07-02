import bottleneck as bn


def moving_avg(x, n):
    """
    Smooth the sequence x using moving average algorithm.
        - x: Source sequence
        - n: Window size
    """
    return bn.move_mean(x.T, n).T
