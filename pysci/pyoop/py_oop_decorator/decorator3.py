def rangetest(*argchecks):  # Validate positional arg ranges
    def onDecorator(func):
        if not __debug__:  # Ture, if "python -O main.py args..."
            return func
        else:
            def onCall(*args):
                for (ix, low, high) in argchecks:
                    if args[ix] < low or args[ix] > high:
                        errmsg = "Argument %s not in %s..%s" % (ix, low, high)
                        raise TypeError(errmsg)
                return func(*args)
            return onCall
    return onDecorator
