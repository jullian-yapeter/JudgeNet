import time


class AttrDict(dict):
    '''AttrDict wraps around a python dictionary and makes keys accessible
    as attributes: dictionary.key instead of dictionary["key"].
    '''
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        return self

    def __setstate__(self, dictionary):
        self = dictionary


class Timer():
    def __init__(self, section_name):
        self.section_name = section_name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        print(f"starting {self.section_name}")

    def __exit__(self, *args, **kwargs):
        print(f"finished {self.section_name}: "
              f"{time.time() - self.start}s")
