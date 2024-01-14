'implementation of single-instance classes'

class Singleton(type):
    instances = {}

    def __call__(cls, *args, **kwargs):
        try:
            return cls.instances[cls]
        except KeyError as ke:
            if cls not in cls.instances:
                instance = super(Singleton, cls).__call__(*args, **kwargs)
                cls.instances[cls] = instance
                return instance
            raise ke