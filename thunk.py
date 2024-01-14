'Thunks for lazy evaluation with memoization'

from singleton import Singleton
from typing import TypeVar, Generic, Callable

# Not-Evaluated singleton class
class NotEvaluated(metaclass=Singleton): pass
_NE = NotEvaluated()

_T = TypeVar('_T')

class Thunk(Generic[_T]):
    susp: Callable[[], _T]
    memo: NotEvaluated | _T

    def __init__(self, f: Callable[[], _T]):
        self.susp = f
        self.memo = _NE
    
    def __call__(self):
        if self.memo is _NE:
            self.memo = self.susp()
        return self.memo