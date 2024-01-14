'Thunks for lazy evaluation with memoization'

from singleton import Singleton
from typing import TypeVar, Generic, Callable
from functools import total_ordering

# Not-Evaluated singleton class
class NotEvaluated(metaclass=Singleton): pass
_NE = NotEvaluated()

_T = TypeVar('_T')

@total_ordering
class Thunk(Generic[_T]):
    susp: Callable[[], _T]
    memo: NotEvaluated | _T

    def __init__(self, f: Callable[..., _T], *args, **kwargs):
        # the if-else distinction decreases overhead from lambda in case of 0-parameter functions
        if args or kwargs:
            self.susp = lambda: f(*args, **kwargs)
        else:
            self.susp = f
        self.memo = _NE
    
    def __call__(self):
        if self.memo is _NE:
            self.memo = self.susp()
        return self.memo
    
    # repr still shows the thunk, but for readability it is better to evaluate the thunk
    def __str__(self):
        return str(self())
    
    def __eq__(self, other):
        return self() == other()
    
    def __lt__(self, other):
        return self() < other()
    

# for testing purposes only
def fib(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n < 0:
        return -fib(n+1) + fib(n+2)
    else:
        return fib(n-1) + fib(n-2)