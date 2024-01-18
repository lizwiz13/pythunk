'Thunks for lazy evaluation with memoization'

from .singleton import Singleton
from typing import TypeVar, Generic, Callable, TypeGuard
from functools import total_ordering, wraps

# Not-Evaluated singleton class
class _NotEvaluated(metaclass=Singleton): pass
_NE = _NotEvaluated()

_T = TypeVar('_T')

def _is_evaluated(x: _NotEvaluated | _T) -> TypeGuard[_T]:
    'This function is useful to assert the type of Thunk.memo'
    return x is not _NE


@total_ordering
class Thunk(Generic[_T]):
    susp: Callable[[], _T]
    memo: _NotEvaluated | _T

    def __init__(self, f: Callable[..., _T], *args, **kwargs):
        # the if-else distinction decreases overhead from lambda and forcing in case of 0-parameter functions
        if args or kwargs:
            self.susp = lambda: f(*map(force, args), **{k: force(v) for k, v in kwargs.items()})
        else:
            self.susp = f
        self.memo = _NE
    
    def __call__(self) -> _T:
        if not _is_evaluated(self.memo):
            self.memo = self.susp()
        return self.memo

    # repr still shows the thunk, but for readability it is better to evaluate the thunk
    def __str__(self):
        return str(force(self))

    def __eq__(self, other):
        return force(self) == force(other)
    
    def __lt__(self, other):
        return force(self) < force(other)
    
    def __hash__(self):
        return hash(force(self))
    
    def __bool__(self):
        return bool(force(self))

    def __add__(self, other):
        return Thunk(lambda x, y: x + y, self, other)
    
    def __sub__(self, other):
        return Thunk(lambda x, y: x - y, self, other)
    
    def __mul__(self, other):
        return Thunk(lambda x, y: x * y, self, other)
    
    def __matmul__(self, other):
        return Thunk(lambda x, y: x @ y, self, other)
    
    def __truediv__(self, other):
        return Thunk(lambda x, y: x / y, self, other)
    
    def __floordiv__(self, other):
        return Thunk(lambda x, y: x // y, self, other)
    
    def __mod__(self, other):
        return Thunk(lambda x, y: x % y, self, other)
    
    def __divmod__(self, other):
        return Thunk(divmod, self, other)
    
    def __pow__(self, other, modulo=None):
        return Thunk(pow, self, other, modulo)
    
    def __lshift__(self, other):
        return Thunk(lambda x, y: x << y, self, other)
    
    def __rshift__(self, other):
        return Thunk(lambda x, y: x >> y, self, other)
    
    def __and__(self, other):
        return Thunk(lambda x, y: x & y, self, other)
    
    def __xor__(self, other):
        return Thunk(lambda x, y: x ^ y, self, other)
    
    def __or__(self, other):
        return Thunk(lambda x, y: x | y, self, other)
    
    def __neg__(self):
        return Thunk(lambda x: -x, self)
    
    def __pos__(self):
        return Thunk(lambda x: +x, self)
    
    def __abs__(self):
        return Thunk(abs, self)
    
    def __invert__(self):
        return Thunk(lambda x: ~x, self)


_LazyT = _T | Thunk['_LazyT[_T]']


# using force instead of __call__ is safer and allows binary operations between Thunks and other types
def force(x: _LazyT[_T]) -> _T:
    '''Forces the value inside of a given thunk. 
    
    If a regular value is given, it is immediately returned. Flattens nested thunks.'''
    if isinstance(x, Thunk):
        return force(x())
    return x


def const(x: _T) -> Thunk[_T]:
    'Shortcut for creating a thunk with an already evaluated object.'
    return Thunk(lambda: x)


def lazy(f: Callable[..., _T]) -> Callable[..., Thunk[_T]]:

    @wraps(f)
    def wrapper(*args, **kwargs):
        return Thunk(f, *args, **kwargs)
    
    return wrapper


# for testing purposes only
@lazy
def fib(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n < 0:
        return -fib(n+1) + fib(n+2)
    return fib(n-1) + fib(n-2)