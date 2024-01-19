'Thunks for lazy evaluation with memoization'

from typing import TypeVar, Generic, Callable
from functools import total_ordering, wraps, cached_property

_T = TypeVar('_T')

@total_ordering
class Thunk(Generic[_T]):
    susp: Callable[..., _T]
    args: tuple
    kwargs: dict

    @cached_property
    def memo(self) -> _T:
        return self.susp(*map(force, self.args), **{k: force(v) for k, v in self.kwargs.items()})

    def __init__(self, func: Callable[..., _T], *args, **kwargs):
        # the if-else distinction decreases overhead from lambda and forcing in case of 0-parameter functions
        self.susp = func
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self) -> _T: return self.memo

    # repr still shows the thunk, but for readability it is better to evaluate the thunk
    def __str__(self): return str(force(self))

    def __eq__(self, other): return force(self) == force(other)
    
    def __lt__(self, other): return force(self) < force(other)
    
    def __hash__(self): return hash(force(self))
    
    def __bool__(self): return bool(force(self))

    def __add__(self, other): return Thunk(lambda x, y: x + y, self, other)
    
    def __sub__(self, other): return Thunk(lambda x, y: x - y, self, other)
    
    def __mul__(self, other): return Thunk(lambda x, y: x * y, self, other)
    
    def __matmul__(self, other): return Thunk(lambda x, y: x @ y, self, other)
    
    def __truediv__(self, other): return Thunk(lambda x, y: x / y, self, other)
    
    def __floordiv__(self, other): return Thunk(lambda x, y: x // y, self, other)
    
    def __mod__(self, other): return Thunk(lambda x, y: x % y, self, other)
    
    def __divmod__(self, other): return Thunk(divmod, self, other)
    
    def __pow__(self, other, modulo=None): return Thunk(pow, self, other, modulo)
    
    def __lshift__(self, other): return Thunk(lambda x, y: x << y, self, other)
    
    def __rshift__(self, other): return Thunk(lambda x, y: x >> y, self, other)
    
    def __and__(self, other): return Thunk(lambda x, y: x & y, self, other)
    
    def __xor__(self, other): return Thunk(lambda x, y: x ^ y, self, other)
    
    def __or__(self, other): return Thunk(lambda x, y: x | y, self, other)
    
    def __neg__(self): return Thunk(lambda x: -x, self)
    
    def __pos__(self): return Thunk(lambda x: +x, self)
    
    def __abs__(self): return Thunk(abs, self)
    
    def __invert__(self): return Thunk(lambda x: ~x, self)


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
    'Decorator for functions. Changes the output to Thunk, so that the actual result of the function may be evaluated easily.'
    @wraps(f)
    def wrapper(*args, **kwargs):
        return Thunk(lambda: force(f(*args, **kwargs)))
    
    return wrapper


# for testing
@lazy
def _fib(n):
    if n <= 1: return n
    return _fib(n-1) + _fib(n-2)