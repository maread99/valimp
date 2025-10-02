<!-- NB any links not defined as absolute will not resolve on PyPI page -->
# valimp

<!-- UPDATE BADGE ADDRESSES! -->
[![PyPI](https://img.shields.io/pypi/v/valimp)](https://pypi.org/project/valimp/) ![Python Support](https://img.shields.io/pypi/pyversions/valimp) [![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-D7FF64.svg)](https://github.com/astral-sh/ruff) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/maread99/valimp/main.svg)](https://results.pre-commit.ci/latest/github/maread99/valimp/main)

In Python use type hints to validate, parse and coerce inputs to **public functions and dataclasses**. 

This is the sole use of `valimp`. It's a single short module with no depenencies that does one thing and makes it simple to do.

Works like this:
```python
from valimp import parse, Parser, Coerce
from typing import Annotated, Union, Optional, Any

@parse  # add the `valimp.parse`` decorator to a public function or method
def public_function(
    # validate against built-in or custom types
    a: str,
    # support for type unions
    b: Union[int, float],  # or from Python 3.10 `int | float`
    # validate type of container items
    c: dict[str, Union[int, float]],  # dict[str, int | float]
    # coerce input to a specific type
    d: Annotated[
        Union[int, float, str],  # int | float | str
        Coerce(int)
    ],
    # parse input with reference to earlier inputs...
    e: Annotated[
        str,
        Parser(lambda name, obj, params: obj + f"_{name}_{params['a']}")
    ],
    # coerce and parse input...
    f: Annotated[
        Union[str, int],  # str | int
        Coerce(str),
        Parser(lambda name, obj, _: obj + f"_{name}")
    ],
    # support for packing extra arguments if required, can be optionally typed...
    *args: Annotated[
        Union[int, float, str],  # int | float | str
        Coerce(int)
    ],
    # support for optional types
    g: Optional[str],  # str | None
    # define default values dynamically with reference to earlier inputs
    h: Annotated[
        Optional[float],  # float | None
        Parser(lambda _, obj, params: params["b"] if obj is None else obj)
    ] = None,
    # support for packing excess kwargs if required, can be optionally typed...
    # **kwargs: Union[int, float]
) -> dict[str, Any]:
    return {"a":a, "b":b, "c":c, "d":d, "e":e, "f":f, "args",args, "g":g, "h":h}

public_function(
    # NB parameters 'a' through 'f' could be passed positionally
    "zero",  # a
    1.0,  # b
    {"two": 2},  # c
    3.3,  # d, will be coerced from float to int, i.e. to 3
    "four",  # e, will be parsed to "four_e_zero"
    5,  # f, will be coerced to str and then parsed to "5_f"
    "10",  # extra arg, will be coerced to int and packed
    20,  # extra arg, will be packed
    g="keyword_arg_g",
    # h, not passed, will be assigned dynamically as parameter b (i.e. 1.0)
)
```
returns:
```
{'a': 'zero',
 'b': 1.0,
 'c': {'two': 2},
 'd': 3,
 'e': 'four_e_zero',
 'f': '5_f',
 'args': (10, 20),
 'g': 'keyword_arg_g',
 'h': 1.0}
 ```
 And if there are invalid inputs...
```python
public_function(
    a=["not a string"],  # INVALID
    b="not an int or a float",  # INVALID
    c={2: "two"},  # INVALID, key not a str and value not an int or float
    d=3.2, # valid input
    e="valid input",
    f=5.0,  # INVALID, not a str or an int
    g="valid input",
)
```
raises:
```
InputsError: The following inputs to 'public_function' do not conform with the corresponding type annotation:

a
	Takes type <class 'str'> although received '['not a string']' of type <class 'list'>.

b
	Takes input that conforms with <(<class 'int'>, <class 'float'>)> although received 'not an int or a float' of type <class 'str'>.

c
	Takes type <class 'dict'> with keys that conform to the first argument and values that conform to the second argument of <dict[str, typing.Union[int, float]]>, although the received dictionary contains an item with key '2' of type <class 'int'> and value 'two' of type <class 'str'>.

f
	Takes input that conforms with <(<class 'str'>, <class 'int'>)> although received '5.0' of type <class 'float'>.
```
And if the inputs do not match the signature...
```python
public_function(
    "zero",
    "invalid input",  # invalid (not int or float), included in errors
    {"two": 2},
    3.2,
    # no argument passed for required positional arg 'e'
    # no argument passed for required positional arg 'f'
    a="a again",  # passing multiple values for parameter 'a'
    # no argument passed for required keyword arg 'g'
    not_a_kwarg="not a kwarg",  # including an unexpected kwarg
)
```
raises:
```
InputsError: Inputs to 'public_function' do not conform with the function signature:

Got multiple values for argument: 'a'.

Got unexpected keyword argument: 'not_a_kwarg'.

Missing 2 positional arguments: 'e' and 'f'.

Missing 1 keyword-only argument: 'g'.

The following inputs to 'public_function' do not conform with the corresponding type annotation:

b
	Takes input that conforms with <(<class 'int'>, <class 'float'>)> although received 'invalid input' of type <class 'str'>.
```
Use all the same functionality to validate, parse and coerce the fields of a dataclass...
```python
from valimp import parse_cls
import dataclasses

@parse_cls  # place valimp decorator above the dataclass decorator
@dataclasses.dataclass
class ADataclass:
    
    a: str
    b: Annotated[
        Union[str, int],
        Coerce(str),
        Parser(lambda name, obj, params: obj + f" {name} {params['a']}")
    ]

rtrn = ADataclass("I'm a and will appear at the end of b", 33)
dataclasses.asdict(rtrn)
```
output:
```
{'a': "I'm a and will appear at the end of b",
 'b': "33 b I'm a and will appear at the end of b"}
```
## Installation

`$ pip install valimp`

No dependencies!

## Documentation
[tutorial.ipynb](https://github.com/maread99/valimp/blob/master/docs/tutorials/tutorial.ipynb) offers a walk-through of all the functionality.

Further documentation can be found in the module docstring of [valimp.py](https://github.com/maread99/valimp/blob/master/src/valimp/valimp.py).

## Why another validation library!?

### Why even validate input type?
Some may argue that validating the type of public inputs is not pythonic and we can 'duck' out of it and let the errors arise where they may. I'd argue that for the sake of adding a decorator I'd rather raise an intelligible error message than have to respond to an issue asking 'why am I getting this error...'.

> :information_source: `valimp` is only intended for handling inputs to **public functions and dataclasses**. For internal validation, consider using a type checker (for example, [mypy](https://github.com/python/mypy)). 

Also, I like the option of abstracting away all parsing, coercion and validation of public inputs and just receiving the formal parameter as required. For example, public methods in [market-prices](https://github.com/maread99/market_prices) often include a 'date' parameter. I like to offer users the convenience to pass this as either a `str`, a `datetime.date` or a `pandas.Timestamp`, although internally I want it as a `pandas.Timestamp`. I can do this with Valimp by simply including `Coerce(pandas.Timestamp)` to the metadata of the type annotation of each 'date' parameter. I also need to validate that the input is timezone-naive and does indeed represent a date rather than a time. I can do this by defining a single `valimp.Parser` and similarly including it to the annotation metadata of the 'date' parameters. Everything's abstracted away. With a little understanding of type annotations the user can see what's going on by simple inspection of the function's signature (as included within the standard help).

### Why wouldn't I just use Pydantic?
[Pydantic](https://github.com/pydantic/pydantic) is orientated towards the validation of inputs to dataclasses. Whilst the Valimp `@parse_cls` decorator does this well for non-complex cases, if you're looking to do more then Pydantic is the place to go.

As for validating public function input, in the early releases of Pydantic V2 the `@validate_call` decorator failed to provide for validating later parameters based on values received by earlier parameters (a [regression](https://github.com/pydantic/pydantic/issues/6794) from the Pydantic V1 `@validate_arguments` decorator). This loss of functionality, together with finding Pydantic somewhat clunky to do anything beyond simple type validation, is what led me to write `valimp`. (I believe functionality to validate later parameters based on values receive by earlier parameters may have since been restored in Pydantic V2, see the [issue](https://github.com/pydantic/pydantic/issues/6794).)

In short, if you only want to validate the type of function inputs then Pydantic V2 `@validate_call` will do the trick. If you're after additional validation, parsing or coercion then chances are you'll find `valimp` to be a simpler option.

## Limitations and Development

`valimp` does NOT currently support:
  - Positional-only arguments. Any '/' in the signature (to define
  positional-only arguments) will be ignored. Consequently valimp DOES
  allow intended positional-only arguments to be passed as keyword
  arguments.
  - Validation of subscripted types in `collections.abc.Callable` (although Valimp will verify that the passed value is callable).

`valimp` currently supports:
* use of the following type annotations:
    * built-in classes, for example `int`, `str`, `list`, `dict` etc
    * custom classes
    * `collections.abc.Sequence`
    * `collections.abc.Mapping`
    * typing.Any
    * typing.Literal
    * typing.Union ( `|` from 3.10 )
    * typing.Optional ( `<cls> | None` from 3.10)
    * collections.abc.Callable, although validation of subscripted types is **not** supported
* validation of container items for the following generic classes:
    * `list`
    * `dict`
    * `tuple`
    * `set`
    * `collections.abc.Sequence`
    * `collections.abc.Mapping`
* packing and optionally coercing, parsing and validating packed objects, i.e. objects received to, for example, *args and **kwargs.

The library has been built with development in mind and PRs are very much welcome!

## License

[MIT License][license]


[license]: https://github.com/maread99/valimp/blob/master/LICENSE.txt
