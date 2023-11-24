"""Validation and parsing of inputs to public functions.

The `parse` decorator provides for:
  - validating public function/method inputs against function signature
    and type annotations, including optional type validation of items in
    containers.
  - coercing inputs to a specific type.
  - user-defined parsing and validation.

`parse` does NOT currently support:
  - variable length arguments (i.e. *args in the function signature).
  - variable length keyword arguments (i.e. **kwargs in the function
    signature).
  - precluding positional-only arguments being passed as keyword arguments.

See the tutorial for a walk-through of all functionality:
https://github.com/maread99/valimp/blob/master/docs/tutorials/tutorial.ipynb

A version of the following example is included to the README with
explanatory comments and inputs:
    from valimp import parse, Parser, Coerce
    from typing import Annotated, Union, Optional, Any

    @parse
    def public_function(
        a: str,
        b: Union[int, float],  # or from python 3.10, `int | float`
        c: dict[str, Union[int, float]],  # dict[str, int | float]
        d: Annotated[
            Union[int, float, str],  # int | float | str
            Coerce(int)
        ],
        e: Annotated[
            str,
            Parser(lambda name, obj, params: obj + f"_{name}_{params['a']}")
        ],
        f: Annotated[
            Union[str, int],  # str | int
            Coerce(str),
            Parser(lambda name, obj, _: obj + f"_{name}")
        ],
        *,
        g: Optional[str],  # str | None
        h: Annotated[
            Optional[float],  # float | None
            Parser(lambda _, obj, params: params["b"] if obj is None else obj)
        ] = None,
    ) -> dict[str, Any]:
        return {"a":a, "b":b, "c":c, "d":d, "e":e, "f":f, "g":g, "h":h}

Type validation
---------------

  - All inputs that are annotated with a type will be validated as
  conforming with the type annotation.

  - Container items are by default validated as conforming with subscripted
  types (for example, list[str]).

  - Inputs annoated with a typing.Literal type can be validated as either
  comparing equal (default) or being equal.

From Python 3.10 use of the | operator is supported to define annotations
describing type unions and optional types.

The following type annotations are supported:

    A single type, builtin or user defined. Examples:
        f(param: int)
        f(param: CustomType)

    typing.Union or (from python 3.10) the | operator. Examples:
        f(param: typing.Union[int, float])
        f(param: int | float)  # from python 3.10

    `list`. Example:
        f(param: list)
    If the type of the list items is included then all items will be
    validated against that type. For example, to validate that all list
    items are of type str:
        f(param: list[str])

    `tuple`. Example:
        f(param: tuple)
    If the type(s) of the tuple items is included then all tuple items will
    be validated against the type(s).
    Examples:
        To validate that all tuple items are of type int:
            f(param: tuple[int, ...])
        To validate the types of the items of a 4-tuple:
            f(
                param: tuple[int, str, list[str], dict[str, int]]
            )

    `dict`. Example:
        (param: dict)
    If the type of the dict keys and values is included then all dict keys
    and values will be validated against those types. For example:
        f(param: dict[str, int])
        f(param: dict[str, Union[int, float])

    `set`. Example:
        f(param: set)
    If the type of the set items is included then all set items will be
    validated against that type. For example, to validate that all set
    items are of type int:
        f(param: set[int])

   typing.Optional or (from python 3.10) <type> | None. Examples:
        f(param: typing.Optional[str])
        f(param: str | None)  # from python 3.10

    typing.Literal. Examples:
        f(param: typing.Literal["foo", "spam"])
        f(param: typing.Literal[1, 2, 3])
    By default input will be validated as comparing equal with one of the
    arguments to Literal. This can be changed to validate based on strictly
    being the same object by using `typing.Annotated` to define the
    annotation and including the `valimp.STRICT_LITERAL` constant to
    the metadata. For example:
        f(
            param: typing.Annotated[
                typing.Literal[FOO, SPAM], STRICT_LITERAL,
            ]
        )

    collections.abc.Sequence.
        Validation as for `list`.

    collections.abc.Mapping.
        Validation as for `dict`.

    collections.abc.Callable. Example:
        f(param: collections.abc.Callable)
    Subscriptions of Callable are ignored, i.e. the following will be
    validated in the same way as the above unsubscripted example:
        f(param: collections.abc.Callable[[str, int], int])

NO_ITEM_VALIDATION
Validation of the type of items in a container can be skipped for any
parameter by using `typing.Annotated` to define the annotation and
including the `valimp.NO_ITEM_VALIDATION` constant to the metadata. For
example, the following will validate that 'param' receives a dictionary,
but not that the keys are str or that the values are either int or float:
    f(
        param: typing.Annotated[
            dict[str, typing.Union[int | float]],
            NO_ITEM_VALIDATION,
        ]
    )

Nested containers
Items in nested containers will by default by validated against the
subscripted annoations. For example the following will validate that the
outer tuple contains any number of lists, that those lists contain only
sets and that the sets contains only int or float.
    f(param: tuple[list[set[Union[int, float]]], ...])

Including NO_ITEM_VALIDATION to the annotation's metadata will result in
the contained items not being validated at any level of nesting.

Coercion and Parsing
--------------------
An input can be coerced to a specific type by annotating the parameter with
typing.Annotated and including an instance of the `valimp.Coerce` class to
the metadata. For example to coerce an int or str input to a float:
    f(
        param: typing.Annotated[
            typing.Union[float, int, str], Coerce(float)
        ]
    )
    f(
        param: typing.Annotated[float | int | str, Coerce(float)]
    )  # from python 3.10

NB An input receieved as None is not coerced.

NB Type checkers will continue to assume that a coerced object could be of
any type indicated in the type annotation. To 'right' the type checker it's
necessary to narrow the type by including type guard expression at the
start of the function. For example in mypy:
    @parse
    def f(param: typing.Annotated[Union[float, int, str], Coerce(float)):
        if TYPE_CHECKING:
            assert isinstance(param, float)

An input can be parsed and/or validated by a user-defined function by
including an instance of the `valimp.Parser` class to the typing.Annotation
metadata. For exmaple, to add a suffix to a string before the function
receives the input:
    f(
        param: typing.Annotated[
            str,
            Parser(lambda name, obj, params: obj + "suffix"),
        ]
    )

Parser takes its first argument as a callable which should parse and/or
validate the input. The callable should have the signature:
    (name: str, obj: Any, params: dict[str, Any]) -> Any
        where the parameters will be receieved as:
            name:
                Name of the function argument being parsed.
            obj:
                The input object being parsed. This `obj` will be received
                as coerced by any `Coerce` instance if the Coerce instance
                is passed to the Annotated metadata ahead of the Parser,
                otherwrise Parser will receive the input as passed by the
                client.
            params:
                Shallow copy of prior inputs that have already been parsed
                and, if applicable, coerced.
            -> :
                Parsed object.

User-defined validation can be included within the callable by simply
raising an appropriate exception in the event the validation fails. This
error will be raised directly.

By default inputs received as None are passed to the Parser function. This
provides for dynamically setting default values. The 'parse_none' argument
can be passed to Parser as False to not parse a None value. For example:
    f(
        param: typing.Annotated[
            Optional[str],
            Parser(
                lambda name, obj, params: obj + "suffix",
                parse_none=False,
            ),
        ] = None
    )

Coerce and Parser can be used in the same annoation. In this case the
parser function will receieve the coerced input if Coerce is included in
the Annotated metadata before the Parser instance, for example:
    f(
        param: typing.Annotated[
            typing.Union[str, int, float],
            Coerce(str),
            Parser(lambda name, obj, params: obj + "suffix",
        ]
    )

If Coerce follows Parser then the parser function will receive the input as
passed by the client and the output from the parser function will be
subsequently coreced.

Notes
-----
`typing.Annotated` should only be used once in a type annotation and must
be used as the outermost wrapper. For exmaple, do NOT do either of:
    param: typing.Union[typing.Annotated[str, Parser(parser_func)], int]
    param: typing.Annotated[
        typing.Union[typing.Annotated[str, Parser(parser_func)], int]
    ]
rather do this:
    param: typing.Annotated[typing.Union[str, int], Parser(parser_func)]
If you require the parser to only apply to one of the acceptable types then
this should be handled within the parser function, for example:
    def parser_func(name: str, obj: Any, params: Dict[str, Any]):
        if isinstance(obj, str):
            # validation/parsing code here
        return obj

Prior to python 3.11 if Optional is ommited from a type annotation then it
will be added if the variable has None as a default value. valimp does NOT
follow this behaviour (at least not if the type annotation is wrapped in
Annotated). If an input can take None, then this should be explicitly
definined within the type annotation. So, do NOT do this:
    param: typing.Annotated[str, Parser(parser_func)] = None
Rather, do any of these:
    param: typing.Annotated[Optional[str], Parser(parser_func)] = None
    param: typing.Annotated[Union[str, None], Parser(parser_func)] = None
    param: typing.Annotated[
        str | None, Parser(parser_func)
    ] = None  # from python 3.10
"""

from __future__ import annotations

import collections
import functools
import inspect
import sys
import types
import typing
from typing import Any


def is_annotated(hint: typing._Final) -> bool:
    """Query if a type hint is wrapped in typing.Annotated."""
    return hasattr(hint, "__metadata__")


class Coerce:
    """Holds user-defined coercion class.

    Parameters
    ----------
    coerce_to : type[Any]
        Type to coerce an input to.
    """

    def __init__(self, coerce_to: type[Any]):
        self.coerce_to = coerce_to


class Parser:
    """Holds user-defined parsing and/or validation code.

    If simply coercing input to a different type, consider using
    `Coerce`.

    Parameters
    ----------
    func : Callable
        Function to parse input. Should have signature:
            (name: str, obj: Any, params: dict[str, Any]) -> Any
        where:
            name:
                Name of the argument being parsed.
            obj:
                The input object being parsed. This `obj` will be received
                as coerced by any `Coerce` metadata.
            params:
                Shallow copy of prior inputs that have already been parsed
                and, if applicable, coerced.
            -> :
                Parsed object.

        Function should raise an exception if validation fails.

    parse_none : bool
        True (default): pass input received as None to `func`. This
        behavior provides for dynamically setting default values.

        False: do not pass input received as None `func`, rather pass
        through None value.
    """

    def __init__(
        self,
        func: collections.abc.Callable[[str, Any, dict[str, Any]], Any],
        parse_none: bool = True,
    ):
        self.function = func
        self.parse_none = parse_none


VALIDATED = (True, None)
FAILED_SIMPLE = (False, None)
NO_ITEM_VALIDATION = "dlkj3ow61"
STRICT_LITERAL = "dlkj3ow62"


def validates_against_hint(
    obj: Any,
    hint: type[Any] | typing._Final,
    annotated: typing._AnnotatedAlias | None,
    rtrn_error: bool = True,
) -> tuple[bool, ValueError | TypeError | None]:
    """Query if object conforms with type hint.

    Parameters
    ----------
    obj
        Object to validate against `hint`.

    hint
        Type hint against which to validate `obj`.

    annotated
        typing.Annotated instance that wraps `hint`. None if `hint` not
        wrapped.

    rtrn_error
        Whether to include an error in the return if the validation fails.
        See Returns section.

        NB not returning an error will be more performant, especially for
        validating hints comprising of nested hints, for example
        Union[str, List[str], Dict[str, Union[int, float]].

    Returns
    -------
    tuple[bool, ValueError | TypeError | None]
        [0] bool indicating if `obj` conforms to type.

        [1] None if 'obj' conforms to hint or `rtrn_error` is False,
        otherwise error advising why validaition failed.
    """
    if hint is typing.Any:
        return VALIDATED

    origin = typing.get_origin(hint)

    # handle Union hint
    if origin is typing.Union or (
        hasattr(types, "UnionType") and isinstance(hint, types.UnionType)
    ):
        hint_args = typing.get_args(hint)
        for hint_ in hint_args:
            validated, _ = validates_against_hint(obj, hint_, annotated, False)
            if validated:
                return VALIDATED
        if not rtrn_error:
            return FAILED_SIMPLE
        return False, ValueError(
            f"Takes input that conforms with <{hint_args}> although received"
            f" '{obj}' of type {type(obj)}."
        )

    # handle annotation defining a single type
    # NOTE ASSUMES only supported hint that does not have an __origin__ is
    # a union defined with | operation from python 3.10
    if origin is None:
        if isinstance(obj, hint):
            return VALIDATED
        if not rtrn_error:
            return FAILED_SIMPLE
        return False, ValueError(
            f"Takes type {hint} although received '{obj}' of type {type(obj)}."
        )

    hint_args = typing.get_args(hint)

    # handle literals hints
    if origin is typing.Literal:
        strict = annotated is not None and STRICT_LITERAL in annotated.__metadata__
        if not strict:
            if obj in hint_args:
                return VALIDATED
        else:
            try:
                _ = next((lit for lit in hint_args if id(obj) == id(lit)))
            except StopIteration:
                pass
            else:
                return VALIDATED
        if not rtrn_error:
            return FAILED_SIMPLE
        return False, ValueError(
            f"Takes a {'literal' if strict else 'value'} from <{hint_args}>"
            f" although received '{obj}'."
        )

    # validate object is instance of the origin 'type'
    if not isinstance(obj, origin):
        if not rtrn_error:
            return FAILED_SIMPLE
        return False, ValueError(
            f"Takes type {origin} although received '{obj}' of type {type(obj)}."
        )

    if origin is collections.abc.Callable:
        # validation of any subscripted types is not currently supported
        return VALIDATED

    if origin is tuple and hint_args[-1] is not Ellipsis:
        if len(obj) != len(hint_args):
            if not rtrn_error:
                return FAILED_SIMPLE
            return False, ValueError(
                f"Takes type {origin} of length {len(hint_args)} although received"
                f" '{obj}' of length {len(obj)}."
            )

    # Validation of container ITEMS

    if not obj or (
        annotated is not None and NO_ITEM_VALIDATION in annotated.__metadata__
    ):
        return VALIDATED

    if origin in (list, set, collections.abc.Sequence) or (
        origin is tuple and hint_args[-1] is Ellipsis
    ):
        sub_hint = hint_args[0]
        for e in obj:
            validated, _ = validates_against_hint(e, sub_hint, None, False)
            if not validated:
                if not rtrn_error:
                    return FAILED_SIMPLE
                return False, TypeError(
                    f"Takes type {origin} containing items that conform with <{hint}>,"
                    f" although the received container contains item '{e}' of type"
                    f" {type(e)}."
                )
        return VALIDATED

    if origin is tuple:
        for i, (e, sub_hint) in enumerate(zip(obj, hint_args)):
            validated, _ = validates_against_hint(e, sub_hint, None, False)
            if not validated:
                if not rtrn_error:
                    return FAILED_SIMPLE
                return False, TypeError(
                    f"Takes type {origin} containing items that conform with <{hint}>,"
                    f" although the item in position {i} is '{e}' of type {type(e)}."
                )
        return VALIDATED

    if origin in (dict, collections.abc.Mapping):
        key_hint = hint_args[0]
        key_error = False
        for i_k, k in enumerate(obj.keys()):
            validated, _ = validates_against_hint(k, key_hint, None, False)
            if not validated:
                if not rtrn_error:
                    return FAILED_SIMPLE
                key_error = True
                break

        val_hint = hint_args[1]
        val_error = False
        for i_v, v in enumerate(obj.values()):
            validated, _ = validates_against_hint(v, val_hint, None, False)
            if not validated:
                if not rtrn_error:
                    return FAILED_SIMPLE
                val_error = True
                break

        inset = "dictionary" if origin is dict else "mapping"
        if key_error and val_error and i_k == i_v:
            return False, TypeError(
                f"Takes type {origin} with keys that conform to the first argument and"
                f" values that conform to the second argument of <{hint}>, although the"
                f" received {inset} contains an item with key '{k}' of type {type(k)}"
                f" and value '{v}' of type {type(v)}."
            )
        if key_error:
            return False, TypeError(
                f"Takes type {origin} with keys that conform to the first argument"
                f" of <{hint}>, although the received {inset} contains key '{k}'"
                f" of type {type(k)}."
            )
        if val_error:
            return False, TypeError(
                f"Takes type {origin} with values that conform to the second"
                f" argument of <{hint}>, although the received {inset} contains"
                f" value '{v}' of type {type(v)}."
            )

        return VALIDATED

    raise TypeError(
        f"The following type annotation is not currently supported by `valimp`:"
        f"\n{hint}.\n\n The object receieved against this type annotation was'{obj}'"
        f" of type {type(obj)}."
    )


def validate_against_hints(
    kwargs: dict[str, Any],
    hints: dict[str, type[Any] | typing._Final],
) -> dict[str, ValueError | TypeError]:
    """Validate inputs against hints.

    Parameters
    ----------
    hints
        Dictionary of hints with key as parameter name and value as typing
        hint for parameter.

    kwargs
        All parameter inputs to be validated. Key as parameter name, value
        as object received by parameter.

    Returns
    -------
    errors
      Dictionary of any errors. Key as name of parameter for which
      validation failed, value as corresponding error.
    """
    errors = {}
    for name, obj in kwargs.items():
        if name not in hints:
            continue  # type not annotated for parameter
        hint = hints[name]
        if is_annotated(hint):
            annotated: typing._AnnotatedAlias | None = hint
            hint = hint.__origin__
        else:
            annotated = None

        validated, error = validates_against_hint(obj, hint, annotated)
        if not validated:
            assert error is not None
            errors[name] = error
    return errors


def args_name_inset(arg_names: list[str]) -> str:
    """Get string of argument names.

    Parameters
    ----------
    arg_names
        List of argument names to be included in inset.

    Examples
    --------
    >>> args_name_inset(["spam", "foo", "bar"])
    "'spam', 'foo' and 'bar'"
    >>> args_name_inset(["spam", "foo"])
    "'spam' and 'foo'"
    >>> args_name_inset(["spam"])
    "'spam'"
    """
    inset = f"'{arg_names[0]}'"
    for name in arg_names[1:]:
        if name == arg_names[-1]:
            inset += f" and '{name}'"
        else:
            inset += f", '{name}'"
    return inset


def get_missing_arg_error(missing: list[str], positional: bool = True) -> TypeError:
    """Get a TypeError for a missing positional or keyword-only argument.

    Parameters
    ----------
    missing
        List of names of missing arguments.

    positional
        True: Missing arguments are positional arguments.
        False: Missing arguments are keyword-only arguments.
    """
    inset = args_name_inset(missing)
    pos_inset = "positional" if positional else "keyword-only"
    return TypeError(
        f"Missing {len(missing)} {pos_inset}"
        f" argument{'s' if len(missing) > 1 else ''}: {inset}."
    )


def validate_against_signature(
    args_as_kwargs: dict[str, Any],
    kwargs: dict[str, Any],
    req_args: list[str],
    req_kwargs: list[str],
    all_arg_names: list[str],
) -> list[TypeError]:
    """Validate inputs against arguments expected by signature.

    Parameters
    ----------
    args_as_kwargs
        Inputs for arguments received positionaly. Key as argument
        name, value as received input (i.e. as if were received as a
        keyword argument).

        NB module does not support positional-only arguments (i.e. these
        could have been receieved as keyword args).

    kwargs
        Inputs for arguments receieved as keyword arguments. Key
        as argument name, value as received input.

    req_args
        List of names of required positional arguments.

    req_kwargs
        List of names of required keyword-only arguments.

    all_arg_names
        List of all possible argument names (positional and
        keyword only).

    Returns
    -------
    errors
        List of any TypeError instances relating to inputs that are
        invalid for the given signature information.
    """
    errors = []

    # duplicated arguments
    duplicated = [a for a in args_as_kwargs if a in kwargs]
    if duplicated:
        errors.append(
            TypeError(
                f"Got multiple values for argument{'s' if len(duplicated) > 1 else ''}:"
                f" {args_name_inset(duplicated)}."
            )
        )

    # excess arguments
    extra_args = [a for a in args_as_kwargs if a.startswith("_xtra")]
    if extra_args:
        obj_0 = args_as_kwargs[extra_args[0]]
        msg_end = f"\t'{obj_0}' of type {type(obj_0)}."
        for a in extra_args[1:]:
            obj = args_as_kwargs[a]
            msg_end += f"\n\t'{obj}' of type {type(obj)}."
        errors.append(
            TypeError(
                f"Received {len(extra_args)} excess positional"
                f" argument{'s' if len(extra_args) > 1 else ''} as:\n{msg_end}"
            )
        )

    extra_kwargs = [a for a in kwargs if a not in all_arg_names]
    if extra_kwargs:
        errors.append(
            TypeError(
                f"Got unexpected keyword"
                f" argument{'s' if len(extra_kwargs) > 1 else ''}"
                f": {args_name_inset(extra_kwargs)}."
            )
        )

    # missing required arguments
    all_as_kwargs = args_as_kwargs | kwargs
    missing = [a for a in req_args if a not in all_as_kwargs]
    if missing:
        errors.append(get_missing_arg_error(missing, True))

    missing_kw = [kwarg for kwarg in req_kwargs if kwarg not in all_as_kwargs]
    if missing_kw:
        errors.append(get_missing_arg_error(missing_kw, False))

    return errors


class InputsError(Exception):
    """Inputs do not conform with signature and/or type annotations.

    Consolidates TypeError and ValueErrors errors relating to inputs not
    conforming with signature and associated type annotations.
    """

    def __init__(
        self,
        func_name: str,
        sig_errors: list[TypeError],
        ann_errors: dict[str, TypeError | ValueError],
    ):
        msg = ""
        if sig_errors:
            msg += (
                f"Inputs to '{func_name}' do not conform with the"
                " function signature:"
            )
            for e_sig in sig_errors:
                msg += f"\n\n{e_sig.args[0]}"

        if ann_errors:
            if sig_errors:
                msg += "\n\n"
            msg += (
                f"The following inputs to '{func_name}' do not conform with the"
                " corresponding type annotation:"
            )
            for param_name, e in ann_errors.items():
                msg += f"\n\n{param_name}\n\t{e.args[0]}"
        self._msg = msg

    def __str__(self) -> str:
        return self._msg


# NOTE: can be removed from when min supported python version advances to 3.11
def fix_hints_for_none_default(
    hints: dict[str, type[Any] | typing._Final],
    spec: inspect.FullArgSpec,
) -> dict[str, type[Any] | typing._Final]:
    """Implement fix for differing behaviour between python versions.

    Prior to py3.11 type annotations of parameters that take None by default
    would be wrapped in typing.Optional if the annotation didn't already
    include Optional. This resulted in typing.Annotation being wrapped in
    typing.Optional, such that the annotation:
        a: typing.Annotation[Optional[str], "some meta"] = None
    has a hint as:
        'a': typing.Optional[typing.Annotated[typing.Optional[str], 'some_meta']]

    This fix removes the outer typing.Optional wrapper.

    Returns
    -------
    hints
        As received, updated by removing any outer typing.Optional wrapper
        around a typing.Annotation.
    """
    if sys.version_info.minor >= 11:
        return hints

    def update_hints(arg: str, dflt: Any):
        if arg not in hints:
            return
        if dflt is not None:
            return
        hint = hints[arg]
        if not typing.get_origin(hint) is typing.Union:
            return
        first_hint_arg = typing.get_args(hint)[0]
        if is_annotated(first_hint_arg):
            hints[arg] = first_hint_arg

    if spec.defaults is not None and None in spec.defaults:
        not_req_args = spec.args[-len(spec.defaults) :]
        for arg, dflt in zip(not_req_args, spec.defaults):
            update_hints(arg, dflt)

    if spec.kwonlydefaults is not None:
        for k, v in spec.kwonlydefaults.items():
            update_hints(k, v)

    return hints


def get_unreceived_args(
    spec: inspect.FullArgSpec, names_received: list[str]
) -> dict[str, Any]:
    """Get dictionary of unreceived args.

    Parameters
    ----------
    spec
        Function specification.

    names_received
        List of names of all received parameters..

    Returns
    -------
    unreceived_args
        Keys as names of args that were not receieved.
        Values as default values of those unrecieved args.
    """
    if spec.defaults is None:
        return {}
    unreceived = {}
    not_req_args = spec.args[-len(spec.defaults) :]
    for arg, dflt in zip(not_req_args, spec.defaults):
        if arg not in names_received:
            unreceived[arg] = dflt
    return unreceived


def get_unreceived_kwargs(
    spec: inspect.FullArgSpec, names_received: list[str]
) -> dict[str, Any]:
    """Get dictionary of unreceived kwargs.

    Parameters
    ----------
    spec
        Function specification.

    names_received
        List of names of all received parameters..

    Returns
    -------
    unreceived_kwargs
        Keys as names of kwargs that were not receieved.
        Values as default values of those unrecieved kwargs.
    """
    if spec.kwonlydefaults is None:
        return {}
    unreceived = {}
    for k, v in spec.kwonlydefaults.items():
        if k not in names_received:
            unreceived[k] = v
    return unreceived


def parse(f) -> collections.abc.Callable:
    """Decorator to validate and parse user inputs.

    See valimp module doc (valimp.__doc__).
    """
    spec = inspect.getfullargspec(f)
    hints = typing.get_type_hints(f, include_extras=True)
    hints = fix_hints_for_none_default(hints, spec)
    req_args = spec.args if spec.defaults is None else spec.args[: -len(spec.defaults)]
    if spec.kwonlydefaults is None:
        req_kwargs = spec.kwonlyargs
    else:
        req_kwargs = [a for a in spec.kwonlyargs if a not in spec.kwonlydefaults]
    all_param_names = spec.args + (
        spec.kwonlyargs if spec.kwonlyargs is not None else []
    )

    @functools.wraps(f)
    def wrapped_f(*args, **kwargs) -> Any:
        args_as_kwargs = {name: obj for obj, name in zip(args, spec.args)}
        if len(args) > len(spec.args):
            for i, obj in enumerate(args[len(spec.args) :]):
                args_as_kwargs["_xtra" + str(i)] = obj

        sig_errors = validate_against_signature(
            args_as_kwargs, kwargs, req_args, req_kwargs, all_param_names
        )

        params_as_kwargs = {  # remove arguments not in signature
            k: v for k, v in (args_as_kwargs | kwargs).items() if k in all_param_names
        }
        ann_errors = validate_against_hints(params_as_kwargs, hints)

        if sig_errors or ann_errors:
            raise InputsError(f.__name__, sig_errors, ann_errors)

        # coerce and validate
        # add in parameters that were not receieved and will take default value.
        param_names_received = list(params_as_kwargs)
        not_received_args = get_unreceived_args(spec, param_names_received)
        not_received_kwargs = get_unreceived_kwargs(spec, param_names_received)
        all_as_kwargs = (
            args_as_kwargs | not_received_args | kwargs | not_received_kwargs
        )

        new_as_kwargs = {}
        for name, obj in all_as_kwargs.items():
            if name not in hints:
                new_as_kwargs[name] = obj
                continue
            hint = hints[name]
            if is_annotated(hint):
                meta = hint.__metadata__
                for data in meta:
                    # let order of coercion and parsing depend on their
                    # order within metadata
                    if obj is not None and isinstance(data, Coerce):
                        obj = data.coerce_to(obj)
                    if isinstance(data, Parser):
                        if obj is None and not data.parse_none:
                            continue
                        obj = data.function(name, obj, new_as_kwargs.copy())

            new_as_kwargs[name] = obj

        return f(**new_as_kwargs)

    return wrapped_f
