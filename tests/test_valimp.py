"""Tests for market_prices.input module."""

from collections import abc
import inspect
import re
import sys
import typing
from typing import (
    Union,
    Literal,
    Annotated,
    Optional,
    get_type_hints,
    Any,
)

import pytest

import valimp as m

# pylint: disable=line-too-long
# pylint: disable=missing-function-docstring, missing-type-doc
# pylint: disable=missing-param-doc, missing-any-param-doc, redefined-outer-name
# pylint: disable=too-many-public-methods, too-many-arguments, too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=protected-access, no-self-use, unused-argument, invalid-name
# pylint: disable=unused-variable
#   missing-fuction-docstring: doc not required for all tests
#   protected-access: not required for tests
#   not compatible with use of fixtures to parameterize tests:
#       too-many-arguments, too-many-public-methods
#   not compatible with pytest fixtures:
#       redefined-outer-name, no-self-use, missing-any-param-doc, missing-type-doc
#   unused-argument: not compatible with pytest fixtures, caught by pylance anyway.
#   invalid-name: names in tests not expected to strictly conform with snake_case.

# Any flake8 disabled violations handled via per-file-ignores on .flake8


def test_decorator_wrapped():
    """Verify post-decoration function retains hints and docstring."""

    @m.parse
    def f(
        a: int, b: Annotated[float, "third-party-meta-b"], c: str = "some_default_str"
    ) -> tuple[int, float, str]:
        """Decorated func docstring."""
        return a, b, c

    assert f.__doc__ == "Decorated func docstring."
    expected_hints = {
        "a": int,
        "b": typing.Annotated[float, "third-party-meta-b"],
        "c": str,
        "return": tuple[int, float, str],
    }
    assert typing.get_type_hints(f, include_extras=True) == expected_hints


@pytest.fixture
def f() -> abc.Iterator[abc.Callable]:
    """Function with type annotations."""

    @m.parse
    def func(
        a: Annotated[int, m.Parser(lambda n, obj, _: obj + 4)],
        b: Annotated[Union[str, int, float], m.Coerce(int)],
        c: Annotated[
            Union[str, int], m.Coerce(str), m.Parser(lambda n, obj, p: obj + "_suffix")
        ],
        d: Annotated[
            Union[float, int], m.Parser(lambda n, obj, p: obj + 10), m.Coerce(str)
        ],
        e: str,
        f: Annotated[str, m.Parser(lambda name, obj, _: name + "_" + obj)],
        g: Annotated[
            str, m.Parser(lambda _, obj, params: obj + "_" + params["e"] + "_bar")
        ],
        h: Annotated[
            str,
            m.Parser(
                lambda name, obj, params: name + "_" + obj + "_" + params["e"] + "_bar"
            ),
        ],
        i: Annotated[int, "spam meta", "foo meta"],
        j: Literal["spam", "foo"],
        k: list[str],
        l: dict[str, int],
        m: abc.Mapping[str, int],
        n: tuple[str, ...],
        o: tuple[str, int, set],
        p: set[str],
        q: abc.Sequence[Union[str, int, set]],
        r: abc.Sequence[str],
        s: abc.Callable[[str], str],
        t: abc.Callable[[str, int], str],
        u: abc.Callable[..., str],
        v: abc.Callable[..., str],
        w: Union[int, str, None, Literal["spam", "foo"]],
        x: Annotated[Union[str, int, float], "spam meta", "foo meta"],
        y: Annotated[
            Optional[Union[str, int, float]], "foo meta", m.Parser(lambda n, o, p: o)
        ],
        z: Annotated[
            Optional[Union[str, int, float]], "spam meta", m.Parser(lambda n, o, p: o)
        ] = None,  # this annotation requires fixing by `fix_hints_for_none_default``
        aa: Annotated[Literal[3, 4, 5], "spam meta", "foo meta"] = 4,
        bb: Optional[int] = None,
        cc: bool = True,
        dd: Any = 4,
        ee: Optional[dict[str, Any]] = None,
        *,
        kwonly_req_a: bool,
        kwonly_req_b: typing.Annotated[Optional[bool], "meta"],
        kwonly_opt: typing.Annotated[Optional[bool], "meta"] = None,
        kwonly_opt_b: typing.Annotated[Any, "meta"] = "kwonly_opt_b",
    ) -> dict[str, Any]:
        return dict(
            a=a,
            b=b,
            c=c,
            d=d,
            e=e,
            f=f,
            g=g,
            h=h,
            i=i,
            j=j,
            k=k,
            l=l,
            m=m,
            n=n,
            o=o,
            p=p,
            q=q,
            r=r,
            s=s,
            t=t,
            u=u,
            v=v,
            w=w,
            x=x,
            y=y,
            z=z,
            aa=aa,
            bb=bb,
            cc=cc,
            dd=dd,
            ee=ee,
            kwonly_req_a=kwonly_req_a,
            kwonly_req_b=kwonly_req_b,
            kwonly_opt=kwonly_opt,
            kwonly_opt_b=kwonly_opt_b,
        )

    yield func


@pytest.fixture
def inst() -> abc.Iterator[object]:
    """Instance of class with method with type annotations.

    Method as 'func' fixture.
    """

    class A:
        """Class to hold decorated instance method."""

        @m.parse
        def func(
            self,
            a: Annotated[int, m.Parser(lambda n, obj, _: obj + 4)],
            b: Annotated[Union[str, int, float], m.Coerce(int)],
            c: Annotated[
                Union[str, int],
                m.Coerce(str),
                m.Parser(lambda n, obj, p: obj + "_suffix"),
            ],
            d: Annotated[
                Union[float, int], m.Parser(lambda n, obj, p: obj + 10), m.Coerce(str)
            ],
            e: str,
            f: Annotated[str, m.Parser(lambda name, obj, _: name + "_" + obj)],
            g: Annotated[
                str, m.Parser(lambda _, obj, params: obj + "_" + params["e"] + "_bar")
            ],
            h: Annotated[
                str,
                m.Parser(
                    lambda name, obj, params: name
                    + "_"
                    + obj
                    + "_"
                    + params["e"]
                    + "_bar"
                ),
            ],
            i: Annotated[int, "spam meta", "foo meta"],
            j: Literal["spam", "foo"],
            k: list[str],
            l: dict[str, int],
            m: abc.Mapping[str, int],
            n: tuple[str, ...],
            o: tuple[str, int, set],
            p: set[str],
            q: abc.Sequence[Union[str, int, set]],
            r: abc.Sequence[str],
            s: abc.Callable[[str], str],
            t: abc.Callable[[str, int], str],
            u: abc.Callable[..., str],
            v: abc.Callable[..., str],
            w: Union[int, str, None, Literal["spam", "foo"]],
            x: Annotated[Union[str, int, float], "spam meta", "foo meta"],
            y: Annotated[
                Optional[Union[str, int, float]],
                "foo meta",
                m.Parser(lambda n, o, p: o),
            ],
            z: Annotated[
                Optional[Union[str, int, float]],
                "spam meta",
                m.Parser(lambda n, o, p: o),
            ] = None,  # this annotation requires fixing by `fix_hints_for_none_default``
            aa: Annotated[Literal[3, 4, 5], "spam meta", "foo meta"] = 4,
            bb: Optional[int] = None,
            cc: bool = True,
            dd: Any = 4,
            ee: Optional[dict[str, Any]] = None,
            *,
            kwonly_req_a: bool,
            kwonly_req_b: typing.Annotated[Optional[bool], "meta"],
            kwonly_opt: typing.Annotated[Optional[bool], "meta"] = None,
            kwonly_opt_b: typing.Annotated[Any, "meta"] = "kwonly_opt_b",
        ) -> dict[str, Any]:
            return dict(
                a=a,
                b=b,
                c=c,
                d=d,
                e=e,
                f=f,
                g=g,
                h=h,
                i=i,
                j=j,
                k=k,
                l=l,
                m=m,
                n=n,
                o=o,
                p=p,
                q=q,
                r=r,
                s=s,
                t=t,
                u=u,
                v=v,
                w=w,
                x=x,
                y=y,
                z=z,
                aa=aa,
                bb=bb,
                cc=cc,
                dd=dd,
                ee=ee,
                kwonly_req_a=kwonly_req_a,
                kwonly_req_b=kwonly_req_b,
                kwonly_opt=kwonly_opt,
                kwonly_opt_b=kwonly_opt_b,
            )

    yield A()


@pytest.fixture
def dflt_values() -> abc.Iterator[dict[str, Any]]:
    """Default values of optional arguments (positional and keyword-only)."""
    yield dict(z=None, aa=4, bb=None, cc=True, kwonly_opt=None)


def _func(x: Any) -> Any:
    return x


FUNC = _func


# FIXTURES valid args
@pytest.fixture
def valid_args_req_as_kwargs() -> abc.Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Valid values and expected returns for required positional arguments.

    Verifying inputs against expected tests:
        Parser and Coerce, independently and together
            Tests each of name, object and params fields of `Parser.function`
        simple type annotation (e.g, param: str)
        annotations: List, Dict, Literal, Set, Tuple, Sequence, Callable
    """
    inputs = dict(
        a=4,
        b="3",
        c="7",
        d=4.5,
        e="param_e",
        f="param_f",
        g="param_g",
        h="param_h",
        i=3,
        j="foo",
        k=["foo", "bar"],
        l={"foo": 1},
        m={"foo": 1},
        n=("foo", "bar"),
        o=("foo", 1, {"spam", "bar"}),
        p={"spam", "bar"},
        q=("foo", 1, {"spam", "bar"}),
        r=["foo", "bar"],
        s=FUNC,
        t=FUNC,
        u=FUNC,
        v=FUNC,
        w=3,
        x=3,
        y=3,
    )
    expected_rtrns = dict(
        a=8,  # parsed
        b=3,  # coerced
        c="7_suffix",  # coerced then parsed
        d="14.5",  # parsed then coerced - verifies order of Parser and Coerce matters
        e="param_e",  # as passed
        f="f_param_f",  # parses using name parser's name arg
        g="param_g_param_e_bar",  # parses using value from earlier parsed parameter
        # parses using value from earlier parsed parameter and includes name
        h="h_param_h_param_e_bar",
        # as passed...
        i=3,
        j="foo",
        k=["foo", "bar"],
        l={"foo": 1},
        m={"foo": 1},
        n=("foo", "bar"),
        o=("foo", 1, {"spam", "bar"}),
        p={"spam", "bar"},
        q=("foo", 1, {"spam", "bar"}),
        r=["foo", "bar"],
        s=FUNC,
        t=FUNC,
        u=FUNC,
        v=FUNC,
        w=3,
        x=3,
        y=3,
    )
    yield inputs, expected_rtrns


@pytest.fixture
def valid_args_opt_as_kwargs() -> abc.Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Valid values and returns for optional positional arguments.

    Values differ from default values.
    """
    inputs = dict(
        z=3,
        aa=3,
        bb=3,
        cc=False,
        dd="dd",
        ee={"one": 1, "two": 2.2, "three": "three"},
    )
    expected_rtrns = inputs.copy()  # all returned as passed
    yield inputs, expected_rtrns


@pytest.fixture
def valid_args_req(valid_args_req_as_kwargs) -> abc.Iterator[list[Any]]:
    """Valid values for required positional arguments."""
    yield list(valid_args_req_as_kwargs[0].values())


@pytest.fixture
def valid_args_opt(valid_args_opt_as_kwargs) -> abc.Iterator[list[Any]]:
    """Valid values for optional positional arguments."""
    yield list(valid_args_opt_as_kwargs[0].values())


@pytest.fixture
def valid_args(valid_args_req, valid_args_opt) -> abc.Iterator[list[Any]]:
    """Valid values for arguments that can be passed positionally."""
    yield valid_args_req + valid_args_opt


@pytest.fixture
def valid_args_as_kwargs(
    valid_args_req_as_kwargs, valid_args_opt_as_kwargs
) -> abc.Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Valid values and returns for arguments that can be passed positionally."""
    inputs = valid_args_req_as_kwargs[0] | valid_args_opt_as_kwargs[0]
    expected_rtrns = valid_args_req_as_kwargs[1] | valid_args_opt_as_kwargs[1]
    yield inputs, expected_rtrns


# FIXTURES valid kwargs
@pytest.fixture
def valid_kwargs_req() -> abc.Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Valid values and expected returns for required keyword-only arguments."""
    inputs = dict(kwonly_req_a=False, kwonly_req_b=None)
    expected_rtrns = dict(kwonly_req_a=False, kwonly_req_b=None)
    yield inputs, expected_rtrns


@pytest.fixture
def valid_kwargs_opt() -> abc.Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Valid values and returns for optional keyword-only arguments.

    Values differ from default values.
    """
    inputs = dict(kwonly_opt=True, kwonly_opt_b=22)
    expected_rtrns = inputs.copy()  # all returned as passed
    yield inputs, expected_rtrns


@pytest.fixture
def valid_kwargs(
    valid_kwargs_req, valid_kwargs_opt
) -> abc.Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Valid values for all keyword-only arguments."""
    inputs = valid_kwargs_req[0] | valid_kwargs_opt[0]
    expected_rtrns = valid_kwargs_req[1] | valid_kwargs_opt[1]
    yield inputs, expected_rtrns


# FIXTURES valid args for ALL parameters
@pytest.fixture
def valid_args_all(
    valid_args_as_kwargs, valid_kwargs
) -> abc.Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Valid values and expected returns for all arguments."""
    inputs = valid_args_as_kwargs[0] | valid_kwargs[0]
    expected_rtrns = valid_args_as_kwargs[1] | valid_kwargs[1]
    yield inputs, expected_rtrns


def assertion(rtrn: Any, expected: Any):
    if rtrn is None:
        assert expected is None
    else:
        assert rtrn == expected


def test_general_valid(
    f,
    inst,
    valid_args_req_as_kwargs,
    valid_kwargs_req,
    dflt_values,
    valid_args_req,
    valid_args_opt_as_kwargs,
    valid_args_opt,
    valid_kwargs_opt,
):
    """General for valid inputs.

    Tests return as expected when passing:
        pos req args as kwargs, passing no optional args and verifying defaults.
        pos req args positionally, passing no optional args and verifying defaults.
        pos req args positionally, pos opt args positionally, all keyword-only
        pos req args positionally, pos opt args as kwarg, all keyword-only
    """
    for func in (f, inst.func):
        rtrns0 = func(**valid_args_req_as_kwargs[0], **valid_kwargs_req[0])
        rtrns1 = func(*valid_args_req, **valid_kwargs_req[0])
        rtrns2 = func(
            *valid_args_req,
            *valid_args_opt,
            **valid_kwargs_req[0],
            **valid_kwargs_opt[0],
        )
        rtrns3 = func(
            *valid_args_req,
            **valid_args_opt_as_kwargs[0],
            **valid_kwargs_req[0],
            **valid_kwargs_opt[0],
        )

        for k, v in valid_args_req_as_kwargs[1].items():
            assertion(rtrns0[k], v)
            assertion(rtrns1[k], v)
            assertion(rtrns2[k], v)
            assertion(rtrns3[k], v)

        for k, v in valid_kwargs_req[1].items():
            assertion(rtrns0[k], v)
            assertion(rtrns1[k], v)
            assertion(rtrns2[k], v)
            assertion(rtrns3[k], v)

        # as default values
        for k, v in dflt_values.items():
            assertion(rtrns0[k], v)
            assertion(rtrns1[k], v)

        # as expected non-default values

        for k, v in valid_args_opt_as_kwargs[1].items():
            assertion(rtrns2[k], v)
            assertion(rtrns3[k], v)
        for k, v in valid_kwargs_opt[1].items():
            assertion(rtrns2[k], v)
            assertion(rtrns3[k], v)


def test_union_optional_valid(f, valid_args_all):
    """Additional tests for Union and Optional annotations.

    Furthers `test_general_valid` to cover additional valid inputs to
    parameters annotated with Union and Option.
    """
    inputs, expected_rtrns = valid_args_all[0].copy(), valid_args_all[1].copy()
    chgs = dict(
        w="param_w",
        x="param_x",
        y="param_y",
        z="param_z",
        #
        bb=None,
        kwonly_req_b=False,
    )
    inputs |= chgs
    expected_rtrns |= chgs

    rtrns = f(**inputs)
    for k, v in rtrns.items():
        assertion(v, expected_rtrns[k])

    chgs_1 = dict(
        w=None,
        x=1.0,
        y=1.0,
        z=1.0,
    )
    inputs |= chgs_1
    expected_rtrns |= chgs_1

    rtrns = f(**inputs)
    for k, v in rtrns.items():
        assertion(v, expected_rtrns[k])

    chgs_2 = dict(
        y=None,
        z=None,
    )
    inputs |= chgs_2
    expected_rtrns |= chgs_2

    rtrns = f(**inputs)
    for k, v in rtrns.items():
        assertion(v, expected_rtrns[k])


def test_tuple_valid():
    """Test valid inputs for annotation of format tuple[Union[<type>, <type>], ...]."""

    @m.parse
    def f(
        a: tuple[Union[str, int, set], ...],
        b: Union[str, tuple[Union[str, int, set], ...]],
        c: Optional[tuple[Union[str, int, set], ...]],
        d: Annotated[tuple[Union[str, int, set], ...], "some_meta_d"],
        e: Annotated[Union[str, tuple[Union[str, int, set], ...]], "some_meta_e"],
        f: tuple[Union[str, int, set], ...],
    ) -> dict[str, Any]:
        return dict(a=a, b=b, c=c, d=d, e=e, f=f)

    rtrn = f(
        ("zero", "one", "two"),
        (0, 1, 2),
        ({0.0, "zero", (0.0, 0.1)}, {1.1, "one", (1.0, 1.1)}, {2.2, "two", (2.0, 2.1)}),
        ("zero", 1, {2.2}),
        ("zero", "one", "two", 3),
        (0, 1, 2, {3}),
    )

    assert rtrn["a"] == ("zero", "one", "two")
    assert rtrn["b"] == (0, 1, 2)
    expected_c = (
        {0.0, "zero", (0.0, 0.1)},
        {1.1, "one", (1.0, 1.1)},
        {2.2, "two", (2.0, 2.1)},
    )
    assert rtrn["c"] == expected_c
    assert rtrn["d"] == ("zero", 1, {2.2})
    assert rtrn["e"] == ("zero", "one", "two", 3)
    assert rtrn["f"] == (0, 1, 2, {3})


INVALID_MSG = re.escape(
    """The following inputs to 'func' do not conform with the corresponding type annotation:

h
	Takes type <class 'str'> although received '3' of type <class 'int'>.

i
	Takes type <class 'int'> although received 'three' of type <class 'str'>.

j
	Takes a value from <('spam', 'foo')> although received 'not in literal'.

k
	Takes type <class 'list'> although received '{'not': 'a list'}' of type <class 'dict'>.

l
	Takes type <class 'dict'> although received '['not a dict']' of type <class 'list'>.

m
	Takes type <class 'collections.abc.Mapping'> although received '['not a mapping']' of type <class 'list'>.

n
	Takes type <class 'tuple'> although received '['not a tuple']' of type <class 'list'>.

p
	Takes type <class 'set'> although received '['not a set']' of type <class 'list'>.

q
	Takes type <class 'collections.abc.Sequence'> although received '{'not': 'a sequence'}' of type <class 'dict'>.

s
	Takes type <class 'collections.abc.Callable'> although received 'not callable' of type <class 'str'>.

w
	Takes input that conforms with <(<class 'int'>, <class 'str'>, <class 'NoneType'>, typing.Literal['spam', 'foo'])> although received '['list not in union']' of type <class 'list'>.

x
	Takes input that conforms with <(<class 'str'>, <class 'int'>, <class 'float'>)> although received '{'dict': 'not in annotated union'}' of type <class 'dict'>.

y
	Takes input that conforms with <(<class 'str'>, <class 'int'>, <class 'float'>, <class 'NoneType'>)> although received '{'dict': 'not in annotated union'}' of type <class 'dict'>.

z
	Takes input that conforms with <(<class 'str'>, <class 'int'>, <class 'float'>, <class 'NoneType'>)> although received '{'dict': 'not in annotated union'}' of type <class 'dict'>.

aa
	Takes a value from <(3, 4, 5)> although received '6'.

bb
	Takes input that conforms with <(<class 'int'>, <class 'NoneType'>)> although received 'not an int' of type <class 'str'>.

cc
	Takes type <class 'bool'> although received 'not a bool' of type <class 'str'>.

kwonly_req_a
	Takes type <class 'bool'> although received 'not a bool' of type <class 'str'>."""
)


def test_invalid_types(f, inst, valid_args):
    regex = re.compile("^" + INVALID_MSG + "$")
    for func in (f, inst.func):
        with pytest.raises(m.InputsError, match=regex):
            func(
                *valid_args[:7],
                h=3,
                i="three",
                j="not in literal",
                k={"not": "a list"},
                l=["not a dict"],
                m=["not a mapping"],
                n=["not a tuple"],
                o=("foo", 1, {"spam", "bar"}),  # valid
                p=["not a set"],
                q={"not": "a sequence"},
                r=["foo", "bar"],  # valid
                s="not callable",
                t=lambda x: x,  # valid
                u=lambda x: x,  # valid
                v=lambda x: x,  # valid
                w=["list not in union"],
                x={"dict": "not in annotated union"},
                y={"dict": "not in annotated union"},
                z={"dict": "not in annotated union"},
                aa=6,
                bb="not an int",
                cc="not a bool",
                kwonly_req_a="not a bool",
                kwonly_req_b=None,  # valid
            )


INVALID_MSG_SIG_SINGLE = re.escape(
    """Inputs to 'func' do not conform with the function signature:

Got multiple values for argument: 'a'.

Got unexpected keyword argument: 'not_a_kwarg'.

Missing 1 positional argument: 'y'.

Missing 1 keyword-only argument: 'kwonly_req_b'.

The following inputs to 'func' do not conform with the corresponding type annotation:

b
	Takes input that conforms with <(<class 'str'>, <class 'int'>, <class 'float'>)> although received '[1]' of type <class 'list'>."""
)

INVALID_MSG_SIG_SINGLE_1 = re.escape(
    """Inputs to 'func' do not conform with the function signature:

Got multiple values for argument: 'a'.

Received 1 excess positional argument as:
	'4' of type <class 'int'>.

Got unexpected keyword argument: 'not_a_kwarg'.

Missing 1 keyword-only argument: 'kwonly_req_b'.

The following inputs to 'func' do not conform with the corresponding type annotation:

b
	Takes input that conforms with <(<class 'str'>, <class 'int'>, <class 'float'>)> although received '[1]' of type <class 'list'>."""
)


def test_invalid_sig_single(f, valid_args_req, valid_args, valid_kwargs_req):
    """Test for invalid signature errors.

    Matches expected message when one argument fails for each error.
    """
    regex = re.compile("^" + INVALID_MSG_SIG_SINGLE + "$")
    with pytest.raises(m.InputsError, match=regex):
        f(
            0,
            [1],
            *valid_args_req[2:-1],
            a=2,
            kwonly_req_a=valid_kwargs_req[0]["kwonly_req_a"],
            not_a_kwarg="foo",
        )

    # as above, although tests for an excess pos arg as opposed to a missing pos arg
    regex = re.compile("^" + INVALID_MSG_SIG_SINGLE_1 + "$")
    with pytest.raises(m.InputsError, match=regex):
        f(
            0,
            [1],
            *valid_args[2:],
            4,
            a=2,
            kwonly_req_a=valid_kwargs_req[0]["kwonly_req_a"],
            not_a_kwarg="foo",
        )


INVALID_MSG_MULT_SINGLE = re.escape(
    """Inputs to 'func' do not conform with the function signature:

Got multiple values for arguments: 'a' and 'b'.

Got unexpected keyword arguments: 'not_a_kwarg' and 'another_not_a_kwarg'.

Missing 2 positional arguments: 'x' and 'y'.

Missing 2 keyword-only arguments: 'kwonly_req_a' and 'kwonly_req_b'.

The following inputs to 'func' do not conform with the corresponding type annotation:

c
	Takes input that conforms with <(<class 'str'>, <class 'int'>)> although received '2.0' of type <class 'float'>.

e
	Takes type <class 'str'> although received '4' of type <class 'int'>."""
)

INVALID_MSG_MULT_SINGLE_1 = re.escape(
    """Inputs to 'func' do not conform with the function signature:

Got multiple values for arguments: 'a' and 'b'.

Received 2 excess positional arguments as:
	'4' of type <class 'int'>.
	'[5]' of type <class 'list'>.

Got unexpected keyword arguments: 'not_a_kwarg' and 'another_not_a_kwarg'.

Missing 2 keyword-only arguments: 'kwonly_req_a' and 'kwonly_req_b'.

The following inputs to 'func' do not conform with the corresponding type annotation:

c
	Takes input that conforms with <(<class 'str'>, <class 'int'>)> although received '2.0' of type <class 'float'>.

e
	Takes type <class 'str'> although received '4' of type <class 'int'>."""
)


def test_invalid_sig_multiple(f, valid_args_req, valid_args):
    """Test for invalid signature errors.

    Matches expected message when two arguments fail for each error.
    """
    regex = re.compile("^" + INVALID_MSG_MULT_SINGLE + "$")
    with pytest.raises(m.InputsError, match=regex):
        f(
            0,
            1,
            2.0,
            3,  # valid
            4,
            *valid_args_req[5:-2],
            a=2,
            b=3,
            not_a_kwarg="foo",
            another_not_a_kwarg="bar",
        )

    # as above, although tests for excess pos args as opposed to missing pos args
    regex = re.compile("^" + INVALID_MSG_MULT_SINGLE_1 + "$")
    with pytest.raises(m.InputsError, match=regex):
        f(
            0,
            1,
            2.0,
            3,  # valid
            4,
            *valid_args[5:],
            4,
            [5],
            a=2,
            b=3,
            not_a_kwarg="foo",
            another_not_a_kwarg="bar",
        )


INVALID_MSG_SET_ITEMS = re.escape(
    """The following inputs to 'f' do not conform with the corresponding type annotation:

a
	Takes type <class 'set'> containing items that conform with <set[int]>, although the received container contains item 'zero' of type <class 'str'>.

b
	Takes input that conforms with <(<class 'str'>, set[int])> although received '{'one', 'zero'}'.

c
	Takes input that conforms with <(set[int], <class 'NoneType'>)> although received '{0, 2, 'one'}'.

d
	Takes type <class 'set'> containing items that conform with <set[int]>, although the received container contains item 'two' of type <class 'str'>.

e
	Takes input that conforms with <(<class 'str'>, set[int])> although received '{1, 2, 'zero'}'."""
)


def test_invalid_inputs_set_items():
    @m.parse
    def f(
        a: set[int],
        b: Union[str, set[int]],
        c: Optional[set[int]],
        d: Annotated[set[int], "some_meta_c"],
        e: Annotated[Union[str, set[int]], "some_meta_d"],
    ):
        return

    # selected parts or error message only given unable to replicate ordering of a set
    msg_a = re.escape(
        "Takes type <class 'set'> containing items that conform with <set[int]>, although the received container contains item 'zero' of type <class 'str'>."
    )
    msg_b = re.escape(
        "Takes input that conforms with <(<class 'str'>, set[int])> although received '"
    )
    msg_c = re.escape(
        "Takes input that conforms with <(set[int], <class 'NoneType'>)> although received '"
    )
    msg_d = re.escape(
        "Takes type <class 'set'> containing items that conform with <set[int]>, although the received container contains item 'two' of type <class 'str'>."
    )
    msg_e = re.escape(
        "Takes input that conforms with <(<class 'str'>, set[int])> although received '"
    )

    pat = f".*(?=a\n\t{msg_a}).*(?=b\n\t{msg_b}).*(?=c\n\t{msg_c}).*(?=d\n\t{msg_d}).*(?=e\n\t{msg_e}).*"

    regex = re.compile("^" + pat + "$", re.DOTALL)
    with pytest.raises(m.InputsError, match=regex):
        f(
            {"zero", 1},  # includes non-int
            {"zero", "one"},  # all non-int
            {0, "one", 2},
            {0, 1, "two"},
            {"zero", 1, 2},
        )


INVALID_MSG_LIST_ITEMS = re.escape(
    """The following inputs to 'f' do not conform with the corresponding type annotation:

a
	Takes type <class 'list'> containing items that conform with <list[int]>, although the received container contains item 'one' of type <class 'str'>.

b
	Takes input that conforms with <(<class 'str'>, list[int])> although received '['one', 'one']' of type <class 'list'>.

c
	Takes input that conforms with <(list[int], <class 'NoneType'>)> although received '[1, 'one', 1]' of type <class 'list'>.

d
	Takes type <class 'list'> containing items that conform with <list[int]>, although the received container contains item 'one' of type <class 'str'>.

e
	Takes input that conforms with <(<class 'str'>, list[int])> although received '['one', 1, 1]' of type <class 'list'>."""
)


def test_invalid_inputs_list_items():
    @m.parse
    def f(
        a: list[int],
        ax: Annotated[list[int], m.NO_ITEM_VALIDATION],
        b: Union[str, list[int]],
        bx: Annotated[Union[str, list[int]], m.NO_ITEM_VALIDATION],
        c: Optional[list[int]],
        cx: Annotated[Optional[list[int]], m.NO_ITEM_VALIDATION],
        d: Annotated[list[int], "some_meta_d"],
        dx: Annotated[list[int], "some_meta_d", m.NO_ITEM_VALIDATION],
        e: Annotated[Union[str, list[int]], "some_meta_e"],
        ex: Annotated[Union[str, list[int]], m.NO_ITEM_VALIDATION, "some_meta_e"],
    ):
        return

    regex = re.compile("^" + INVALID_MSG_LIST_ITEMS + "$")
    with pytest.raises(m.InputsError, match=regex):
        f(
            ["one", 1],  # list includes non-int
            ["one", 1],  # again, now valid as NO_ITEM_VALIDATION
            ["one", "one"],  # all non-int
            ["one", "one"],  # again, now valid as NO_ITEM_VALIDATION
            [1, "one", 1],
            [1, "one", 1],  # again, now valid as NO_ITEM_VALIDATION
            [1, 1, "one"],
            [1, 1, "one"],  # again, now valid as NO_ITEM_VALIDATION
            ["one", 1, 1],
            ["one", 1, 1],  # again, now valid as NO_ITEM_VALIDATION
        )


INVALID_MSG_SEQ_ITEMS = re.escape(
    """The following inputs to 'f' do not conform with the corresponding type annotation:

a
	Takes type <class 'collections.abc.Sequence'> containing items that conform with <collections.abc.Sequence[int]>, although the received container contains item 'one' of type <class 'str'>.

b
	Takes input that conforms with <(<class 'str'>, collections.abc.Sequence[int])> although received '['one', 'one']' of type <class 'list'>.

c
	Takes input that conforms with <(collections.abc.Sequence[int], <class 'NoneType'>)> although received '[1, 'one', 1]' of type <class 'list'>.

d
	Takes type <class 'collections.abc.Sequence'> containing items that conform with <collections.abc.Sequence[int]>, although the received container contains item 'one' of type <class 'str'>.

e
	Takes input that conforms with <(<class 'str'>, collections.abc.Sequence[int])> although received '['one', 1, 1]' of type <class 'list'>."""
)


def test_invalid_inputs_seq_items():
    @m.parse
    def f(
        a: abc.Sequence[int],
        ax: Annotated[abc.Sequence[int], m.NO_ITEM_VALIDATION],
        b: Union[str, abc.Sequence[int]],
        bx: Annotated[Union[str, abc.Sequence[int]], m.NO_ITEM_VALIDATION],
        c: Optional[abc.Sequence[int]],
        cx: Annotated[Optional[abc.Sequence[int]], m.NO_ITEM_VALIDATION],
        d: Annotated[abc.Sequence[int], "some_meta_d"],
        dx: Annotated[abc.Sequence[int], "some_meta_d", m.NO_ITEM_VALIDATION],
        e: Annotated[Union[str, abc.Sequence[int]], "some_meta_e"],
        ex: Annotated[
            Union[str, abc.Sequence[int]], m.NO_ITEM_VALIDATION, "some_meta_e"
        ],
    ):
        return

    regex = re.compile("^" + INVALID_MSG_SEQ_ITEMS + "$")
    with pytest.raises(m.InputsError, match=regex):
        f(
            ["one", 1],  # list includes non-int
            ["one", 1],  # again, now valid as NO_ITEM_VALIDATION
            ["one", "one"],  # all non-int
            ["one", "one"],  # again, now valid as NO_ITEM_VALIDATION
            [1, "one", 1],
            [1, "one", 1],  # again, now valid as NO_ITEM_VALIDATION
            [1, 1, "one"],
            [1, 1, "one"],  # again, now valid as NO_ITEM_VALIDATION
            ["one", 1, 1],
            ["one", 1, 1],  # again, now valid as NO_ITEM_VALIDATION
        )


INVALID_MSG_TUPLE_ITEMS = re.escape(
    """The following inputs to 'f' do not conform with the corresponding type annotation:

a
	Takes type <class 'tuple'> containing items that conform with <tuple[int, ...]>, although the received container contains item '0.0' of type <class 'float'>.

a1
	Takes type <class 'tuple'> containing items that conform with <tuple[typing.Union[str, int, set], ...]>, although the received container contains item '0.0' of type <class 'float'>.

b
	Takes input that conforms with <(<class 'str'>, tuple[int, ...])> although received '(0.0, 1.1, 2.2)' of type <class 'tuple'>.

b1
	Takes input that conforms with <(<class 'str'>, tuple[typing.Union[str, int, set], ...])> although received '(0.0, 1.1, 2.2)' of type <class 'tuple'>.

c
	Takes input that conforms with <(tuple[int, ...], <class 'NoneType'>)> although received '(0, 1, 2.2)' of type <class 'tuple'>.

c1
	Takes input that conforms with <(tuple[typing.Union[str, int, set], ...], <class 'NoneType'>)> although received '('zero', 1, {2.222}, 3.3)' of type <class 'tuple'>.

d
	Takes type <class 'tuple'> containing items that conform with <tuple[int, ...]>, although the received container contains item '1.1' of type <class 'float'>.

d1
	Takes type <class 'tuple'> containing items that conform with <tuple[typing.Union[str, int, set], ...]>, although the received container contains item '2.2' of type <class 'float'>.

e
	Takes input that conforms with <(<class 'str'>, tuple[int, ...])> although received '(0.0, 1, 2)' of type <class 'tuple'>.

e1
	Takes input that conforms with <(<class 'str'>, tuple[typing.Union[str, int, set], ...])> although received '(0.0, 1, 'two', {3.333})' of type <class 'tuple'>.

f
	Takes type <class 'tuple'> containing items that conform with <tuple[str, int, set]>, although the item in position 0 is '0' of type <class 'int'>.

g
	Takes input that conforms with <(<class 'str'>, tuple[str, int, set])> although received '(0.0, 1, {2.2})' of type <class 'tuple'>.

h
	Takes input that conforms with <(tuple[str, int, set], <class 'NoneType'>)> although received '('zero', 1, 2.2)' of type <class 'tuple'>.

i
	Takes type <class 'tuple'> containing items that conform with <tuple[str, int, set]>, although the item in position 1 is '1.1' of type <class 'float'>.

j
	Takes input that conforms with <(<class 'str'>, tuple[str, int, set])> although received '('zero', {1}, 2.2)' of type <class 'tuple'>.

k
	Takes type <class 'tuple'> of length 3 although received '(0, {1})' of length 2.

krpt
	Takes type <class 'tuple'> of length 3 although received '(0, {1})' of length 2.

l
	Takes type <class 'tuple'> of length 3 although received '(0, {1}, 'two', 'three')' of length 4."""
)


def test_invalid_inputs_tuple_items():
    @m.parse
    def f(
        a: tuple[int, ...],
        a1: tuple[Union[str, int, set], ...],
        b: Union[str, tuple[int, ...]],
        b1: Union[str, tuple[Union[str, int, set], ...]],
        c: Optional[tuple[int, ...]],
        c1: Optional[tuple[Union[str, int, set], ...]],
        d: Annotated[tuple[int, ...], "some_meta_c"],
        d1: Annotated[tuple[Union[str, int, set], ...], "some_meta_c"],
        e: Annotated[Union[str, tuple[int, ...]], "some_meta_d"],
        e1: Annotated[Union[str, tuple[Union[str, int, set], ...]], "some_meta_d"],
        #
        f: tuple[str, int, set],
        fx: Annotated[tuple[str, int, set], m.NO_ITEM_VALIDATION],
        g: Union[str, tuple[str, int, set]],
        h: Optional[tuple[str, int, set]],
        i: Annotated[tuple[str, int, set], "some_meta_c"],
        j: Annotated[Union[str, tuple[str, int, set]], "some_meta_d"],
        #
        k: tuple[str, int, set],
        krpt: tuple[str, int, set],
        l: tuple[str, int, set],
    ):
        return

    regex = re.compile("^" + INVALID_MSG_TUPLE_ITEMS + "$")
    with pytest.raises(m.InputsError, match=regex):
        f(
            (0.0,),  # single item invalid
            (0.0,),
            (0.0, 1.1, 2.2),  # all invalid
            (0.0, 1.1, 2.2),
            (0, 1, 2.2),  # includes one invalid at end
            ("zero", 1, {2.222}, 3.3),
            (0, 1.1, 2),  # includes one invalid in middle
            (0, "one", 2.2, {3.333}),
            (0.0, 1, 2),  # includes one invalid at start
            (0.0, 1, "two", {3.333}),
            #
            (0, {1}, "two"),  # all invalid
            (0, {1}, "two"),  # all invalid but not checking as NO_ITEM_VALIDATION
            (0.0, 1, {2.2}),  # first invalid
            ("zero", 1, 2.2),  # last invalid
            ("zero", 1.1, {2.2}),  # middle invalid
            ("zero", {1}, 2.2),  # only first valid
            (0, {1}),  # too short
            (0, {1}),  # too short, NO_ITEM_VALIDATION shoudn't make any difference
            (0, {1}, "two", "three"),  # too long
        )


INVALID_MSG_DICT_ITEMS = re.escape(
    """The following inputs to 'f' do not conform with the corresponding type annotation:

a
	Takes type <class 'dict'> with keys that conform to the first argument of <dict[str, int]>, although the received dictionary contains key '0' of type <class 'int'>.

a1
	Takes type <class 'dict'> with values that conform to the second argument of <dict[str, int]>, although the received dictionary contains value 'val0' of type <class 'str'>.

a2
	Takes type <class 'dict'> with keys that conform to the first argument and values that conform to the second argument of <dict[str, int]>, although the received dictionary contains an item with key '0' of type <class 'int'> and value 'val0' of type <class 'str'>.

b
	Takes input that conforms with <(<class 'str'>, dict[str, int])> although received '{0: 0, 1: 1}' of type <class 'dict'>.

b1
	Takes input that conforms with <(<class 'str'>, dict[str, int])> although received '{'key0': 'val0', 'key1': 'val1'}' of type <class 'dict'>.

b2
	Takes input that conforms with <(<class 'str'>, dict[str, int])> although received '{0: 'val0', 1: 'val1'}' of type <class 'dict'>.

c
	Takes input that conforms with <(dict[str, int], <class 'NoneType'>)> although received '{'key0': 0, 1: 1, 'key2': 2}' of type <class 'dict'>.

c1
	Takes input that conforms with <(dict[str, int], <class 'NoneType'>)> although received '{'key0': 0, 'key1': 'val1', 'key2': 2}' of type <class 'dict'>.

c2
	Takes input that conforms with <(dict[str, int], <class 'NoneType'>)> although received '{'key0': 0, 1: 'val1', 'key2': 2}' of type <class 'dict'>.

d
	Takes type <class 'dict'> with keys that conform to the first argument of <dict[str, int]>, although the received dictionary contains key '2' of type <class 'int'>.

d1
	Takes type <class 'dict'> with values that conform to the second argument of <dict[str, int]>, although the received dictionary contains value 'val2' of type <class 'str'>.

d2
	Takes type <class 'dict'> with keys that conform to the first argument and values that conform to the second argument of <dict[str, int]>, although the received dictionary contains an item with key '2' of type <class 'int'> and value 'val2' of type <class 'str'>.

e
	Takes input that conforms with <(<class 'str'>, dict[str, int])> although received '{0: 0, 'key1': 1, 'key2': 2}' of type <class 'dict'>.

e1
	Takes input that conforms with <(<class 'str'>, dict[str, int])> although received '{'key0': 'val0', 'key1': 1, 'key2': 2}' of type <class 'dict'>.

e2
	Takes input that conforms with <(<class 'str'>, dict[str, int])> although received '{0: 'val0', 'key1': 1, 'key2': 2}' of type <class 'dict'>."""
)


def test_invalid_inputs_dict_items():
    @m.parse
    def f(
        a: dict[str, int],
        a1: dict[str, int],
        a2: dict[str, int],
        ax: Annotated[dict[str, int], m.NO_ITEM_VALIDATION],
        b: Union[str, dict[str, int]],
        b1: Union[str, dict[str, int]],
        b2: Union[str, dict[str, int]],
        bx: Annotated[Union[str, dict[str, int]], m.NO_ITEM_VALIDATION],
        c: Optional[dict[str, int]],
        c1: Optional[dict[str, int]],
        c2: Optional[dict[str, int]],
        cx: Annotated[Optional[dict[str, int]], m.NO_ITEM_VALIDATION],
        d: Annotated[dict[str, int], "some_meta_d"],
        d1: Annotated[dict[str, int], "some_meta_d"],
        d2: Annotated[dict[str, int], "some_meta_d"],
        dx: Annotated[dict[str, int], "some_meta_d", m.NO_ITEM_VALIDATION],
        e: Annotated[Union[str, dict[str, int]], "some_meta_e"],
        e1: Annotated[Union[str, dict[str, int]], "some_meta_e"],
        e2: Annotated[Union[str, dict[str, int]], "some_meta_e"],
        ex: Annotated[Union[str, dict[str, int]], m.NO_ITEM_VALIDATION, "some_meta_e"],
    ):
        return

    regex = re.compile("^" + INVALID_MSG_DICT_ITEMS + "$")
    with pytest.raises(m.InputsError, match=regex):
        f(
            {0: 0, "key1": 1},  # includes non-str key
            {"key0": "val0", "key1": 1},  # includes non-int value
            {0: "val0", "key1": 1},  # includes non-str key and non-int value
            {0: "val0", "key1": 1},  # ...again, but not checking as NO_ITEM_VALIDATION
            #
            {0: 0, 1: 1},  # all keys non-str
            {"key0": "val0", "key1": "val1"},  # all values non-int
            {0: "val0", 1: "val1"},  # all keys non-str and all values non-int
            {0: "val0", 1: "val1"},  # ...again, but not checking as NO_ITEM_VALIDATION
            #
            {"key0": 0, 1: 1, "key2": 2},  # includes non-str key
            {"key0": 0, "key1": "val1", "key2": 2},  # includes non-int value
            {"key0": 0, 1: "val1", "key2": 2},  # includes non-str key and non-int value
            {
                "key0": 0,
                1: "val1",
                "key2": 2,
            },  # ...again, but not checking as NO_ITEM_VALIDATION
            #
            {"key0": 0, "key1": 1, 2: 2},  # includes non-str key
            {"key0": 0, "key1": 1, "key2": "val2"},  # includes non-int value
            {"key0": 0, "key1": 1, 2: "val2"},  # includes non-str key and non-int value
            {
                "key0": 0,
                "key1": 1,
                2: "val2",
            },  # ...again, but not checking as NO_ITEM_VALIDATION
            #
            {0: 0, "key1": 1, "key2": 2},  # includes non-str key
            {"key0": "val0", "key1": 1, "key2": 2},  # includes non-int value
            {0: "val0", "key1": 1, "key2": 2},  # includes non-str key and non-int value
            {
                0: "val0",
                "key1": 1,
                "key2": 2,
            },  # ...again, but not checking as NO_ITEM_VALIDATION
        )


INVALID_MSG_MAPPING_ITEMS = re.escape(
    """The following inputs to 'f' do not conform with the corresponding type annotation:

a
	Takes type <class 'collections.abc.Mapping'> with keys that conform to the first argument of <collections.abc.Mapping[str, int]>, although the received mapping contains key '0' of type <class 'int'>.

a1
	Takes type <class 'collections.abc.Mapping'> with values that conform to the second argument of <collections.abc.Mapping[str, int]>, although the received mapping contains value 'val0' of type <class 'str'>.

a2
	Takes type <class 'collections.abc.Mapping'> with keys that conform to the first argument and values that conform to the second argument of <collections.abc.Mapping[str, int]>, although the received mapping contains an item with key '0' of type <class 'int'> and value 'val0' of type <class 'str'>.

b
	Takes input that conforms with <(<class 'str'>, collections.abc.Mapping[str, int])> although received '{0: 0, 1: 1}' of type <class 'dict'>.

b1
	Takes input that conforms with <(<class 'str'>, collections.abc.Mapping[str, int])> although received '{'key0': 'val0', 'key1': 'val1'}' of type <class 'dict'>.

b2
	Takes input that conforms with <(<class 'str'>, collections.abc.Mapping[str, int])> although received '{0: 'val0', 1: 'val1'}' of type <class 'dict'>.

c
	Takes input that conforms with <(collections.abc.Mapping[str, int], <class 'NoneType'>)> although received '{'key0': 0, 1: 1, 'key2': 2}' of type <class 'dict'>.

c1
	Takes input that conforms with <(collections.abc.Mapping[str, int], <class 'NoneType'>)> although received '{'key0': 0, 'key1': 'val1', 'key2': 2}' of type <class 'dict'>.

c2
	Takes input that conforms with <(collections.abc.Mapping[str, int], <class 'NoneType'>)> although received '{'key0': 0, 1: 'val1', 'key2': 2}' of type <class 'dict'>.

d
	Takes type <class 'collections.abc.Mapping'> with keys that conform to the first argument of <collections.abc.Mapping[str, int]>, although the received mapping contains key '2' of type <class 'int'>.

d1
	Takes type <class 'collections.abc.Mapping'> with values that conform to the second argument of <collections.abc.Mapping[str, int]>, although the received mapping contains value 'val2' of type <class 'str'>.

d2
	Takes type <class 'collections.abc.Mapping'> with keys that conform to the first argument and values that conform to the second argument of <collections.abc.Mapping[str, int]>, although the received mapping contains an item with key '2' of type <class 'int'> and value 'val2' of type <class 'str'>.

e
	Takes input that conforms with <(<class 'str'>, collections.abc.Mapping[str, int])> although received '{0: 0, 'key1': 1, 'key2': 2}' of type <class 'dict'>.

e1
	Takes input that conforms with <(<class 'str'>, collections.abc.Mapping[str, int])> although received '{'key0': 'val0', 'key1': 1, 'key2': 2}' of type <class 'dict'>.

e2
	Takes input that conforms with <(<class 'str'>, collections.abc.Mapping[str, int])> although received '{0: 'val0', 'key1': 1, 'key2': 2}' of type <class 'dict'>."""
)


def test_invalid_inputs_mapping_items():
    @m.parse
    def f(
        a: abc.Mapping[str, int],
        a1: abc.Mapping[str, int],
        a2: abc.Mapping[str, int],
        ax: Annotated[abc.Mapping[str, int], m.NO_ITEM_VALIDATION],
        b: Union[str, abc.Mapping[str, int]],
        b1: Union[str, abc.Mapping[str, int]],
        b2: Union[str, abc.Mapping[str, int]],
        bx: Annotated[Union[str, abc.Mapping[str, int]], m.NO_ITEM_VALIDATION],
        c: Optional[abc.Mapping[str, int]],
        c1: Optional[abc.Mapping[str, int]],
        c2: Optional[abc.Mapping[str, int]],
        cx: Annotated[Optional[abc.Mapping[str, int]], m.NO_ITEM_VALIDATION],
        d: Annotated[abc.Mapping[str, int], "some_meta_d"],
        d1: Annotated[abc.Mapping[str, int], "some_meta_d"],
        d2: Annotated[abc.Mapping[str, int], "some_meta_d"],
        dx: Annotated[abc.Mapping[str, int], "some_meta_d", m.NO_ITEM_VALIDATION],
        e: Annotated[Union[str, abc.Mapping[str, int]], "some_meta_e"],
        e1: Annotated[Union[str, abc.Mapping[str, int]], "some_meta_e"],
        e2: Annotated[Union[str, abc.Mapping[str, int]], "some_meta_e"],
        ex: Annotated[
            Union[str, abc.Mapping[str, int]], m.NO_ITEM_VALIDATION, "some_meta_e"
        ],
    ):
        return

    regex = re.compile("^" + INVALID_MSG_MAPPING_ITEMS + "$")
    with pytest.raises(m.InputsError, match=regex):
        f(
            {0: 0, "key1": 1},  # includes non-str key
            {"key0": "val0", "key1": 1},  # includes non-int value
            {0: "val0", "key1": 1},  # includes non-str key and non-int value
            {0: "val0", "key1": 1},  # ...again, but not checking as NO_ITEM_VALIDATION
            #
            {0: 0, 1: 1},  # all keys non-str
            {"key0": "val0", "key1": "val1"},  # all values non-int
            {0: "val0", 1: "val1"},  # all keys non-str and all values non-int
            {0: "val0", 1: "val1"},  # ...again, but not checking as NO_ITEM_VALIDATION
            #
            {"key0": 0, 1: 1, "key2": 2},  # includes non-str key
            {"key0": 0, "key1": "val1", "key2": 2},  # includes non-int value
            {"key0": 0, 1: "val1", "key2": 2},  # includes non-str key and non-int value
            {
                "key0": 0,
                1: "val1",
                "key2": 2,
            },  # ...again, but not checking as NO_ITEM_VALIDATION
            #
            {"key0": 0, "key1": 1, 2: 2},  # includes non-str key
            {"key0": 0, "key1": 1, "key2": "val2"},  # includes non-int value
            {"key0": 0, "key1": 1, 2: "val2"},  # includes non-str key and non-int value
            {
                "key0": 0,
                "key1": 1,
                2: "val2",
            },  # ...again, but not checking as NO_ITEM_VALIDATION
            #
            {0: 0, "key1": 1, "key2": 2},  # includes non-str key
            {"key0": "val0", "key1": 1, "key2": 2},  # includes non-int value
            {0: "val0", "key1": 1, "key2": 2},  # includes non-str key and non-int value
            {
                0: "val0",
                "key1": 1,
                "key2": 2,
            },  # ...again, but not checking as NO_ITEM_VALIDATION
        )


INVALID_MSG_NEATED_ITEMS_0 = re.escape(
    """The following inputs to 'f' do not conform with the corresponding type annotation:

a
	Takes type <class 'tuple'> containing items that conform with <tuple[list[set[typing.Union[int, float]]], ...]>, although the received container contains item '[{0}, {1.0}, 'not a set']' of type <class 'list'>."""
)


INVALID_MSG_NEATED_ITEMS_1 = re.escape(
    """The following inputs to 'f' do not conform with the corresponding type annotation:

a
	Takes type <class 'tuple'> containing items that conform with <tuple[list[set[typing.Union[int, float]]], ...]>, although the received container contains item '[{0}, {'one'}]' of type <class 'list'>."""
)


def test_nested_containers():
    """Test contents of subscripted nested containers are validated."""

    @m.parse
    def f(
        a: tuple[list[set[Union[int, float]]], ...],
    ) -> tuple[list[set[Union[int, float]]], ...]:
        return a

    assert f(([{0}, {1.0}],)) == ([{0}, {1.0}],)

    # invalid as inner list contains an invalid item
    regex = re.compile("^" + INVALID_MSG_NEATED_ITEMS_0 + "$")
    with pytest.raises(m.InputsError, match=regex):
        f(([{0}, {1.0}, "not a set"],))

    # invalid as a set in second level of nesting contains an invalid object
    regex = re.compile("^" + INVALID_MSG_NEATED_ITEMS_1 + "$")
    with pytest.raises(m.InputsError, match=regex):
        f(([{0}, {"one"}],))

    @m.parse
    def f_no_validation(
        a: Annotated[tuple[list[set[int, float]], ...], m.NO_ITEM_VALIDATION],
    ) -> tuple[list[set[int, float]], ...]:
        return a

    rtrn = f_no_validation(([{0}, {"one"}, "not a set"],))
    assert rtrn == ([{0}, {"one"}, "not a set"],)
    assert f_no_validation(("any 'ol tuple", 1, 2.0)) == ("any 'ol tuple", 1, 2.0)


LIT = "spam"


def test_strict_literal():
    @m.parse
    def f(
        a: Annotated[Literal[LIT], m.STRICT_LITERAL],
        b: Literal[LIT],
    ):
        return

    assert f(LIT, LIT) is None
    same_val_diff_obj = "spamX"[:-1]
    assert f(LIT, same_val_diff_obj) is None

    # verify error message, including that parameter b does not appear
    msg = (
        "The following inputs to 'f' do not conform with the corresponding type"
        " annotation:\n\na\n\tTakes a literal from <('spam',)> although received"
        " 'spam'."
    )
    pat = f"^{re.escape(msg)}$"  # ensure nothing either side of expected msg
    with pytest.raises(m.InputsError, match=re.compile(pat, re.DOTALL)):
        f(same_val_diff_obj, LIT)

    msg = re.escape("b\n\tTakes a value from <('spam',)> although received 'notspam'.")
    with pytest.raises(m.InputsError, match=msg):
        f(LIT, "notspam")


# NOTE: can be removed from when min supported python version advances to 3.11
def test_fix_hints_for_none_default():
    if sys.version_info.minor >= 11:
        pytest.skip("only applicable to py <11.")

    def f(
        a: Union[int, str],
        b: Optional[str],
        c: Optional[Union[int, str]],
        d: Annotated[Union[str, int, float], "some_meta"],
        e: Annotated[Optional[Union[str, int, float]], "some_meta"],
        f: Optional[Union[int, str]] = None,
        g: Annotated[Optional[str], "some_meta"] = None,
        h: Annotated[Optional[Union[str, int, float]], "some_meta"] = None,
        *,
        i: Annotated[Optional[Union[str, int, float]], "some_meta"] = None,
    ):
        return

    spec = inspect.getfullargspec(f)

    # hints as would be returned by py3.9 and py3.10
    # typing.get_type_hints(f, include_extras=True)
    # 'f' and 'h' wrapped with optional. Others ensure fix does not alter
    # expected behavior for other annotations.
    hints = {
        "a": Union[int, str],
        "b": Optional[str],
        "c": Optional[Union[int, str]],
        "d": Annotated[Union[str, int, float], "some_meta"],
        "e": Annotated[Optional[Union[str, int, float]], "some_meta"],
        "f": Optional[Union[int, str]],
        "g": Optional[Annotated[Optional[str], "some_meta"]],
        "h": Optional[Annotated[Optional[Union[str, int, float]], "some_meta"]],
        "i": Annotated[Optional[Union[str, int, float]], "some_meta"],
    }

    rtrn = m.fix_hints_for_none_default(hints, spec)

    NoneType = type(None)
    expected = {
        "a": Union[int, str],
        "b": Optional[str],
        "c": Union[int, str, NoneType],
        "d": Annotated[Union[str, int, float], "some_meta"],
        "e": Annotated[Union[str, int, float, NoneType], "some_meta"],
        "f": Union[int, str, NoneType],
        "g": Annotated[Optional[str], "some_meta"],  # unwrapped
        "h": Annotated[Union[str, int, float, NoneType], "some_meta"],  # unwrapped
        "i": Annotated[Union[str, int, float, NoneType], "some_meta"],
    }
    assert rtrn == expected

    # validate as expected when default not None
    def f(
        f: Optional[Union[int, str]] = 1,
        g: Annotated[Optional[str], "some_meta"] = "foo",
        h: Annotated[Optional[Union[str, int, float]], "some_meta"] = 2,
        *,
        i: Annotated[Optional[Union[str, int, float]], "some_meta"] = 3,
    ):
        return

    spec = inspect.getfullargspec(f)
    hints = get_type_hints(f, include_extras=True)
    rtrn = m.fix_hints_for_none_default(hints, spec)

    expected = {k: v for k, v in expected.items() if k in hints}
    assert rtrn == expected

    # validate as expected when no default values
    def f(
        f: Optional[Union[int, str]],
        g: Annotated[Optional[str], "some_meta"],
        h: Annotated[Optional[Union[str, int, float]], "some_meta"],
        *,
        i: Annotated[Optional[Union[str, int, float]], "some_meta"],
    ):
        return

    spec = inspect.getfullargspec(f)
    hints = get_type_hints(f, include_extras=True)
    rtrn = m.fix_hints_for_none_default(hints, spec)
    assert rtrn == expected

    # validate as expected when no arg default values but kwarg defaults
    def f(
        f: Optional[Union[int, str]],
        g: Annotated[Optional[str], "some_meta"],
        h: Annotated[Optional[Union[str, int, float]], "some_meta"],
        *,
        i: Annotated[Optional[Union[str, int, float]], "some_meta"] = None,
    ):
        return

    spec = inspect.getfullargspec(f)
    hints = get_type_hints(f, include_extras=True)
    rtrn = m.fix_hints_for_none_default(hints, spec)
    assert rtrn == expected

    # validate as expected when no kwonly params

    def f(
        f: Optional[Union[int, str]],
        g: Annotated[Optional[str], "some_meta"],
        h: Annotated[Optional[Union[str, int, float]], "some_meta"] = None,
    ):
        return

    spec = inspect.getfullargspec(f)
    hints = get_type_hints(f, include_extras=True)
    rtrn = m.fix_hints_for_none_default(hints, spec)
    expected.pop("i")
    assert rtrn == expected


def test_coerce_none():
    """Test None input is not coerced."""

    @m.parse
    def f(
        a: Annotated[Optional[int], m.Coerce(str)],
        b: Annotated[Optional[int], m.Coerce(str)],
        c: Annotated[Union[int, str, None], m.Coerce(str)],
        b1: Annotated[Optional[int], m.Coerce(str)] = None,
        c1: Annotated[Union[int, str, None], m.Coerce(str)] = None,
    ) -> dict[str, Any]:
        return dict(a=a, b=b, c=c, b1=b1, c1=c1)

    assert f(0, 1, 2, 3, 4) == {"a": "0", "b": "1", "c": "2", "b1": "3", "c1": "4"}
    expected = {
        "a": "0",
        "b": None,
        "c": None,
        "b1": None,
        "c1": None,
    }
    assert f(0, None, None) == expected
    assert f(0, None, None, None, None) == expected


def parse_b(name: str, obj: int, params: dict[str, Any]) -> int:
    return obj + 1


def parse_c(name: str, obj: Optional[int], params: dict[str, Any]) -> int:
    """Dynamically define default value for parameter c."""
    return params["a"] if obj is None else obj


def test_parse_none():
    """Test effect of parse_none.

    Includes verifying dynamically definition of default values.
    """

    @m.parse
    def f(
        a: int,
        b: Annotated[Optional[int], m.Parser(parse_b, parse_none=False)],
        b2: Annotated[Optional[int], m.Parser(parse_b, parse_none=False)] = None,
        c: Annotated[Optional[int], m.Parser(parse_c)] = None,
    ) -> dict[str, Any]:
        return dict(a=a, b=b, b2=b2, c=c)

    expected = {"a": 7, "b": 11, "b2": 21, "c": 30}
    assert f(7, 10, 20, 30) == expected

    # verify that parse_none results in None being passed through (b and b2)
    # verify that None provides for setting default values dynamically (c)
    assert f(3, None) == {"a": 3, "b": None, "b2": None, "c": 3}
