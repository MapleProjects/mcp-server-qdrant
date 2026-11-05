import inspect
from functools import wraps
from typing import Any, Callable


def make_partial_function(
    original_func: Callable, fixed_values: dict[str, Any]
) -> Callable:
    """Bind a subset of a callable's parameters and keep the rest visible to MCP.

    The returned wrapper exposes a signature containing only the parameters that
    were not fixed, so MCP clients build a schema with exactly the arguments the
    caller is allowed to send. Fixed parameters are injected on every call and
    cannot be overridden by the client.

    The wrapper keeps the async/sync nature of the original callable. A sync
    wrapper around a coroutine function makes FastMCP run the call in a worker
    thread and receive a coroutine object instead of a result, which breaks
    context propagation and serialization.
    """
    sig = inspect.signature(original_func)

    unknown = set(fixed_values) - set(sig.parameters)
    if unknown:
        raise ValueError(
            f"Cannot bind unknown parameters {sorted(unknown)} on "
            f"{getattr(original_func, '__name__', original_func)!r}"
        )

    remaining_params = [name for name in sig.parameters if name not in fixed_values]
    new_params = [sig.parameters[name] for name in remaining_params]
    new_signature = sig.replace(parameters=new_params)

    def _merge(args: tuple, kwargs: dict) -> dict[str, Any]:
        bound: dict[str, Any] = dict(fixed_values)
        for name, value in zip(remaining_params, args):
            bound[name] = value
        for name, value in kwargs.items():
            if name in fixed_values:
                raise TypeError(
                    f"{name!r} is fixed by the server and cannot be overridden"
                )
            bound[name] = value
        return bound

    if inspect.iscoroutinefunction(original_func):

        async def _async_invoke(*args, **kwargs):
            return await original_func(**_merge(args, kwargs))

        wrapper = wraps(original_func)(_async_invoke)

    else:

        def _sync_invoke(*args, **kwargs):
            return original_func(**_merge(args, kwargs))

        wrapper = wraps(original_func)(_sync_invoke)

    # `functools.wraps` copied the original signature and annotations, which
    # still contain the parameters that were just fixed. Rebuild both from the
    # reduced signature so MCP clients see every remaining parameter and none
    # of the fixed ones.
    wrapper.__signature__ = new_signature  # type: ignore[attr-defined]
    wrapper.__annotations__ = {
        param.name: param.annotation
        for param in new_params
        if param.annotation is not inspect.Parameter.empty
    }
    if sig.return_annotation is not inspect.Signature.empty:
        wrapper.__annotations__["return"] = sig.return_annotation

    positional_defaults = [
        param.default
        for param in new_params
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and param.default is not inspect.Parameter.empty
    ]
    wrapper.__defaults__ = tuple(positional_defaults) or None  # type: ignore[attr-defined]

    keyword_defaults = {
        param.name: param.default
        for param in new_params
        if param.kind is inspect.Parameter.KEYWORD_ONLY
        and param.default is not inspect.Parameter.empty
    }
    wrapper.__kwdefaults__ = keyword_defaults or None  # type: ignore[attr-defined]

    return wrapper
