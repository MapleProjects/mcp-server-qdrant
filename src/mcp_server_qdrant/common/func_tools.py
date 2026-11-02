import inspect
from functools import wraps
from typing import Callable


def make_partial_function(original_func: Callable, fixed_values: dict) -> Callable:
    sig = inspect.signature(original_func)

    @wraps(original_func)
    def wrapper(*args, **kwargs):
        # Start with fixed values
        bound_args = dict(fixed_values)

        # Bind positional/keyword args from caller
        for name, value in zip(remaining_params, args):
            bound_args[name] = value
        bound_args.update(kwargs)

        return original_func(**bound_args)

    # Only keep parameters NOT in fixed_values
    remaining_params = [name for name in sig.parameters if name not in fixed_values]
    new_params = [sig.parameters[name] for name in remaining_params]

    # Set the new __signature__ for introspection
    wrapper.__signature__ = sig.replace(parameters=new_params)  # type:ignore

    # functools.wraps copied the *original* annotations, which still contain the
    # parameters we just fixed. Rebuild them from the new signature so that MCP
    # clients see (and validate against) every remaining parameter, including
    # ones the wrapper introduces.
    wrapper.__annotations__ = {  # type:ignore
        param.name: param.annotation
        for param in new_params
        if param.annotation is not inspect.Parameter.empty
    }
    if sig.return_annotation is not inspect.Signature.empty:
        wrapper.__annotations__["return"] = sig.return_annotation

    return wrapper
