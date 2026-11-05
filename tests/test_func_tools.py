import inspect
from typing import Annotated

import pytest
from pydantic import Field

from mcp_server_qdrant.common.func_tools import make_partial_function


def _build_async():
    async def tool(
        ctx: str,
        point_id: Annotated[str, Field(description="id")],
        collection_name: str,
        content: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        return f"{ctx}|{point_id}|{collection_name}|{content}|{metadata}"

    return tool


def test_partial_keeps_only_remaining_parameters():
    tool = _build_async()
    wrapped = make_partial_function(tool, {"collection_name": "fixed"})

    params = list(inspect.signature(wrapped).parameters)
    assert params == ["ctx", "point_id", "content", "metadata"]
    assert "collection_name" not in params


def test_partial_exposes_defaults():
    tool = _build_async()
    wrapped = make_partial_function(tool, {"collection_name": "fixed"})

    sig = inspect.signature(wrapped)
    assert sig.parameters["content"].default is None
    assert sig.parameters["metadata"].default is None
    assert sig.parameters["point_id"].default is inspect.Parameter.empty


def test_partial_rebuilds_annotations():
    tool = _build_async()
    wrapped = make_partial_function(tool, {"collection_name": "fixed"})

    assert set(wrapped.__annotations__) == {
        "ctx",
        "point_id",
        "content",
        "metadata",
        "return",
    }
    assert "collection_name" not in wrapped.__annotations__


def test_partial_preserves_coroutine_function():
    tool = _build_async()
    wrapped = make_partial_function(tool, {"collection_name": "fixed"})

    assert inspect.iscoroutinefunction(wrapped)


@pytest.mark.asyncio
async def test_partial_injects_fixed_value():
    tool = _build_async()
    wrapped = make_partial_function(tool, {"collection_name": "fixed"})

    result = await wrapped("c", point_id="p1", content="hello")
    assert result == "c|p1|fixed|hello|None"


@pytest.mark.asyncio
async def test_partial_accepts_positional_arguments():
    tool = _build_async()
    wrapped = make_partial_function(tool, {"collection_name": "fixed"})

    result = await wrapped("c", "p2", "body", {"k": "v"})
    assert result == "c|p2|fixed|body|{'k': 'v'}"


@pytest.mark.asyncio
async def test_partial_rejects_overriding_fixed_value():
    tool = _build_async()
    wrapped = make_partial_function(tool, {"collection_name": "fixed"})

    with pytest.raises(TypeError, match="fixed by the server"):
        await wrapped("c", point_id="p1", collection_name="other")


def test_partial_rejects_unknown_fixed_parameter():
    tool = _build_async()

    with pytest.raises(ValueError, match="Cannot bind unknown parameters"):
        make_partial_function(tool, {"nope": 1})


def test_partial_supports_sync_functions():
    def tool(a: int, b: int) -> int:
        return a + b

    wrapped = make_partial_function(tool, {"b": 10})

    assert not inspect.iscoroutinefunction(wrapped)
    assert list(inspect.signature(wrapped).parameters) == ["a"]
    assert wrapped(5) == 15
