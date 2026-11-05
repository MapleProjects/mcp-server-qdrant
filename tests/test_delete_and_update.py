import uuid

import pytest

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import Entry, QdrantConnector


@pytest.fixture
async def embedding_provider():
    return FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
async def qdrant_connector(embedding_provider):
    collection_name = f"test_collection_{uuid.uuid4().hex}"
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )
    yield connector


@pytest.mark.asyncio
async def test_store_returns_point_id(qdrant_connector):
    stored = await qdrant_connector.store(Entry(content="An entry with an id"))

    assert stored.id
    assert stored.content == "An entry with an id"

    fetched = await qdrant_connector.get(stored.id)
    assert fetched is not None
    assert fetched.content == "An entry with an id"


@pytest.mark.asyncio
async def test_store_rejects_empty_document(qdrant_connector):
    with pytest.raises(ValueError, match="empty document"):
        await qdrant_connector.store(Entry(content="   "))


@pytest.mark.asyncio
async def test_search_returns_scores(qdrant_connector):
    await qdrant_connector.store(Entry(content="Python is a programming language"))

    [result] = await qdrant_connector.search("Python programming")
    assert result.score is not None
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_search_score_threshold_filters_results(qdrant_connector):
    await qdrant_connector.store(Entry(content="Python is a programming language"))

    results = await qdrant_connector.search(
        "completely unrelated topic about cooking", score_threshold=0.99
    )
    assert results == []


@pytest.mark.asyncio
async def test_get_unknown_id_returns_none(qdrant_connector):
    await qdrant_connector.store(Entry(content="Seed"))

    assert await qdrant_connector.get(uuid.uuid4().hex) is None


@pytest.mark.asyncio
async def test_get_invalid_id_raises_value_error(qdrant_connector):
    await qdrant_connector.store(Entry(content="Seed"))

    with pytest.raises(ValueError, match="not a valid point id"):
        await qdrant_connector.get("not-a-uuid")


@pytest.mark.asyncio
async def test_delete_by_point_id(qdrant_connector):
    first = await qdrant_connector.store(Entry(content="First memory"))
    await qdrant_connector.store(Entry(content="Second memory"))

    deleted = await qdrant_connector.delete(point_id=first.id)

    assert len(deleted) == 1
    assert deleted[0].id == first.id
    assert deleted[0].content == "First memory"

    remaining = await qdrant_connector.search("memory", limit=10)
    assert len(remaining) == 1
    assert remaining[0].content == "Second memory"


@pytest.mark.asyncio
async def test_delete_by_point_id_unknown_returns_empty(qdrant_connector):
    await qdrant_connector.store(Entry(content="Seed"))

    assert await qdrant_connector.delete(point_id=uuid.uuid4().hex) == []


@pytest.mark.asyncio
async def test_delete_requires_exactly_one_selector(qdrant_connector):
    with pytest.raises(ValueError, match="exactly one"):
        await qdrant_connector.delete()

    with pytest.raises(ValueError, match="exactly one"):
        await qdrant_connector.delete("query", point_id=uuid.uuid4().hex)


@pytest.mark.asyncio
async def test_delete_dry_run_keeps_entries(qdrant_connector):
    await qdrant_connector.store(Entry(content="Python is a programming language"))

    preview = await qdrant_connector.delete("Python programming", dry_run=True)

    assert len(preview) == 1
    assert preview[0].score is not None

    still_there = await qdrant_connector.search("Python programming")
    assert len(still_there) == 1


@pytest.mark.asyncio
async def test_delete_by_query_respects_limit(qdrant_connector):
    await qdrant_connector.store(Entry(content="Python is a programming language"))
    await qdrant_connector.store(Entry(content="Python is also a snake"))
    await qdrant_connector.store(Entry(content="The Eiffel Tower is in Paris"))

    deleted = await qdrant_connector.delete("Python", limit=2)

    assert len(deleted) == 2
    remaining = await qdrant_connector.search("anything", limit=10)
    assert len(remaining) == 1
    assert "Eiffel" in remaining[0].content


@pytest.mark.asyncio
async def test_delete_by_query_respects_score_threshold(qdrant_connector):
    await qdrant_connector.store(Entry(content="Python is a programming language"))

    deleted = await qdrant_connector.delete(
        "completely unrelated cooking topic", score_threshold=0.99
    )

    assert deleted == []
    assert len(await qdrant_connector.search("Python", limit=10)) == 1


@pytest.mark.asyncio
async def test_update_returns_updated_entry(qdrant_connector):
    stored = await qdrant_connector.store(
        Entry(content="Original content", metadata={"status": "draft"})
    )

    updated = await qdrant_connector.update(
        stored.id, content="Revised content", metadata={"status": "final"}
    )

    assert updated.id == stored.id
    assert updated.content == "Revised content"
    assert updated.metadata == {"status": "final"}

    fetched = await qdrant_connector.get(stored.id)
    assert fetched.content == "Revised content"
    assert fetched.metadata == {"status": "final"}


@pytest.mark.asyncio
async def test_update_clear_metadata(qdrant_connector):
    stored = await qdrant_connector.store(
        Entry(content="Keep me", metadata={"status": "draft"})
    )

    updated = await qdrant_connector.update(stored.id, clear_metadata=True)

    assert updated.metadata is None
    assert updated.content == "Keep me"


@pytest.mark.asyncio
async def test_update_rejects_metadata_and_clear_together(qdrant_connector):
    stored = await qdrant_connector.store(Entry(content="Seed"))

    with pytest.raises(ValueError, match="not both"):
        await qdrant_connector.update(stored.id, metadata={"a": 1}, clear_metadata=True)


@pytest.mark.asyncio
async def test_update_rejects_empty_content(qdrant_connector):
    stored = await qdrant_connector.store(Entry(content="Seed"))

    with pytest.raises(ValueError, match="empty document"):
        await qdrant_connector.update(stored.id, content="  ")


@pytest.mark.asyncio
async def test_update_keeps_unknown_payload_keys(qdrant_connector):
    stored = await qdrant_connector.store(Entry(content="Seed"))

    # Simulate a payload written by another tool.
    await qdrant_connector._client.set_payload(
        collection_name=qdrant_connector._default_collection_name,
        payload={"extra": "kept"},
        points=[stored.id],
    )

    await qdrant_connector.update(stored.id, content="New body")

    points = await qdrant_connector._client.retrieve(
        collection_name=qdrant_connector._default_collection_name,
        ids=[stored.id],
        with_payload=True,
    )
    assert points[0].payload["extra"] == "kept"
    assert points[0].payload["document"] == "New body"
