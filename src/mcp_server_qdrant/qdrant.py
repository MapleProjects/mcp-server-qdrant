import logging
import uuid
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]

DOCUMENT_PATH = "document"


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None
    # The Qdrant point id. It is populated when an entry is read back via
    # `search`, `get`, `update` or `delete`, and is the handle required to
    # modify an existing entry.
    id: str | None = None
    # Cosine similarity of the entry against the query. Only set by `search`
    # and by query-based `delete`.
    score: float | None = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url,
            api_key=qdrant_api_key,
            path=qdrant_local_path,
            check_compatibility=False,
        )
        self._field_indexes = field_indexes

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    def _resolve_collection(self, collection_name: str | None) -> str:
        resolved = collection_name or self._default_collection_name
        if not resolved:
            raise ValueError(
                "No collection name provided and no default collection configured."
            )
        return resolved

    @staticmethod
    def _entry_from_point(point: models.Record, score: float | None = None) -> Entry:
        payload = point.payload or {}
        return Entry(
            content=payload.get(DOCUMENT_PATH, ""),
            metadata=payload.get(METADATA_PATH),
            id=str(point.id),
            score=score,
        )

    async def store(self, entry: Entry, *, collection_name: str | None = None) -> Entry:
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        :return: The stored entry, including the generated point id.
        """
        if not entry.content or not entry.content.strip():
            raise ValueError("Cannot store an empty document.")

        collection_name = self._resolve_collection(collection_name)
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        point_id = uuid.uuid4().hex
        payload = {DOCUMENT_PATH: entry.content, METADATA_PATH: entry.metadata}
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={vector_name: embeddings[0]},
                    payload=payload,
                )
            ],
        )
        return Entry(content=entry.content, metadata=entry.metadata, id=point_id)

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        score_threshold: float | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.
        :param score_threshold: Minimum similarity score for a result to be returned, optional.

        :return: A list of entries found, each carrying its id and similarity score.
        """
        collection_name = self._resolve_collection(collection_name)
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        return [
            self._entry_from_point(result, score=result.score)
            for result in search_results.points
        ]

    async def get(
        self, point_id: str, *, collection_name: str | None = None
    ) -> Entry | None:
        """
        Read a single entry by its point id.
        :param point_id: The id of the point to read.
        :param collection_name: The name of the collection to read from, optional. If not provided,
                                the default collection is used.
        :return: The entry, or None when the collection or the point does not exist.
        """
        collection_name = self._resolve_collection(collection_name)
        if not await self._client.collection_exists(collection_name):
            return None

        points = await self._retrieve(collection_name, point_id)
        if not points:
            return None
        return self._entry_from_point(points[0])

    @staticmethod
    def _id_candidates(point_id: str | int) -> list[str]:
        """Return the accepted spellings of a point id, most specific first.

        Qdrant accepts an unsigned integer or a UUID, and the same UUID can be
        written with or without dashes. The remote server normalizes both forms,
        local mode matches the stored string exactly, so every spelling is tried
        before concluding that the point does not exist.
        """
        if isinstance(point_id, int):
            if point_id < 0:
                raise ValueError(f"'{point_id}' is not a valid point id: negative.")
            return [str(point_id)]

        candidate = str(point_id).strip()
        if candidate.isdigit():
            return [candidate]

        try:
            parsed = uuid.UUID(candidate)
        except (ValueError, AttributeError) as exc:
            raise ValueError(
                f"'{point_id}' is not a valid point id: expected a UUID or an "
                "unsigned integer."
            ) from exc

        candidates: list[str] = []
        for form in (candidate, parsed.hex, str(parsed)):
            if form not in candidates:
                candidates.append(form)
        return candidates

    async def _retrieve(
        self, collection_name: str, point_id: str | int
    ) -> list[models.Record]:
        for candidate in self._id_candidates(point_id):
            points = await self._client.retrieve(
                collection_name=collection_name,
                ids=[candidate],
                with_payload=True,
                with_vectors=False,
            )
            if points:
                return points
        return []

    async def update(
        self,
        point_id: str,
        *,
        content: str | None = None,
        metadata: Metadata | None = None,
        clear_metadata: bool = False,
        collection_name: str | None = None,
    ) -> Entry:
        """
        Update an existing entry in the Qdrant collection in place, identified by its point id.

        Only the fields that are provided are modified; anything left as ``None`` is untouched.
        A ``content`` change re-embeds the document and rewrites the point in a single upsert,
        so the vector and the payload can never diverge. A ``metadata`` change rewrites the
        payload without touching the vector, so it is cheap and never triggers a re-embedding.

        :param point_id: The id of the point to update. Must refer to an existing point.
        :param content: The new content to store. If provided, the document is re-embedded.
        :param metadata: The new metadata. If provided, it replaces the existing metadata.
        :param clear_metadata: Remove the metadata key entirely. Mutually exclusive with ``metadata``.
        :param collection_name: The name of the collection to update in, optional. If not provided,
                                the default collection is used.
        :return: The updated entry.
        :raises ValueError: If nothing is provided to update, if the collection does not
                            exist, or if no point with the given id exists.
        """
        if content is None and metadata is None and not clear_metadata:
            raise ValueError(
                "Nothing to update: provide 'content', 'metadata', or 'clear_metadata'."
            )
        if clear_metadata and metadata is not None:
            raise ValueError("Provide either 'metadata' or 'clear_metadata', not both.")
        if content is not None and not content.strip():
            raise ValueError("Cannot set an empty document.")

        collection_name = self._resolve_collection(collection_name)

        if not await self._client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")

        existing = await self._retrieve(collection_name, point_id)
        if not existing:
            raise ValueError(
                f"No entry with id '{point_id}' found in collection '{collection_name}'."
            )

        # Use the id exactly as Qdrant stored it, so the write targets the same
        # key regardless of how the caller spelled the UUID.
        point_id = str(existing[0].id)
        payload = dict(existing[0].payload or {})

        if content is not None:
            payload[DOCUMENT_PATH] = content
        if clear_metadata:
            payload.pop(METADATA_PATH, None)
        elif metadata is not None:
            payload[METADATA_PATH] = metadata

        if content is not None:
            # Single atomic write: vector and payload land together.
            embeddings = await self._embedding_provider.embed_documents([content])
            vector_name = self._embedding_provider.get_vector_name()
            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector={vector_name: embeddings[0]},
                        payload=payload,
                    )
                ],
            )
        else:
            # Metadata-only change: rewrite the payload, leave the vector alone.
            await self._client.overwrite_payload(
                collection_name=collection_name,
                payload=payload,
                points=[point_id],
            )

        return Entry(
            content=payload.get(DOCUMENT_PATH, ""),
            metadata=payload.get(METADATA_PATH),
            id=str(point_id),
        )

    async def delete(
        self,
        query: str | None = None,
        *,
        point_id: str | None = None,
        collection_name: str | None = None,
        limit: int = 1,
        score_threshold: float | None = None,
        query_filter: models.Filter | None = None,
        dry_run: bool = False,
    ) -> list[Entry]:
        """
        Delete entries either by exact point id or by semantic search.

        Exactly one of ``query`` or ``point_id`` must be provided. Deleting by id is
        deterministic and is the safe path. Deleting by query removes the ``limit``
        closest matches, optionally restricted by ``score_threshold`` and
        ``query_filter``; use ``dry_run`` to preview the matches without removing them.

        :param query: Semantic query used to find the entries to delete.
        :param point_id: Exact id of the entry to delete.
        :param collection_name: The name of the collection to delete from, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to delete when deleting by query.
        :param score_threshold: Minimum similarity score for a match to be deleted.
        :param query_filter: The filter to apply to the query, if any.
        :param dry_run: Return the matches without deleting them.
        :return: The deleted entries, or the matches when ``dry_run`` is set.
        :raises ValueError: If both or neither of ``query`` and ``point_id`` are provided.
        """
        if (query is None) == (point_id is None):
            raise ValueError("Provide exactly one of 'query' or 'point_id' to delete.")

        collection_name = self._resolve_collection(collection_name)
        if not await self._client.collection_exists(collection_name):
            return []

        if point_id is not None:
            points = await self._retrieve(collection_name, point_id)
            entries = [self._entry_from_point(point) for point in points]
        else:
            assert query is not None
            query_vector = await self._embedding_provider.embed_query(query)
            vector_name = self._embedding_provider.get_vector_name()
            results = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )
            entries = [
                self._entry_from_point(result, score=result.score)
                for result in results.points
            ]

        if not entries or dry_run:
            return entries

        point_ids = [entry.id for entry in entries if entry.id is not None]
        await self._client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )
        return entries

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            try:
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        vector_name: models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                )
            except Exception:
                # Another writer may have created it between the check and the
                # create. Only swallow the error if the collection now exists.
                if not await self._client.collection_exists(collection_name):
                    raise

            # Create payload indexes if configured

            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
