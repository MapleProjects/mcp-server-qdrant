import json
import logging
from typing import Annotated, Any, Optional
from xml.sax.saxutils import escape

from fastmcp import Context, FastMCP
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)

# Hard ceiling for a single search or delete call. A client asking for more is
# almost always a mistake, and an unbounded limit can pull the whole collection
# into the response.
MAX_RESULTS = 100


def _clamp_limit(limit: int | None, default: int) -> int:
    if limit is None:
        return default
    if limit < 1:
        raise ValueError("'limit' must be at least 1.")
    return min(limit, MAX_RESULTS)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        entry_id = entry.id or ""
        score = f"<score>{entry.score:.4f}</score>" if entry.score is not None else ""
        return (
            f"<entry><id>{escape(entry_id)}</id>"
            f"<content>{escape(entry.content)}</content>"
            f"<metadata>{escape(entry_metadata)}</metadata>{score}</entry>"
        )

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored, including its id.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            try:
                stored = await self.qdrant_connector.store(
                    entry, collection_name=collection_name
                )
            except ValueError as e:
                return f"Could not store the memory: {e}"

            return (
                f"Remembered: {information} "
                f"(id: {stored.id}, collection: {collection_name})"
            )

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            query_filter: ArbitraryFilter | None = None,
            limit: Annotated[
                int | None,
                Field(
                    description="Maximum number of memories to return. "
                    "Defaults to the server's configured search limit."
                ),
            ] = None,
            score_threshold: Annotated[
                float | None,
                Field(
                    description="Minimum similarity score, between 0 and 1 for cosine "
                    "distance. Results below it are dropped. Optional."
                ),
            ] = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param query_filter: The filter to apply to the query.
            :param limit: Maximum number of results, optional. Falls back to the configured search limit.
            :param score_threshold: Minimum similarity score, optional.
            :return: A list of entries found or None.
            """

            # Log query_filter
            await ctx.debug(f"Query filter: {query_filter}")

            try:
                query_filter = models.Filter(**query_filter) if query_filter else None
                effective_limit = _clamp_limit(limit, self.qdrant_settings.search_limit)
            except (ValueError, TypeError) as e:
                return [f"Invalid search arguments: {e}"]

            await ctx.debug(f"Finding results for query {query}")

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=effective_limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        async def update(
            ctx: Context,
            point_id: Annotated[
                str,
                Field(
                    description="The id of the memory to update, as returned in the "
                    "'id' field by the find tool."
                ),
            ],
            collection_name: Annotated[
                str, Field(description="The collection the memory lives in")
            ],
            # `content` and `metadata` are optional: provide either or both. Anything
            # left out is kept unchanged. They are typed as non-optional with a None
            # default for the same MCP client compatibility reason as `store`'s metadata.
            content: Annotated[
                str | None,
                Field(
                    description="The new content for the memory. If omitted, the "
                    "existing content is kept. Providing it re-embeds the memory."
                ),
            ] = None,
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="The new metadata for the memory. If omitted, the "
                    "existing metadata is kept. If provided, it replaces the existing "
                    "metadata. Any json is accepted."
                ),
            ] = None,
            clear_metadata: Annotated[
                bool,
                Field(
                    description="Remove the metadata entirely. Cannot be combined "
                    "with 'metadata'."
                ),
            ] = False,
        ) -> str:
            """
            Update an existing memory in Qdrant, identified by its point id.
            :param ctx: The context for the request.
            :param point_id: The id of the memory to update (from the find tool).
            :param collection_name: The name of the collection the memory lives in.
            :param content: The new content, optional. If provided, the memory is re-embedded.
            :param metadata: The new metadata, optional. If provided, it replaces the old metadata.
            :param clear_metadata: Remove the metadata key entirely, optional.
            :return: A message indicating that the memory was updated.
            """
            await ctx.debug(f"Updating entry {point_id} in Qdrant")

            try:
                updated = await self.qdrant_connector.update(
                    point_id,
                    content=content,
                    metadata=metadata,
                    clear_metadata=clear_metadata,
                    collection_name=collection_name,
                )
            except ValueError as e:
                return f"Could not update entry {point_id}: {e}"

            return (
                f"Updated entry {updated.id} in collection {collection_name}. "
                f"Current content: {updated.content}"
            )

        async def delete(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to delete from")
            ],
            point_id: Annotated[
                str | None,
                Field(
                    description="Exact id of the memory to delete, as returned in the "
                    "'id' field by the find tool. Use this for a deterministic delete. "
                    "Mutually exclusive with 'query'."
                ),
            ] = None,
            query: Annotated[
                str | None,
                Field(
                    description="Semantic search query used to find the memories to "
                    "delete. Mutually exclusive with 'point_id'. Prefer 'point_id' "
                    "when you already know it."
                ),
            ] = None,
            limit: Annotated[
                int | None,
                Field(
                    description="Maximum number of memories to delete when deleting by "
                    "query. Defaults to 1."
                ),
            ] = None,
            score_threshold: Annotated[
                float | None,
                Field(
                    description="Minimum similarity score for a match to be deleted. "
                    "Use it to avoid deleting an unrelated memory. Optional."
                ),
            ] = None,
            dry_run: Annotated[
                bool,
                Field(
                    description="Preview the matches without deleting anything. "
                    "Recommended before a query-based delete."
                ),
            ] = False,
            query_filter: ArbitraryFilter | None = None,
        ) -> str:
            """
            Delete a memory from Qdrant, either by exact id or by semantic search.
            :param ctx: The context for the request.
            :param collection_name: The name of the collection to delete from.
            :param point_id: Exact id of the memory to delete, optional.
            :param query: Semantic query to find the memory to delete, optional.
            :param limit: Maximum number of memories to delete by query, optional.
            :param score_threshold: Minimum similarity score to delete, optional.
            :param dry_run: Preview without deleting, optional.
            :param query_filter: The filter to apply to the query, optional.
            :return: A message indicating what was deleted, or what would be deleted.
            """
            await ctx.debug(f"Deleting memory (point_id={point_id}, query={query})")

            try:
                effective_limit = _clamp_limit(limit, 1)
                parsed_filter = models.Filter(**query_filter) if query_filter else None
                deleted = await self.qdrant_connector.delete(
                    query,
                    point_id=point_id,
                    collection_name=collection_name,
                    limit=effective_limit,
                    score_threshold=score_threshold,
                    query_filter=parsed_filter,
                    dry_run=dry_run,
                )
            except (ValueError, TypeError) as e:
                return f"Could not delete: {e}"

            if not deleted:
                target = point_id if point_id is not None else query
                return f"No matching memory found for: {target}"

            verb = "Would delete" if dry_run else "Deleted"
            lines = [f"{verb} {len(deleted)} memory(ies):"]
            for entry in deleted:
                score = f" (score {entry.score:.4f})" if entry.score is not None else ""
                preview = entry.content[:100]
                if len(entry.content) > 100:
                    preview += "..."
                lines.append(f"- id {entry.id}{score}: {preview}")
            return "\n".join(lines)

        find_foo = find
        store_foo = store
        update_foo = update
        delete_foo = delete

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
            delete_foo = wrap_filters(delete_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})
            delete_foo = make_partial_function(delete_foo, {"query_filter": None})

        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            update_foo = make_partial_function(
                update_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            delete_foo = make_partial_function(
                delete_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )
            self.tool(
                update_foo,
                name="qdrant-update",
                description=self.tool_settings.tool_update_description,
            )
            self.tool(
                delete_foo,
                name="qdrant-delete",
                description=self.tool_settings.tool_delete_description,
            )
