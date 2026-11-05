from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Keep the memory for later use, when you are asked to remember something."
)
DEFAULT_TOOL_DELETE_DESCRIPTION = (
    "Delete a memory from Qdrant. Two modes, exactly one per call: \n"
    " - 'point_id': delete the exact memory with that id, as returned by the find "
    "tool. Deterministic and preferred when the id is known. \n"
    " - 'query': delete the closest semantic matches. Use 'limit' to cap how many, "
    "'score_threshold' to avoid deleting an unrelated memory, and 'dry_run' to "
    "preview the matches before removing them. \n"
    "The response lists the id, the score and a preview of every deleted memory."
)
DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Look up memories in Qdrant. Use this tool when you need to: \n"
    " - Find memories by their content \n"
    " - Access memories for further analysis \n"
    " - Get some personal information about the user"
)
DEFAULT_TOOL_UPDATE_DESCRIPTION = (
    "Update a memory that is already stored in Qdrant. Use this tool when you need to "
    "correct, extend, or replace the content or metadata of an existing memory. \n"
    "First use the find tool to locate the memory and read its 'id', then call this tool "
    "with that id. You can update the content, the metadata, or both; anything you leave "
    "out is kept unchanged."
)
DEFAULT_TOOL_GET_DESCRIPTION = (
    "Read one memory by its exact id, without a semantic search. Use it to confirm what "
    "a memory contains before updating or deleting it, or to re-read a memory whose id "
    "you already have."
)
DEFAULT_TOOL_LIST_DESCRIPTION = (
    "Page through the stored memories in insertion order, without a semantic query. Use "
    "it to audit or clean up the collection. Pass the returned 'next_offset' back as "
    "'offset' to read the following page."
)

METADATA_PATH = "metadata"


class ToolSettings(BaseSettings):
    """
    Configuration for all the tools.
    """

    tool_store_description: str = Field(
        default=DEFAULT_TOOL_STORE_DESCRIPTION,
        validation_alias="TOOL_STORE_DESCRIPTION",
    )
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION",
    )
    tool_update_description: str = Field(
        default=DEFAULT_TOOL_UPDATE_DESCRIPTION,
        validation_alias="TOOL_UPDATE_DESCRIPTION",
    )
    tool_delete_description: str = Field(
        default=DEFAULT_TOOL_DELETE_DESCRIPTION,
        validation_alias="TOOL_DELETE_DESCRIPTION",
    )
    tool_get_description: str = Field(
        default=DEFAULT_TOOL_GET_DESCRIPTION,
        validation_alias="TOOL_GET_DESCRIPTION",
    )
    tool_list_description: str = Field(
        default=DEFAULT_TOOL_LIST_DESCRIPTION,
        validation_alias="TOOL_LIST_DESCRIPTION",
    )


class EmbeddingProviderSettings(BaseSettings):
    """
    Configuration for the embedding provider.
    """

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.FASTEMBED,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )


class FilterableField(BaseModel):
    name: str = Field(description="The name of the field payload field to filter on")
    description: str = Field(
        description="A description for the field used in the tool description"
    )
    field_type: Literal["keyword", "integer", "float", "boolean"] = Field(
        description="The type of the field"
    )
    condition: Literal["==", "!=", ">", ">=", "<", "<=", "any", "except"] | None = (
        Field(
            default=None,
            description=(
                "The condition to use for the filter. If not provided, the field will be indexed, but no "
                "filter argument will be exposed to MCP tool."
            ),
        )
    )
    required: bool = Field(
        default=False,
        description="Whether the field is required for the filter.",
    )


class QdrantSettings(BaseSettings):
    """
    Configuration for the Qdrant connector.
    """

    location: str | None = Field(default=None, validation_alias="QDRANT_URL")
    api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: str | None = Field(
        default=None, validation_alias="COLLECTION_NAME"
    )
    local_path: str | None = Field(default=None, validation_alias="QDRANT_LOCAL_PATH")
    search_limit: int = Field(default=10, validation_alias="QDRANT_SEARCH_LIMIT")
    read_only: bool = Field(default=False, validation_alias="QDRANT_READ_ONLY")
    dedupe: bool = Field(default=True, validation_alias="QDRANT_DEDUPE")
    retries: int = Field(default=3, validation_alias="QDRANT_RETRIES")

    filterable_fields: list[FilterableField] | None = Field(default=None)

    allow_arbitrary_filter: bool = Field(
        default=False, validation_alias="QDRANT_ALLOW_ARBITRARY_FILTER"
    )

    def filterable_fields_dict(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {field.name: field for field in self.filterable_fields}

    def filterable_fields_dict_with_conditions(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {
            field.name: field
            for field in self.filterable_fields
            if field.condition is not None
        }

    @model_validator(mode="after")
    def check_local_path_conflict(self) -> "QdrantSettings":
        if self.local_path:
            if self.location is not None or self.api_key is not None:
                raise ValueError(
                    "If 'local_path' is set, 'location' and 'api_key' must be None."
                )
        return self
