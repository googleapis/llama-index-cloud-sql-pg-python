# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Type

from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from .async_vector_store import AsyncPostgresVectorStore
from .engine import PostgresEngine
from .indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    BaseIndex,
    DistanceStrategy,
    QueryOptions,
)


class PostgresVectorStore(BasePydanticVectorStore):
    """Google Cloud SQL Vector Store class"""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: PostgresEngine,
        vs: AsyncPostgresVectorStore,
        stores_text: bool = True,
        is_embedding_query: bool = True,
    ):
        """PostgresVectorStore constructor.
        Args:
            key (object): Prevent direct constructor usage.
            engine (PostgresEngine): Connection pool engine for managing connections to Postgres database.
            vs (AsyncPostgresVectorStore): The async only Vector Store implementation
            stores_text (bool): Whether the table stores text. Defaults to "True".
            is_embedding_query (bool): Whether the table query can have embeddings. Defaults to "True".

        Raises:
            Exception: If called directly by user.
        """
        if key != PostgresVectorStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )

        # Delegate to Pydantic's __init__
        super().__init__(stores_text=stores_text, is_embedding_query=is_embedding_query)

        self._engine = engine
        self.__vs = vs

    @classmethod
    async def create(
        cls: Type[PostgresVectorStore],
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
        id_column: str = "node_id",
        text_column: str = "text",
        embedding_column: str = "embedding",
        metadata_json_column: str = "li_metadata",
        metadata_columns: List[str] = [],
        ref_doc_id_column: str = "ref_doc_id",
        node_column: str = "node_data",
        stores_text: bool = True,
        is_embedding_query: bool = True,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        index_query_options: Optional[QueryOptions] = None,
    ) -> PostgresVectorStore:
        """Create an PostgresVectorStore instance and validates the table schema.

        Args:
            engine (PostgresEngine): Postgres Engine for managing connections to postgres database.
            table_name (str): Name of the existing table or the table to be created.
            schema_name (str): Name of the database schema. Defaults to "public".
            id_column (str): Column that represents if of a Node. Defaults to "node_id".
            text_column (str): Column that represent text content of a Node. Defaults to "text".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the content of Node. Defaults to "embedding".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "li_metadata".
            metadata_columns (List[str]): Column(s) that represent extracted metadata keys in their own columns.
            ref_doc_id_column (str): Column that represents id of a node's parent document. Defaults to "ref_doc_id".
            node_column (str): Column that represents the whole JSON node. Defaults to "node_data".
            stores_text (bool): Whether the table stores text. Defaults to "True".
            is_embedding_query (bool): Whether the table query can have embeddings. Defaults to "True".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            index_query_options (QueryOptions): Index query option.

        Raises:
            Exception: If table does not exist or follow the provided structure.

        Returns:
            PostgresVectorStore
        """
        coro = AsyncPostgresVectorStore.create(
            engine,
            table_name,
            schema_name=schema_name,
            id_column=id_column,
            text_column=text_column,
            embedding_column=embedding_column,
            metadata_json_column=metadata_json_column,
            metadata_columns=metadata_columns,
            ref_doc_id_column=ref_doc_id_column,
            node_column=node_column,
            stores_text=stores_text,
            is_embedding_query=is_embedding_query,
            distance_strategy=distance_strategy,
            index_query_options=index_query_options,
        )
        vs = await engine._run_as_async(coro)
        return cls(
            cls.__create_key,
            engine,
            vs,
            stores_text,
            is_embedding_query,
        )

    @classmethod
    def create_sync(
        cls: Type[PostgresVectorStore],
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
        id_column: str = "node_id",
        text_column: str = "text",
        embedding_column: str = "embedding",
        metadata_json_column: str = "li_metadata",
        metadata_columns: List[str] = [],
        ref_doc_id_column: str = "ref_doc_id",
        node_column: str = "node_data",
        stores_text: bool = True,
        is_embedding_query: bool = True,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        index_query_options: Optional[QueryOptions] = None,
    ) -> PostgresVectorStore:
        """Create an PostgresVectorStore instance and validates the table schema.

        Args:
            engine (PostgresEngine): Postgres Engine for managing connections to postgres database.
            table_name (str): Name of the existing table or the table to be created.
            schema_name (str): Name of the database schema. Defaults to "public".
            id_column (str): Column that represents if of a Node. Defaults to "node_id".
            text_column (str): Column that represent text content of a Node. Defaults to "text".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the content of Node. Defaults to "embedding".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "li_metadata".
            metadata_columns (List[str]): Column(s) that represent extracted metadata keys in their own columns.
            ref_doc_id_column (str): Column that represents id of a node's parent document. Defaults to "ref_doc_id".
            node_column (str): Column that represents the whole JSON node. Defaults to "node_data".
            stores_text (bool): Whether the table stores text. Defaults to "True".
            is_embedding_query (bool): Whether the table query can have embeddings. Defaults to "True".
            distance_strategy (DistanceStrategy): Distance strategy to use for vector similarity search. Defaults to COSINE_DISTANCE.
            index_query_options (QueryOptions): Index query option.

        Raises:
            Exception: If table does not exist or follow the provided structure.

        Returns:
            PostgresVectorStore
        """
        coro = AsyncPostgresVectorStore.create(
            engine,
            table_name,
            schema_name=schema_name,
            id_column=id_column,
            text_column=text_column,
            embedding_column=embedding_column,
            metadata_json_column=metadata_json_column,
            metadata_columns=metadata_columns,
            ref_doc_id_column=ref_doc_id_column,
            node_column=node_column,
            stores_text=stores_text,
            is_embedding_query=is_embedding_query,
            distance_strategy=distance_strategy,
            index_query_options=index_query_options,
        )
        vs = engine._run_as_sync(coro)
        return cls(
            cls.__create_key,
            engine,
            vs,
            stores_text,
            is_embedding_query,
        )

    @classmethod
    def class_name(cls) -> str:
        return "PostgresVectorStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._engine

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        """Asynchronously add nodes to the table."""
        return await self._engine._run_as_async(self.__vs.async_add(nodes, **kwargs))

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        """Synchronously add nodes to the table."""
        return self._engine._run_as_sync(self.__vs.async_add(nodes, **add_kwargs))

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Asynchronously delete nodes belonging to provided parent document from the table."""
        await self._engine._run_as_async(self.__vs.adelete(ref_doc_id, **delete_kwargs))

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Synchronously delete nodes belonging to provided parent document from the table."""
        self._engine._run_as_sync(self.__vs.adelete(ref_doc_id, **delete_kwargs))

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Asynchronously delete a set of nodes from the table matching the provided nodes and filters."""
        await self._engine._run_as_async(
            self.__vs.adelete_nodes(node_ids, filters, **delete_kwargs)
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Synchronously delete a set of nodes from the table matching the provided nodes and filters."""
        self._engine._run_as_sync(
            self.__vs.adelete_nodes(node_ids, filters, **delete_kwargs)
        )

    async def aclear(self) -> None:
        """Asynchronously delete all nodes from the table."""
        await self._engine._run_as_async(self.__vs.aclear())

    def clear(self) -> None:
        """Synchronously delete all nodes from the table."""
        return self._engine._run_as_sync(self.__vs.aclear())

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Asynchronously get nodes from the table matching the provided nodes and filters."""
        return await self._engine._run_as_async(self.__vs.aget_nodes(node_ids, filters))

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Asynchronously get nodes from the table matching the provided nodes and filters."""
        return self._engine._run_as_sync(self.__vs.aget_nodes(node_ids, filters))

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Asynchronously query vector store."""
        return await self._engine._run_as_async(self.__vs.aquery(query, **kwargs))

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Synchronously query vector store."""
        return self._engine._run_as_sync(self.__vs.aquery(query, **kwargs))

    async def aset_maintenance_work_mem(
        self, num_leaves: int, vector_size: int
    ) -> None:
        """Set database maintenance work memory (for ScaNN index creation)."""
        await self._engine._run_as_async(
            self.__vs.set_maintenance_work_mem(num_leaves, vector_size)
        )

    def set_maintenance_work_mem(self, num_leaves: int, vector_size: int) -> None:
        """Set database maintenance work memory (for ScaNN index creation)."""
        self._engine._run_as_sync(
            self.__vs.set_maintenance_work_mem(num_leaves, vector_size)
        )

    async def aapply_vector_index(
        self,
        index: BaseIndex,
        name: Optional[str] = None,
        concurrently: bool = False,
    ) -> None:
        """Create an index on the vector store table."""
        return await self._engine._run_as_async(
            self.__vs.aapply_vector_index(index, name, concurrently)
        )

    def apply_vector_index(
        self,
        index: BaseIndex,
        name: Optional[str] = None,
        concurrently: bool = False,
    ) -> None:
        """Create an index on the vector store table."""
        return self._engine._run_as_sync(
            self.__vs.aapply_vector_index(index, name, concurrently)
        )

    async def areindex(self, index_name: Optional[str] = None) -> None:
        """Re-index the vector store table."""
        return await self._engine._run_as_async(self.__vs.areindex(index_name))

    def reindex(self, index_name: Optional[str] = None) -> None:
        """Re-index the vector store table."""
        return self._engine._run_as_sync(self.__vs.areindex(index_name))

    async def adrop_vector_index(
        self,
        index_name: Optional[str] = None,
    ) -> None:
        """Drop the vector index."""
        return await self._engine._run_as_async(
            self.__vs.adrop_vector_index(index_name)
        )

    def drop_vector_index(
        self,
        index_name: Optional[str] = None,
    ) -> None:
        """Drop the vector index."""
        return self._engine._run_as_sync(self.__vs.adrop_vector_index(index_name))

    async def ais_valid_index(
        self,
        index_name: Optional[str] = None,
    ) -> bool:
        """Check if index exists in the table."""
        return await self._engine._run_as_async(self.__vs.is_valid_index(index_name))

    def is_valid_index(
        self,
        index_name: Optional[str] = None,
    ) -> bool:
        """Check if index exists in the table."""
        return self._engine._run_as_sync(self.__vs.is_valid_index(index_name))
