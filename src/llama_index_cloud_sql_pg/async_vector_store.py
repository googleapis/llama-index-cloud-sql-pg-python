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

# TODO: Remove below import when minimum supported Python version is 3.10
from __future__ import annotations

import base64
import json
import re
import uuid
import warnings
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np
from llama_index.core.schema import BaseNode, MetadataMode, NodeRelationship, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import (
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from sqlalchemy import RowMapping, text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import PostgresEngine
from .indexes import (
    DEFAULT_DISTANCE_STRATEGY,
    DEFAULT_INDEX_NAME_SUFFIX,
    BaseIndex,
    DistanceStrategy,
    ExactNearestNeighbor,
    QueryOptions,
)


class AsyncPostgresVectorStore(BasePydanticVectorStore):
    """Google Cloud SQL Vector Store class"""

    stores_text: bool = True
    is_embedding_query: bool = True

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AsyncEngine,
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
    ):
        """AsyncPostgresVectorStore constructor.
        Args:
            key (object): Prevent direct constructor usage.
            engine (AsyncEngine): Connection pool engine for managing connections to Cloud SQL database.
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
            Exception: If called directly by user.
        """
        if key != AsyncPostgresVectorStore.__create_key:
            raise Exception("Only create class through 'create' method!")

        # Delegate to Pydantic's __init__
        super().__init__(stores_text=stores_text, is_embedding_query=is_embedding_query)
        self._engine = engine
        self._table_name = table_name
        self._schema_name = schema_name
        self._id_column = id_column
        self._text_column = text_column
        self._embedding_column = embedding_column
        self._metadata_json_column = metadata_json_column
        self._metadata_columns = metadata_columns
        self._ref_doc_id_column = ref_doc_id_column
        self._node_column = node_column
        self._distance_strategy = distance_strategy
        self._index_query_options = index_query_options

    @classmethod
    async def create(
        cls: Type[AsyncPostgresVectorStore],
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
    ) -> AsyncPostgresVectorStore:
        """Create an AsyncPostgresVectorStore instance and validates the table schema.

        Args:
            engine (PostgresEngine): PostgresEngine Engine for managing connections to Cloud SQL PG database.
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
            AsyncPostgresVectorStore
        """
        stmt = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}' AND table_schema = '{schema_name}'"
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(stmt))
            result_map = result.mappings()
            results = result_map.fetchall()
        columns = {}
        for field in results:
            columns[field["column_name"]] = field["data_type"]

        # Check columns
        if id_column not in columns:
            raise ValueError(f"Id column, {id_column}, does not exist.")
        if text_column not in columns:
            raise ValueError(f"Text column, {text_column}, does not exist.")
        text_type = columns[text_column]
        if text_type != "text" and "char" not in text_type:
            raise ValueError(
                f"Text column, {text_column}, is type, {text_type}. It must be a type of character string."
            )
        if embedding_column not in columns:
            raise ValueError(f"Embedding column, {embedding_column}, does not exist.")
        if columns[embedding_column] != "USER-DEFINED":
            raise ValueError(
                f"Embedding column, {embedding_column}, is not type Vector."
            )
        if node_column not in columns:
            raise ValueError(f"Node column, {node_column}, does not exist.")
        if columns[node_column] != "json":
            raise ValueError(f"Node column, {node_column}, is not type JSON.")
        if ref_doc_id_column not in columns:
            raise ValueError(
                f"Reference Document Id column, {ref_doc_id_column}, does not exist."
            )
        if metadata_json_column not in columns:
            raise ValueError(
                f"Metadata column, {metadata_json_column}, does not exist."
            )
        if columns[metadata_json_column] != "jsonb":
            raise ValueError(
                f"Metadata column, {metadata_json_column}, is not type JSONB."
            )
        # If using metadata_columns check to make sure column exists
        for column in metadata_columns:
            if column not in columns:
                raise ValueError(f"Metadata column, {column}, does not exist.")

        return cls(
            cls.__create_key,
            engine._pool,
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

    @classmethod
    def class_name(cls) -> str:
        return "AsyncPostgresVectorStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._engine

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> List[str]:
        """Asynchronously add nodes to the table."""
        ids = []
        metadata_col_names = (
            ", " + ", ".join(self._metadata_columns)
            if len(self._metadata_columns) > 0
            else ""
        )
        metadata_col_values = (
            ", :" + ", :".join(self._metadata_columns)
            if len(self._metadata_columns) > 0
            else ""
        )
        insert_stmt = f"""INSERT INTO "{self._schema_name}"."{self._table_name}"(
            {self._id_column},
            {self._text_column},
            {self._embedding_column},
            {self._metadata_json_column},
            {self._ref_doc_id_column},
            {self._node_column}
            {metadata_col_names}
        ) VALUES (:node_id, :text, :embedding, :li_metadata, :ref_doc_id, :node_data {metadata_col_values})
        """
        node_values_list = []
        for node in nodes:
            metadata = json.dumps(
                node_to_metadata_dict(
                    node, remove_text=self.stores_text, flat_metadata=False
                )
            )
            node_values = {
                "node_id": node.node_id,
                "text": node.get_content(metadata_mode=MetadataMode.NONE),
                "embedding": str(node.get_embedding()),
                "li_metadata": metadata,
                "ref_doc_id": node.ref_doc_id,
                "node_data": node.to_json(),
            }
            for metadata_column in self._metadata_columns:
                if metadata_column in node.metadata:
                    node_values[metadata_column] = node.metadata.get(metadata_column)
                else:
                    node_values[metadata_column] = None
            node_values_list.append(node_values)
            ids.append(node.node_id)
        async with self._engine.connect() as conn:
            await conn.execute(text(insert_stmt), node_values_list)
            await conn.commit()
        return ids

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Asynchronously delete nodes belonging to provided parent document from the table."""
        query = f"""DELETE FROM "{self._schema_name}"."{self._table_name}" WHERE {self._ref_doc_id_column} = '{ref_doc_id}'"""
        async with self._engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def adelete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        """Asynchronously delete a set of nodes from the table matching the provided nodes and filters."""
        if not node_ids and not filters:
            return
        all_filters: List[MetadataFilter | MetadataFilters] = []
        if node_ids:
            all_filters.append(
                MetadataFilter(
                    key=self._id_column, value=node_ids, operator=FilterOperator.IN
                )
            )
        if filters:
            all_filters.append(filters)
        filters_stmt = ""
        if all_filters:
            all_metadata_filters = MetadataFilters(
                filters=all_filters, condition=FilterCondition.AND
            )
            filters_stmt = self.__parse_metadata_filters_recursively(
                all_metadata_filters
            )
        filters_stmt = f"WHERE {filters_stmt}" if filters_stmt else ""
        query = f'DELETE FROM "{self._schema_name}"."{self._table_name}" {filters_stmt}'
        async with self._engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def aclear(self) -> None:
        """Asynchronously delete all nodes from the table."""
        query = f'TRUNCATE TABLE "{self._schema_name}"."{self._table_name}"'
        async with self._engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def aget_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        """Asynchronously get nodes from the table matching the provided nodes and filters."""
        query = VectorStoreQuery(
            node_ids=node_ids, filters=filters, similarity_top_k=-1
        )
        result = await self.aquery(query)
        return list(result.nodes) if result.nodes else []

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """Asynchronously query vector store."""
        results = await self.__query_columns(query)
        nodes = []
        ids = []
        similarities = []

        for row in results:
            node = metadata_dict_to_node(
                row[self._metadata_json_column], row[self._text_column]
            )
            if row[self._ref_doc_id_column]:
                node_source = TextNode(id_=row[self._ref_doc_id_column])
                node.relationships[NodeRelationship.SOURCE] = (
                    node_source.as_related_node_info()
                )
            nodes.append(node)
            ids.append(row[self._id_column])
            if "distance" in row:
                similarities.append(row["distance"])
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> List[str]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresVectorStore. Use PostgresVectorStore interface instead."
        )

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresVectorStore. Use PostgresVectorStore interface instead."
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresVectorStore. Use PostgresVectorStore interface instead."
        )

    def clear(self) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresVectorStore. Use PostgresVectorStore interface instead."
        )

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[BaseNode]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresVectorStore. Use PostgresVectorStore interface instead."
        )

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresVectorStore. Use PostgresVectorStore interface instead."
        )

    async def set_maintenance_work_mem(self, num_leaves: int, vector_size: int) -> None:
        """Set database maintenance work memory (for index creation)."""
        # Required index memory in MB
        buffer = 1
        index_memory_required = (
            round(50 * num_leaves * vector_size * 4 / 1024 / 1024) + buffer
        )  # Convert bytes to MB
        query = f"SET maintenance_work_mem TO '{index_memory_required} MB';"
        async with self._engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def aapply_vector_index(
        self,
        index: BaseIndex,
        name: Optional[str] = None,
        concurrently: bool = False,
    ) -> None:
        """Create index in the vector store table."""
        if isinstance(index, ExactNearestNeighbor):
            await self.adrop_vector_index()
            return

        function = index.distance_strategy.index_function
        filter = f"WHERE ({index.partial_indexes})" if index.partial_indexes else ""
        params = "WITH " + index.index_options()
        if name is None:
            if index.name == None:
                index.name = self._table_name + DEFAULT_INDEX_NAME_SUFFIX
            name = index.name
        stmt = f"CREATE INDEX {'CONCURRENTLY' if concurrently else ''} {name} ON \"{self._schema_name}\".\"{self._table_name}\" USING {index.index_type} ({self._embedding_column} {function}) {params} {filter};"
        if concurrently:
            async with self._engine.connect() as conn:
                await conn.execute(text("COMMIT"))
                await conn.execute(text(stmt))
        else:
            async with self._engine.connect() as conn:
                await conn.execute(text(stmt))
                await conn.commit()

    async def areindex(self, index_name: Optional[str] = None) -> None:
        """Re-index the vector store table."""
        index_name = index_name or self._table_name + DEFAULT_INDEX_NAME_SUFFIX
        query = f"REINDEX INDEX {index_name};"
        async with self._engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def adrop_vector_index(
        self,
        index_name: Optional[str] = None,
    ) -> None:
        """Drop the vector index."""
        index_name = index_name or self._table_name + DEFAULT_INDEX_NAME_SUFFIX
        query = f"DROP INDEX IF EXISTS {index_name};"
        async with self._engine.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    async def is_valid_index(
        self,
        index_name: Optional[str] = None,
    ) -> bool:
        """Check if index exists in the table."""
        index_name = index_name or self._table_name + DEFAULT_INDEX_NAME_SUFFIX
        query = f"""
        SELECT tablename, indexname
        FROM pg_indexes
        WHERE tablename = '{self._table_name}' AND schemaname = '{self._schema_name}' AND indexname = '{index_name}';
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            results = result_map.fetchall()
        return bool(len(results) == 1)

    async def __query_columns(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> Sequence[RowMapping]:
        """Perform search query on database."""
        filters: List[MetadataFilter | MetadataFilters] = []
        if query.doc_ids:
            filters.append(
                MetadataFilter(
                    key=self._ref_doc_id_column,
                    value=query.doc_ids,
                    operator=FilterOperator.IN,
                )
            )
        if query.node_ids:
            filters.append(
                MetadataFilter(
                    key=self._id_column,
                    value=query.node_ids,
                    operator=FilterOperator.IN,
                )
            )
        if query.filters:
            filters.append(query.filters)

        # Note:
        # Hybrid search is not yet supported, so following fields in `query` are ignored:
        #     query_str, mode, alpha, mmr_threshold, sparse_top_k, hybrid_top_k
        # Vectors are already stored `self._embedding_column` so a custom embedding_field is ignored.
        query_filters = MetadataFilters(filters=filters, condition=FilterCondition.AND)

        filters_stmt = self.__parse_metadata_filters_recursively(query_filters)
        filters_stmt = f"WHERE {filters_stmt}" if filters_stmt else ""
        operator = self._distance_strategy.operator
        search_function = self._distance_strategy.search_function

        # query_embedding is used for scoring
        scoring_stmt = (
            f", {search_function}({self._embedding_column}, '{query.query_embedding}') as distance"
            if query.query_embedding
            else ""
        )

        # results are sorted on ORDER BY query_embedding
        order_stmt = (
            f" ORDER BY {self._embedding_column} {operator} '{query.query_embedding}' "
            if query.query_embedding
            else ""
        )

        # similarity_top_k is used for limiting number of retrieved nodes
        limit_stmt = (
            f" LIMIT {query.similarity_top_k} " if query.similarity_top_k >= 1 else ""
        )

        query_stmt = f'SELECT * {scoring_stmt} FROM "{self._schema_name}"."{self._table_name}" {filters_stmt} {order_stmt} {limit_stmt}'
        async with self._engine.connect() as conn:
            if self._index_query_options:
                query_options_stmt = (
                    f"SET LOCAL {self._index_query_options.to_string()};"
                )
                await conn.execute(text(query_options_stmt))
            result = await conn.execute(text(query_stmt))
            result_map = result.mappings()
            results = result_map.fetchall()
        return results

    def __parse_metadata_filters_recursively(
        self, metadata_filters: MetadataFilters
    ) -> str:
        """
        Parses a MetadataFilters object into a SQL WHERE clause.
        Supports a mixed list of MetadataFilter and nested MetadataFilters.
        """
        if not metadata_filters.filters:
            return ""

        where_clauses = []
        for filter_item in metadata_filters.filters:
            if isinstance(filter_item, MetadataFilter):
                clause = self.__parse_metadata_filter(filter_item)
                if clause:
                    where_clauses.append(clause)
            elif isinstance(filter_item, MetadataFilters):
                # Handle nested filters recursively
                nested_clause = self.__parse_metadata_filters_recursively(filter_item)
                if nested_clause:
                    where_clauses.append(f"({nested_clause})")

        # Combine clauses with the specified condition
        condition_value = (
            metadata_filters.condition.value
            if metadata_filters.condition
            else FilterCondition.AND.value
        )
        return f" {condition_value} ".join(where_clauses) if where_clauses else ""

    def __parse_metadata_filter(self, filter: MetadataFilter) -> str:
        key = self.__to_postgres_key(filter.key)
        op = self.__to_postgres_operator(filter.operator)
        if filter.operator == FilterOperator.IS_EMPTY:
            # checks for emptiness of a field, so value is ignored
            # cast to jsonb to check array length
            return f"((({key})::jsonb IS NULL) OR (jsonb_array_length(({key})::jsonb) = 0))"
        if filter.operator == FilterOperator.CONTAINS:
            # Expects a list stored in the metadata, and a single value to compare
            if isinstance(filter.value, list):
                # skip improperly provided filter and raise a warning
                warnings.warn(
                    f"""Expecting a scalar in the filter value, but got {type(filter.value)}.
                    Ignoring this filter:
                    Key -> '{filter.key}'
                    Operator -> '{filter.operator}'
                    Value -> '{filter.value}'"""
                )
                return ""
            return f"({key})::jsonb {op} '[\"{filter.value}\"]' "
        if filter.operator == FilterOperator.TEXT_MATCH:
            return f"{key} {op} '%{filter.value}%' "
        if filter.operator in [
            FilterOperator.ANY,
            FilterOperator.ALL,
            FilterOperator.IN,
            FilterOperator.NIN,
        ]:
            # Expect a single value in metadata and a list to compare
            if not isinstance(filter.value, list):
                # skip improperly provided filter and raise a warning
                warnings.warn(
                    f"""Expecting List in the filter value, but got {type(filter.value)}.
                    Ignoring this filter:
                    Key -> '{filter.key}'
                    Operator -> '{filter.operator}'
                    Value -> '{filter.value}'"""
                )
                return ""
            filter_value = ", ".join(f"'{e}'" for e in filter.value)
            if filter.operator in [FilterOperator.ANY, FilterOperator.ALL]:
                return f"({key})::jsonb {op} (ARRAY[{filter_value}])"
            else:
                return f"{key} {op} ({filter_value})"

        # Check if value is a number. If so, cast the metadata value to a float
        # This is necessary because the metadata is stored as a string.
        if isinstance(filter.value, (int, float, str)):
            try:
                return f"{key}::float {op} {float(filter.value)}"
            except ValueError:
                # If not a number, then treat it as a string
                pass
        return f"{key} {op} '{filter.value}'"

    def __to_postgres_operator(self, operator: FilterOperator) -> str:
        if operator == FilterOperator.EQ:
            return "="
        elif operator == FilterOperator.GT:
            return ">"
        elif operator == FilterOperator.LT:
            return "<"
        elif operator == FilterOperator.NE:
            return "!="
        elif operator == FilterOperator.GTE:
            return ">="
        elif operator == FilterOperator.LTE:
            return "<="
        elif operator == FilterOperator.IN:
            return "IN"
        elif operator == FilterOperator.NIN:
            return "NOT IN"
        elif operator == FilterOperator.ANY:
            return "?|"
        elif operator == FilterOperator.ALL:
            return "?&"
        elif operator == FilterOperator.CONTAINS:
            return "@>"
        elif operator == FilterOperator.TEXT_MATCH:
            return "LIKE"
        elif operator == FilterOperator.IS_EMPTY:
            return "IS_EMPTY"
        else:
            warnings.warn(f"Unknown operator: {operator}, fallback to '='")
            return "="

    def __to_postgres_key(self, key: str) -> str:
        if key in [
            *self._metadata_columns,
            self._id_column,
            self._ref_doc_id_column,
            self._text_column,
        ]:
            return key
        return f"{self._metadata_json_column}->>'{key}'"
