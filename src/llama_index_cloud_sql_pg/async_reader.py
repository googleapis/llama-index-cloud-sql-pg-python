# Copyright 2025 Google LLC
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

import json
from typing import Any, AsyncIterable, Callable, Iterable, Iterator, List, Optional

from llama_index.core.bridge.pydantic import ConfigDict
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import PostgresEngine

DEFAULT_METADATA_COL = "llamaindex_metadata"


def text_formatter(row: dict, content_columns: list[str]) -> str:
    """txt document formatter."""
    return " ".join(str(row[column]) for column in content_columns if column in row)


def csv_formatter(row: dict, content_columns: list[str]) -> str:
    """CSV document formatter."""
    return ", ".join(str(row[column]) for column in content_columns if column in row)


def yaml_formatter(row: dict, content_columns: list[str]) -> str:
    """YAML document formatter."""
    return "\n".join(
        f"{column}: {str(row[column])}" for column in content_columns if column in row
    )


def json_formatter(row: dict, content_columns: list[str]) -> str:
    """JSON document formatter."""
    dictionary = {}
    for column in content_columns:
        if column in row:
            dictionary[column] = row[column]
    return json.dumps(dictionary)


def _parse_doc_from_row(
    content_columns: Iterable[str],
    metadata_columns: Iterable[str],
    row: dict,
    formatter: Callable = text_formatter,
    metadata_json_column: Optional[str] = DEFAULT_METADATA_COL,
) -> Document:
    """Parse row into document."""
    text = formatter(row, content_columns)
    metadata: dict[str, Any] = {}
    # unnest metadata from llamaindex_metadata column
    if metadata_json_column and row.get(metadata_json_column):
        for k, v in row[metadata_json_column].items():
            metadata[k] = v
    # load metadata from other columns
    for column in metadata_columns:
        if column in row and column != metadata_json_column:
            metadata[column] = row[column]

    return Document(text=text, extra_info=metadata)


class AsyncPostgresReader(BasePydanticReader):
    """Load documents from Cloud SQL for PostgreSQL.

    Each document represents one row of the result. The `content_columns` are
    written into the `text` of the document. The `metadata_columns` are written
    into the `metadata` of the document. By default, first columns is written into
    the `text` and everything else into the `metadata`.
    """

    __create_key = object()
    is_remote: bool = True

    def __init__(
        self,
        key: object,
        pool: AsyncEngine,
        query: str,
        content_columns: list[str],
        metadata_columns: list[str],
        formatter: Callable,
        metadata_json_column: Optional[str] = None,
        is_remote: bool = True,
    ) -> None:
        """AsyncPostgresReader constructor.

        Args:
            key (object): Prevent direct constructor usage.
            engine (PostgresEngine): AsyncEngine with pool connection to the Cloud SQL Postgres database
            query (Optional[str], optional): SQL query. Defaults to None.
            content_columns (Optional[list[str]], optional): Column that represent a Document's page_content. Defaults to the first column.
            metadata_columns (Optional[list[str]], optional): Column(s) that represent a Document's metadata. Defaults to None.
            formatter (Optional[Callable], optional): A function to format page content (OneOf: format, formatter). Defaults to None.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to "llamaindex_metadata".
            is_remote (bool): Whether the data is loaded from a remote API or a local file.

        Raises:
            Exception: If called directly by user.
        """
        if key != AsyncPostgresReader.__create_key:
            raise Exception("Only create class through 'create' method!")

        super().__init__(is_remote=is_remote)

        self._pool = pool
        self._query = query
        self._content_columns = content_columns
        self._metadata_columns = metadata_columns
        self._formatter = formatter
        self._metadata_json_column = metadata_json_column

    @classmethod
    async def create(
        cls: type[AsyncPostgresReader],
        engine: PostgresEngine,
        query: Optional[str] = None,
        table_name: Optional[str] = None,
        schema_name: str = "public",
        content_columns: Optional[list[str]] = None,
        metadata_columns: Optional[list[str]] = None,
        metadata_json_column: Optional[str] = None,
        format: Optional[str] = None,
        formatter: Optional[Callable] = None,
        is_remote: bool = True,
    ) -> AsyncPostgresReader:
        """Create an AsyncPostgresReader instance.

        Args:
            engine (PostgresEngine):AsyncEngine with pool connection to the Cloud SQL Postgres database
            query (Optional[str], optional): SQL query. Defaults to None.
            table_name (Optional[str], optional): Name of table to query. Defaults to None.
            schema_name (str, optional): Name of the schema where table is located. Defaults to "public".
            content_columns (Optional[list[str]], optional): Column that represent a Document's page_content. Defaults to the first column.
            metadata_columns (Optional[list[str]], optional): Column(s) that represent a Document's metadata. Defaults to None.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to "llamaindex_metadata".
            format (Optional[str], optional): Format of page content (OneOf: text, csv, YAML, JSON). Defaults to 'text'.
            formatter (Optional[Callable], optional): A function to format page content (OneOf: format, formatter). Defaults to None.
            is_remote (bool): Whether the data is loaded from a remote API or a local file.


        Returns:
            AsyncPostgresReader: A newly created instance of AsyncPostgresReader.
        """
        if table_name and query:
            raise ValueError("Only one of 'table_name' or 'query' should be specified.")
        if not table_name and not query:
            raise ValueError(
                "At least one of the parameters 'table_name' or 'query' needs to be provided"
            )
        if format and formatter:
            raise ValueError("Only one of 'format' or 'formatter' should be specified.")

        if format and format not in ["csv", "text", "JSON", "YAML"]:
            raise ValueError("format must be type: 'csv', 'text', 'JSON', 'YAML'")
        if formatter:
            formatter = formatter
        elif format == "csv":
            formatter = csv_formatter
        elif format == "YAML":
            formatter = yaml_formatter
        elif format == "JSON":
            formatter = json_formatter
        else:
            formatter = text_formatter

        if not query:
            query = f'SELECT * FROM "{schema_name}"."{table_name}"'

        async with engine._pool.connect() as connection:
            result_proxy = await connection.execute(text(query))
            column_names = list(result_proxy.keys())
            # Select content or default to first column
            content_columns = content_columns or [column_names[0]]
            # Select metadata columns
            metadata_columns = metadata_columns or [
                col for col in column_names if col not in content_columns
            ]

            # Check validity of metadata json column
            if metadata_json_column and metadata_json_column not in column_names:
                raise ValueError(
                    f"Column {metadata_json_column} not found in query result {column_names}."
                )

            if metadata_json_column and metadata_json_column in column_names:
                metadata_json_column = metadata_json_column
            elif DEFAULT_METADATA_COL in column_names:
                metadata_json_column = DEFAULT_METADATA_COL
            else:
                metadata_json_column = None

            # check validity of other column
            all_names = content_columns + metadata_columns
            for name in all_names:
                if name not in column_names:
                    raise ValueError(
                        f"Column {name} not found in query result {column_names}."
                    )
        return cls(
            key=cls.__create_key,
            pool=engine._pool,
            query=query,
            content_columns=content_columns,
            metadata_columns=metadata_columns,
            formatter=formatter,
            metadata_json_column=metadata_json_column,
            is_remote=is_remote,
        )

    @classmethod
    def class_name(cls) -> str:
        return "AsyncPostgresReader"

    async def aload_data(self) -> list[Document]:
        """Asynchronously load Cloud SQL Postgres data into Document objects."""
        return [doc async for doc in self.alazy_load_data()]

    async def alazy_load_data(self) -> AsyncIterable[Document]:  # type: ignore
        """Asynchronously load Cloud SQL Postgres data into Document objects lazily."""
        async with self._pool.connect() as connection:
            result_proxy = await connection.execute(text(self._query))
            # load document one by one
            while True:
                row = result_proxy.fetchone()
                if not row:
                    break

                row_data = {}
                column_names = self._content_columns + self._metadata_columns
                column_names += (
                    [self._metadata_json_column] if self._metadata_json_column else []
                )
                for column in column_names:
                    value = getattr(row, column)
                    row_data[column] = value

                yield _parse_doc_from_row(
                    self._content_columns,
                    self._metadata_columns,
                    row_data,
                    self._formatter,
                    self._metadata_json_column,
                )

    def lazy_load_data(self) -> Iterator[Document]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresReader. Use PostgresReader interface instead."
        )

    def load_data(self) -> List[Document]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresReader. Use PostgresReader interface instead."
        )