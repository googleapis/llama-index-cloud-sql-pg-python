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

from typing import AsyncIterable, Callable, Iterable, List, Optional

from llama_index.core.bridge.pydantic import ConfigDict
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

from .async_reader import AsyncPostgresReader
from .engine import PostgresEngine

DEFAULT_METADATA_COL = "li_metadata"


class PostgresReader(BasePydanticReader):
    """Chat Store Table stored in an Cloud SQL for PostgreSQL database."""

    __create_key = object()
    is_remote: bool = True

    def __init__(
        self,
        key: object,
        engine: PostgresEngine,
        reader: AsyncPostgresReader,
        is_remote: bool = True,
    ) -> None:
        """PostgresReader constructor.

        Args:
            key (object): Prevent direct constructor usage.
            engine (PostgresEngine): PostgresEngine with pool connection to the Cloud SQL postgres database
            reader (AsyncPostgresReader): The async only PostgresReader implementation
            is_remote (Optional[bool]): Whether the data is loaded from a remote API or a local file.

        Raises:
            Exception: If called directly by user.
        """
        if key != PostgresReader.__create_key:
            raise Exception("Only create class through 'create' method!")

        super().__init__(is_remote=is_remote)

        self._engine = engine
        self.__reader = reader

    @classmethod
    async def create(
        cls: type[PostgresReader],
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
    ) -> PostgresReader:
        """Asynchronously create an PostgresReader instance.

        Args:
            engine (PostgresEngine): PostgresEngine with pool connection to the Cloud SQL postgres database
            query (Optional[str], optional): SQL query. Defaults to None.
            table_name (Optional[str], optional): Name of table to query. Defaults to None.
            schema_name (str, optional): Name of the schema where table is located. Defaults to "public".
            content_columns (Optional[list[str]], optional): Column that represent a Document's page_content. Defaults to the first column.
            metadata_columns (Optional[list[str]], optional): Column(s) that represent a Document's metadata. Defaults to None.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to "li_metadata".
            format (Optional[str], optional): Format of page content (OneOf: text, csv, YAML, JSON). Defaults to 'text'.
            formatter (Optional[Callable], optional): A function to format page content (OneOf: format, formatter). Defaults to None.
            is_remote (Optional[bool]): Whether the data is loaded from a remote API or a local file.


        Returns:
            PostgresReader: A newly created instance of PostgresReader.
        """
        coro = AsyncPostgresReader.create(
            engine=engine,
            query=query,
            table_name=table_name,
            schema_name=schema_name,
            content_columns=content_columns,
            metadata_columns=metadata_columns,
            metadata_json_column=metadata_json_column,
            format=format,
            formatter=formatter,
            is_remote=is_remote,
        )
        reader = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, reader, is_remote)

    @classmethod
    def create_sync(
        cls: type[PostgresReader],
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
    ) -> PostgresReader:
        """Synchronously create an PostgresReader instance.

        Args:
            engine (PostgresEngine): PostgresEngine with pool connection to the Cloud SQL postgres database
            query (Optional[str], optional): SQL query. Defaults to None.
            table_name (Optional[str], optional): Name of table to query. Defaults to None.
            schema_name (str, optional): Name of the schema where table is located. Defaults to "public".
            content_columns (Optional[list[str]], optional): Column that represent a Document's page_content. Defaults to the first column.
            metadata_columns (Optional[list[str]], optional): Column(s) that represent a Document's metadata. Defaults to None.
            metadata_json_column (Optional[str], optional): Column to store metadata as JSON. Defaults to "li_metadata".
            format (Optional[str], optional): Format of page content (OneOf: text, csv, YAML, JSON). Defaults to 'text'.
            formatter (Optional[Callable], optional): A function to format page content (OneOf: format, formatter). Defaults to None.
            is_remote (Optional[bool]): Whether the data is loaded from a remote API or a local file.


        Returns:
            PostgresReader: A newly created instance of PostgresReader.
        """
        coro = AsyncPostgresReader.create(
            engine=engine,
            query=query,
            table_name=table_name,
            schema_name=schema_name,
            content_columns=content_columns,
            metadata_columns=metadata_columns,
            metadata_json_column=metadata_json_column,
            format=format,
            formatter=formatter,
            is_remote=is_remote,
        )
        reader = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, reader, is_remote)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "PostgresReader"

    async def aload_data(self) -> list[Document]:
        """Asynchronously load Cloud SQL postgres data into Document objects."""
        return await self._engine._run_as_async(self.__reader.aload_data())

    def load_data(self) -> list[Document]:
        """Synchronously load Cloud SQL postgres data into Document objects."""
        return self._engine._run_as_sync(self.__reader.aload_data())

    async def alazy_load_data(self) -> AsyncIterable[Document]:  # type: ignore
        """Asynchronously load Cloud SQL postgres data into Document objects lazily."""
        # The return type in the underlying base class is an Iterable which we are overriding to an AsyncIterable in this implementation.
        iterator = self.__reader.alazy_load_data().__aiter__()
        while True:
            try:
                result = await self._engine._run_as_async(iterator.__anext__())
                yield result
            except StopAsyncIteration:
                break

    def lazy_load_data(self) -> Iterable[Document]:  # type: ignore
        """Synchronously load Cloud SQL postgres data into Document objects lazily."""
        iterator = self.__reader.alazy_load_data().__aiter__()
        while True:
            try:
                result = self._engine._run_as_sync(iterator.__anext__())
                yield result
            except StopAsyncIteration:
                break
