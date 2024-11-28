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

from typing import List, Optional

from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.storage.index_store.types import BaseIndexStore

from .async_index_store import AsyncPostgresIndexStore
from .engine import PostgresEngine


class PostgresIndexStore(BaseIndexStore):
    """Index Store Table stored in an Cloud SQL for PostgreSQL database."""

    __create_key = object()

    def __init__(
        self, key: object, engine: PostgresEngine, index_store: AsyncPostgresIndexStore
    ):
        """PostgresIndexStore constructor.

        Args:
            key (object): Key to prevent direct constructor usage.
            engine (PostgresEngine): Database connection pool.
            index_store (AsyncPostgresIndexStore): The async only IndexStore implementation

        Raises:
            Exception: If constructor is directly called by the user.
        """
        if key != PostgresIndexStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._engine = engine
        self.__index_store = index_store

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
    ) -> PostgresIndexStore:
        """Create a new PostgresIndexStore instance.

        Args:
            engine (PostgresEngine): Postgres engine to use.
            table_name (str): Table name that stores the index metadata.
            schema_name (str): The schema name where the table is located. Defaults to "public"

        Raises:
            ValueError: If the table provided does not contain required schema.

        Returns:
            PostgresIndexStore: A newly created instance of PostgresIndexStore.
        """
        coro = AsyncPostgresIndexStore.create(engine, table_name, schema_name)
        index_store = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, index_store)

    @classmethod
    def create_sync(
        cls,
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
    ) -> PostgresIndexStore:
        """Create a new PostgresIndexStore sync instance.

        Args:
            engine (PostgresEngine): Postgres engine to use.
            table_name (str): Table name that stores the index metadata.
            schema_name (str): The schema name where the table is located. Defaults to "public"

        Raises:
            ValueError: If the table provided does not contain required schema.

        Returns:
            PostgresIndexStore: A newly created instance of PostgresIndexStore.
        """
        coro = AsyncPostgresIndexStore.create(engine, table_name, schema_name)
        index_store = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, index_store)

    async def aindex_structs(self) -> List[IndexStruct]:
        """Get all index structs.

        Returns:
            List[IndexStruct]: index structs

        """
        return await self._engine._run_as_async(self.__index_store.aindex_structs())

    def index_structs(self) -> List[IndexStruct]:
        """Get all index structs.

        Returns:
            List[IndexStruct]: index structs

        """
        return self._engine._run_as_sync(self.__index_store.aindex_structs())

    async def aadd_index_struct(self, index_struct: IndexStruct) -> None:
        """Add an index struct.

        Args:
            index_struct (IndexStruct): index struct

        """
        return await self._engine._run_as_async(
            self.__index_store.aadd_index_struct(index_struct)
        )

    def add_index_struct(self, index_struct: IndexStruct) -> None:
        """Add an index struct.

        Args:
            index_struct (IndexStruct): index struct

        """
        return self._engine._run_as_sync(
            self.__index_store.aadd_index_struct(index_struct)
        )

    async def adelete_index_struct(self, key: str) -> None:
        """Delete an index struct.

        Args:
            key (str): index struct key

        """
        return await self._engine._run_as_async(
            self.__index_store.adelete_index_struct(key)
        )

    def delete_index_struct(self, key: str) -> None:
        """Delete an index struct.

        Args:
            key (str): index struct key

        """
        return self._engine._run_as_sync(self.__index_store.adelete_index_struct(key))

    async def aget_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[IndexStruct]:
        """Get an index struct.

        Args:
            struct_id (Optional[str]): index struct id

        """
        return await self._engine._run_as_async(
            self.__index_store.aget_index_struct(struct_id)
        )

    def get_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[IndexStruct]:
        """Get an index struct.

        Args:
            struct_id (Optional[str]): index struct id

        """
        return self._engine._run_as_sync(
            self.__index_store.aget_index_struct(struct_id)
        )
