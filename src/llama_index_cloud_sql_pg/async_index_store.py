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

import json
import warnings
from typing import Optional

from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.core.storage.index_store.utils import (
    index_struct_to_json,
    json_to_index_struct,
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import PostgresEngine


class AsyncPostgresIndexStore(BaseIndexStore):
    """Index Store Table stored in an Cloud SQL for PostgreSQL database."""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AsyncEngine,
        table_name: str,
        schema_name: str = "public",
    ):
        """AsyncPostgresIndexStore constructor.

        Args:
            key (object): Key to prevent direct constructor usage.
            engine (PostgresEngine): Database connection pool.
            table_name (str): Table name that stores the index metadata.
            schema_name (str): The schema name where the table is located. Defaults to "public"

        Raises:
            Exception: If constructor is directly called by the user.
        """
        if key != AsyncPostgresIndexStore.__create_key:
            raise Exception("Only create class through 'create' method!")
        self._engine = engine
        self._table_name = table_name
        self._schema_name = schema_name

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
    ) -> AsyncPostgresIndexStore:
        """Create a new AsyncPostgresIndexStore instance.

        Args:
            engine (PostgresEngine): Postgres engine to use.
            table_name (str): Table name that stores the index metadata.
            schema_name (str): The schema name where the table is located. Defaults to "public"

        Raises:
            ValueError: If the table provided does not contain required schema.

        Returns:
            AsyncPostgresIndexStore: A newly created instance of AsyncPostgresIndexStore.
        """
        table_schema = await engine._aload_table_schema(table_name, schema_name)
        column_names = table_schema.columns.keys()

        required_columns = ["index_id", "type", "index_data"]

        if not (all(x in column_names for x in required_columns)):
            raise ValueError(
                f"Table '{schema_name}'.'{table_name}' has an incorrect schema.\n"
                f"Expected column names: {required_columns}\n"
                f"Provided column names: {column_names}\n"
                "Please create the table with the following schema:\n"
                f"CREATE TABLE {schema_name}.{table_name} (\n"
                "    index_id VARCHAR PRIMARY KEY,\n"
                "    type VARCHAR NOT NULL,\n"
                "    index_data JSONB NOT NULL\n"
                ");"
            )

        return cls(cls.__create_key, engine._pool, table_name, schema_name)

    async def __aexecute_query(self, query, params=None):
        async with self._engine.connect() as conn:
            await conn.execute(text(query), params)
            await conn.commit()

    async def __afetch_query(self, query):
        async with self._engine.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            results = result_map.fetchall()
            await conn.commit()
        return results

    async def aindex_structs(self) -> list[IndexStruct]:
        """Get all index structs.

        Returns:
            list[IndexStruct]: index structs

        """
        query = f"""SELECT * from "{self._schema_name}"."{self._table_name}";"""
        index_list = await self.__afetch_query(query)

        if index_list:
            return [json_to_index_struct(index["index_data"]) for index in index_list]
        return []

    async def aadd_index_struct(self, index_struct: IndexStruct) -> None:
        """Add an index struct.

        Args:
            index_struct (IndexStruct): index struct

        """
        key = index_struct.index_id
        data = index_struct_to_json(index_struct)
        type = index_struct.get_type()

        index_row = {
            "index_id": key,
            "type": type,
            "index_data": json.dumps(data),
        }

        insert_query = f'INSERT INTO "{self._schema_name}"."{self._table_name}"(index_id, type, index_data) '
        values_statement = f"VALUES (:index_id, :type, :index_data)"
        upsert_statement = " ON CONFLICT (index_id) DO UPDATE SET type = EXCLUDED.type, index_data = EXCLUDED.index_data;"

        query = insert_query + values_statement + upsert_statement
        await self.__aexecute_query(query, index_row)

    async def adelete_index_struct(self, key: str) -> None:
        """Delete an index struct.

        Args:
            key (str): index struct key

        """
        query = f"""DELETE FROM "{self._schema_name}"."{self._table_name}" WHERE index_id = '{key}'; """
        await self.__aexecute_query(query)
        return

    async def aget_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[IndexStruct]:
        """Get an index struct.

        Args:
            struct_id (Optional[str]): index struct id

        """
        if struct_id is None:
            structs = await self.aindex_structs()
            if len(structs) == 1:
                return structs[0]
            warnings.warn("No struct_id specified and more than one struct exists.")
            return None
        else:
            query = f"""SELECT * from "{self._schema_name}"."{self._table_name}" WHERE index_id = '{struct_id}';"""
            result = await self.__afetch_query(query)
            if result:
                json = result[0]
                if json is None:
                    return None
                index_data = json.get("index_data")

            if index_data:
                return json_to_index_struct(index_data)
            return None

    def index_structs(self) -> list[IndexStruct]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresIndexStore. Use PostgresIndexStore interface instead."
        )

    def add_index_struct(self, index_struct: IndexStruct) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresIndexStore. Use PostgresIndexStore interface instead."
        )

    def delete_index_struct(self, key: str) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresIndexStore. Use PostgresIndexStore interface instead."
        )

    def get_index_struct(
        self, struct_id: Optional[str] = None
    ) -> Optional[IndexStruct]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresIndexStore. Use PostgresIndexStore interface instead."
        )
