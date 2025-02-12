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
from typing import List, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import PostgresEngine


class AsyncPostgresChatStore(BaseChatStore):
    """Chat Store Table stored in an CloudSQL for PostgreSQL database."""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AsyncEngine,
        table_name: str,
        schema_name: str = "public",
    ):
        """AsyncPostgresChatStore constructor.

        Args:
            key (object): Key to prevent direct constructor usage.
            engine (PostgresEngine): Database connection pool.
            table_name (str): Table name that stores the chat store.
            schema_name (str): The schema name where the table is located.
                Defaults to "public"

        Raises:
            Exception: If constructor is directly called by the user.
        """
        if key != AsyncPostgresChatStore.__create_key:
            raise Exception("Only create class through 'create' method!")

        # Delegate to Pydantic's __init__
        super().__init__()
        self._engine = engine
        self._table_name = table_name
        self._schema_name = schema_name

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
    ) -> AsyncPostgresChatStore:
        """Create a new AsyncPostgresChatStore instance.

        Args:
            engine (PostgresEngine): Postgres engine to use.
            table_name (str): Table name that stores the chat store.
            schema_name (str): The schema name where the table is located.
                Defaults to "public"

        Raises:
            ValueError: If the table provided does not contain required schema.

        Returns:
            AsyncPostgresChatStore: A newly created instance of AsyncPostgresChatStore.
        """
        table_schema = await engine._aload_table_schema(table_name, schema_name)
        column_names = table_schema.columns.keys()

        required_columns = ["id", "key", "message"]

        if not (all(x in column_names for x in required_columns)):
            raise ValueError(
                f"Table '{schema_name}'.'{table_name}' has an incorrect schema.\n"
                f"Expected column names: {required_columns}\n"
                f"Provided column names: {column_names}\n"
                "Please create the table with the following schema:\n"
                f"CREATE TABLE {schema_name}.{table_name} (\n"
                "    id SERIAL PRIMARY KEY,\n"
                "    key VARCHAR NOT NULL,\n"
                "    message JSON NOT NULL\n"
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

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "AsyncPostgresChatStore"

    async def aset_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Asynchronously sets the chat messages for a specific key.

        Args:
            key (str): A unique identifier for the chat.
            messages (List[ChatMessage]): A list of `ChatMessage` objects to upsert.

        Returns:
            None

        """
        query = f"""DELETE FROM "{self._schema_name}"."{self._table_name}" WHERE key = '{key}'; """
        await self.__aexecute_query(query)
        insert_query = f"""
                INSERT INTO "{self._schema_name}"."{self._table_name}" (key, message)
                VALUES (:key, :message);"""

        params = [
            {
                "key": key,
                "message": json.dumps(message.model_dump()),
            }
            for message in messages
        ]

        await self.__aexecute_query(insert_query, params)

    async def aget_messages(self, key: str) -> List[ChatMessage]:
        """Asynchronously retrieves the chat messages associated with a specific key.

        Args:
            key (str): A unique identifier for which the messages are to be retrieved.

        Returns:
            List[ChatMessage]: A list of `ChatMessage` objects associated with the provided key.
            If no messages are found, an empty list is returned.
        """
        query = f"""SELECT message from "{self._schema_name}"."{self._table_name}" WHERE key = '{key}' ORDER BY id;"""
        results = await self.__afetch_query(query)
        if results:
            return [
                ChatMessage.model_validate(result.get("message")) for result in results
            ]
        return []

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        """Asynchronously adds a new chat message to the specified key.

        Args:
            key (str): A unique identifierfor the chat to which the message is added.
            message (ChatMessage): The `ChatMessage` object that is to be added.

        Returns:
            None
        """
        insert_query = f"""
                INSERT INTO "{self._schema_name}"."{self._table_name}" (key, message)
                VALUES (:key, :message);"""
        params = {"key": key, "message": json.dumps(message.model_dump())}

        await self.__aexecute_query(insert_query, params)

    async def adelete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Asynchronously deletes the chat messages associated with a specific key.

        Args:
            key (str): A unique identifier for the chat whose messages are to be deleted.

        Returns:
            Optional[List[ChatMessage]]: A list of `ChatMessage` objects that were deleted, or `None` if no messages
            were associated with the key or could be deleted.
        """
        query = f"""DELETE FROM "{self._schema_name}"."{self._table_name}" WHERE key = '{key}' RETURNING *; """
        results = await self.__afetch_query(query)
        if results:
            return [
                ChatMessage.model_validate(result.get("message")) for result in results
            ]
        return None

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Asynchronously deletes a specific chat message by index from the messages associated with a given key.

        Args:
            key (str): A unique identifier for the chat whose messages are to be deleted.
            idx (int): The index of the `ChatMessage` to be deleted from the list of messages.

        Returns:
            Optional[ChatMessage]: The `ChatMessage` object that was deleted, or `None` if no message
            was associated with the key or could be deleted.
        """
        query = f"""SELECT * from "{self._schema_name}"."{self._table_name}" WHERE key = '{key}' ORDER BY id;"""
        results = await self.__afetch_query(query)
        if results:
            if idx >= len(results):
                return None
            id_to_be_deleted = results[idx].get("id")
            delete_query = f"""DELETE FROM "{self._schema_name}"."{self._table_name}" WHERE id = '{id_to_be_deleted}' RETURNING *;"""
            result = await self.__afetch_query(delete_query)
            result = result[0]
            if result:
                return ChatMessage.model_validate(result.get("message"))
            return None
        return None

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Asynchronously deletes the last chat message associated with a given key.

        Args:
            key (str): A unique identifier for the chat whose message is to be deleted.

        Returns:
            Optional[ChatMessage]: The `ChatMessage` object that was deleted, or `None` if no message
            was associated with the key or could be deleted.
        """
        query = f"""SELECT * from "{self._schema_name}"."{self._table_name}" WHERE key = '{key}' ORDER BY id DESC LIMIT 1;"""
        results = await self.__afetch_query(query)
        if results:
            id_to_be_deleted = results[0].get("id")
            delete_query = f"""DELETE FROM "{self._schema_name}"."{self._table_name}" WHERE id = '{id_to_be_deleted}' RETURNING *;"""
            result = await self.__afetch_query(delete_query)
            result = result[0]
            if result:
                return ChatMessage.model_validate(result.get("message"))
            return None
        return None

    async def aget_keys(self) -> List[str]:
        """Asynchronously retrieves a list of all keys.

        Returns:
            Optional[str]: A list of strings representing the keys. If no keys are found, an empty list is returned.
        """
        query = (
            f"""SELECT distinct key from "{self._schema_name}"."{self._table_name}";"""
        )
        results = await self.__afetch_query(query)
        keys = []
        if results:
            keys = [row.get("key") for row in results]
        return keys

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresChatStore. Use PostgresChatStore interface instead."
        )

    def get_messages(self, key: str) -> List[ChatMessage]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresChatStore. Use PostgresChatStore interface instead."
        )

    def add_message(self, key: str, message: ChatMessage) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresChatStore. Use PostgresChatStore interface instead."
        )

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresChatStore. Use PostgresChatStore interface instead."
        )

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresChatStore. Use PostgresChatStore interface instead."
        )

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresChatStore. Use PostgresChatStore interface instead."
        )

    def get_keys(self) -> List[str]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresChatStore. Use PostgresChatStore interface instead."
        )
