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

from typing import List, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store.base import BaseChatStore

from .async_chat_store import AsyncPostgresChatStore
from .engine import PostgresEngine


class PostgresChatStore(BaseChatStore):
    """Chat Store Table stored in an Cloud SQL for PostgreSQL database."""

    __create_key = object()

    def __init__(
        self, key: object, engine: PostgresEngine, chat_store: AsyncPostgresChatStore
    ):
        """PostgresChatStore constructor.

        Args:
            key (object): Key to prevent direct constructor usage.
            engine (PostgresEngine): Database connection pool.
            chat_store (AsyncPostgresChatStore): The async only IndexStore implementation

        Raises:
            Exception: If constructor is directly called by the user.
        """
        if key != PostgresChatStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )

        # Delegate to Pydantic's __init__
        super().__init__()
        self._engine = engine
        self.__chat_store = chat_store

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
    ) -> PostgresChatStore:
        """Create a new PostgresChatStore instance.

        Args:
            engine (PostgresEngine): Postgres engine to use.
            table_name (str): Table name that stores the chat store.
            schema_name (str): The schema name where the table is located. Defaults to "public"

        Raises:
            ValueError: If the table provided does not contain required schema.

        Returns:
            PostgresChatStore: A newly created instance of PostgresChatStore.
        """
        coro = AsyncPostgresChatStore.create(engine, table_name, schema_name)
        chat_store = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, chat_store)

    @classmethod
    def create_sync(
        cls,
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
    ) -> PostgresChatStore:
        """Create a new PostgresChatStore sync instance.

        Args:
            engine (PostgresEngine): Postgres engine to use.
            table_name (str): Table name that stores the chat store.
            schema_name (str): The schema name where the table is located. Defaults to "public"

        Raises:
            ValueError: If the table provided does not contain required schema.

        Returns:
            PostgresChatStore: A newly created instance of PostgresChatStore.
        """
        coro = AsyncPostgresChatStore.create(engine, table_name, schema_name)
        chat_store = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, chat_store)

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "PostgresChatStore"

    async def aset_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Asynchronously sets the chat messages for a specific key.

        Args:
            key (str): A unique identifier for the chat.
            messages (List[ChatMessage]): A list of `ChatMessage` objects to upsert.

        Returns:
            None

        """
        return await self._engine._run_as_async(
            self.__chat_store.aset_messages(key=key, messages=messages)
        )

    async def aget_messages(self, key: str) -> List[ChatMessage]:
        """Asynchronously retrieves the chat messages associated with a specific key.

        Args:
            key (str): A unique identifier for which the messages are to be retrieved.

        Returns:
            List[ChatMessage]: A list of `ChatMessage` objects associated with the provided key.
            If no messages are found, an empty list is returned.
        """
        return await self._engine._run_as_async(
            self.__chat_store.aget_messages(key=key)
        )

    async def async_add_message(self, key: str, message: ChatMessage) -> None:
        """Asynchronously adds a new chat message to the specified key.

        Args:
            key (str): A unique identifierfor the chat to which the message is added.
            message (ChatMessage): The `ChatMessage` object that is to be added.

        Returns:
            None
        """
        return await self._engine._run_as_async(
            self.__chat_store.async_add_message(key=key, message=message)
        )

    async def adelete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Asynchronously deletes the chat messages associated with a specific key.

        Args:
            key (str): A unique identifier for the chat whose messages are to be deleted.

        Returns:
            Optional[List[ChatMessage]]: A list of `ChatMessage` objects that were deleted, or `None` if no messages
            were associated with the key or could be deleted.
        """
        return await self._engine._run_as_async(
            self.__chat_store.adelete_messages(key=key)
        )

    async def adelete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Asynchronously deletes a specific chat message by index from the messages associated with a given key.

        Args:
            key (str): A unique identifier for the chat whose messages are to be deleted.
            idx (int): The index of the `ChatMessage` to be deleted from the list of messages.

        Returns:
            Optional[ChatMessage]: The `ChatMessage` object that was deleted, or `None` if no message
            was associated with the key or could be deleted.
        """
        return await self._engine._run_as_async(
            self.__chat_store.adelete_message(key=key, idx=idx)
        )

    async def adelete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Asynchronously deletes the last chat message associated with a given key.

        Args:
            key (str): A unique identifier for the chat whose message is to be deleted.

        Returns:
            Optional[ChatMessage]: The `ChatMessage` object that was deleted, or `None` if no message
            was associated with the key or could be deleted.
        """
        return await self._engine._run_as_async(
            self.__chat_store.adelete_last_message(key=key)
        )

    async def aget_keys(self) -> List[str]:
        """Asynchronously retrieves a list of all keys.

        Returns:
            Optional[str]: A list of strings representing the keys. If no keys are found, an empty list is returned.
        """
        return await self._engine._run_as_async(self.__chat_store.aget_keys())

    def set_messages(self, key: str, messages: List[ChatMessage]) -> None:
        """Synchronously sets the chat messages for a specific key.

        Args:
            key (str): A unique identifier for the chat.
            messages (List[ChatMessage]): A list of `ChatMessage` objects to upsert.

        Returns:
            None

        """
        return self._engine._run_as_sync(
            self.__chat_store.aset_messages(key=key, messages=messages)
        )

    def get_messages(self, key: str) -> List[ChatMessage]:
        """Synchronously retrieves the chat messages associated with a specific key.

        Args:
            key (str): A unique identifier for which the messages are to be retrieved.

        Returns:
            List[ChatMessage]: A list of `ChatMessage` objects associated with the provided key.
            If no messages are found, an empty list is returned.
        """
        return self._engine._run_as_sync(self.__chat_store.aget_messages(key=key))

    def add_message(self, key: str, message: ChatMessage) -> None:
        """Synchronously adds a new chat message to the specified key.

        Args:
            key (str): A unique identifierfor the chat to which the message is added.
            message (ChatMessage): The `ChatMessage` object that is to be added.

        Returns:
            None
        """
        return self._engine._run_as_sync(
            self.__chat_store.async_add_message(key=key, message=message)
        )

    def delete_messages(self, key: str) -> Optional[List[ChatMessage]]:
        """Synchronously deletes the chat messages associated with a specific key.

        Args:
            key (str): A unique identifier for the chat whose messages are to be deleted.

        Returns:
            Optional[List[ChatMessage]]: A list of `ChatMessage` objects that were deleted, or `None` if no messages
            were associated with the key or could be deleted.
        """
        return self._engine._run_as_sync(self.__chat_store.adelete_messages(key=key))

    def delete_message(self, key: str, idx: int) -> Optional[ChatMessage]:
        """Synchronously deletes a specific chat message by index from the messages associated with a given key.

        Args:
            key (str): A unique identifier for the chat whose messages are to be deleted.
            idx (int): The index of the `ChatMessage` to be deleted from the list of messages.

        Returns:
            Optional[ChatMessage]: The `ChatMessage` object that was deleted, or `None` if no message
            was associated with the key or could be deleted.
        """
        return self._engine._run_as_sync(
            self.__chat_store.adelete_message(key=key, idx=idx)
        )

    def delete_last_message(self, key: str) -> Optional[ChatMessage]:
        """Synchronously deletes the last chat message associated with a given key.

        Args:
            key (str): A unique identifier for the chat whose message is to be deleted.

        Returns:
            Optional[ChatMessage]: The `ChatMessage` object that was deleted, or `None` if no message
            was associated with the key or could be deleted.
        """
        return self._engine._run_as_sync(
            self.__chat_store.adelete_last_message(key=key)
        )

    def get_keys(self) -> List[str]:
        """Synchronously retrieves a list of all keys.

        Returns:
            Optional[str]: A list of strings representing the keys. If no keys are found, an empty list is returned.
        """
        return self._engine._run_as_sync(self.__chat_store.aget_keys())
