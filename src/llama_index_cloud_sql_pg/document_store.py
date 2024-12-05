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

from typing import Optional, Sequence

from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import BaseDocumentStore
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.kvstore.types import DEFAULT_BATCH_SIZE

from .async_document_store import AsyncPostgresDocumentStore
from .engine import PostgresEngine


class PostgresDocumentStore(BaseDocumentStore):
    """Document Store Table stored in an Cloud SQL for PostgreSQL database."""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: PostgresEngine,
        document_store: AsyncPostgresDocumentStore,
    ):
        """ "PostgresDocumentStore constructor.

        Args:
            key (object): Key to prevent direct constructor usage.
            engine (PostgresEngine): Database connection pool.
            document_store (AsyncPostgresDocumentStore): The async only DocumentStore implementation

        Raises:
            Exception: If constructor is directly called by the user.
        """
        if key != PostgresDocumentStore.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._engine = engine
        self.__document_store = document_store

    @classmethod
    async def create(
        cls: type[PostgresDocumentStore],
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> PostgresDocumentStore:
        """Create a new PostgresDocumentStore instance.

        Args:
            engine (PostgresEngine): Postgres engine to use.
            table_name (str): Table name that stores the documents.
            schema_name (str): The schema name where the table is located. Defaults to "public"
            batch_size (str): The default batch size for bulk inserts. Defaults to 1.

        Raises:
            ValueError: If the table provided does not contain required schema.

        Returns:
            PostgresDocumentStore: A newly created instance of PostgresDocumentStore.
        """
        coro = AsyncPostgresDocumentStore.create(
            engine, table_name, schema_name, batch_size
        )
        document_store = await engine._run_as_async(coro)
        return cls(cls.__create_key, engine, document_store)

    @classmethod
    def create_sync(
        cls: type[PostgresDocumentStore],
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> PostgresDocumentStore:
        """Create a new PostgresDocumentStore sync instance.

        Args:
            engine (PostgresEngine): Postgres engine to use.
            table_name (str): Table name that stores the documents.
            schema_name (str): The schema name where the table is located. Defaults to "public"
            batch_size (str): The default batch size for bulk inserts. Defaults to 1.

        Raises:
            ValueError: If the table provided does not contain required schema.

        Returns:
            PostgresDocumentStore: A newly created instance of PostgresDocumentStore.
        """
        coro = AsyncPostgresDocumentStore.create(
            engine, table_name, schema_name, batch_size
        )
        document_store = engine._run_as_sync(coro)
        return cls(cls.__create_key, engine, document_store)

    @property
    def docs(self) -> dict[str, BaseNode]:
        """Get all documents.

        Returns:
            dict[str, BaseDocument]: documents

        """
        return self._engine._run_as_sync(self.__document_store.adocs)

    async def async_add_documents(
        self,
        docs: Sequence[BaseNode],
        allow_update: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        store_text: bool = True,
    ) -> None:
        """Adds a document to the store.

        Args:
            docs (Sequence[BaseDocument]): documents
            allow_update (bool): allow update of docstore from document
            batch_size (int): batch_size to insert the rows. Defaults to 1.
            store_text (bool): allow the text content of the node to stored. Defaults to "True".

        Returns:
            None
        """
        return await self._engine._run_as_async(
            self.__document_store.async_add_documents(
                docs, allow_update, batch_size, store_text
            )
        )

    def add_documents(
        self,
        docs: Sequence[BaseNode],
        allow_update: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        store_text: bool = True,
    ) -> None:
        """Adds a document to the store.

        Args:
            docs (Sequence[BaseDocument]): documents
            allow_update (bool): allow update of docstore from document
            batch_size (int): batch_size to insert the rows. Defaults to 1.
            store_text (bool): allow the text content of the node to stored. Defaults to "True".

        Returns:
            None
        """
        return self._engine._run_as_sync(
            self.__document_store.async_add_documents(
                docs, allow_update, batch_size, store_text
            )
        )

    async def aget_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseNode]:
        """Retrieves a document from the table by its doc_id.

        Args:
            doc_id (str): Id of the document / node to be retrieved.
            raise_error (bool): to raise error if document is not found.

        Raises:
            ValueError: If a node doesn't exist and `raise_error` is set to True.

        Returns:
            Optional[BaseNode]: Returns a `BaseNode` object if the document is found
        """
        result = await self._engine._run_as_async(
            self.__document_store.aget_document(doc_id, raise_error)
        )
        return result

    def get_document(self, doc_id: str, raise_error: bool = True) -> Optional[BaseNode]:
        """Retrieves a document from the table by its doc_id.

        Args:
            doc_id (str): Id of the document / node to be retrieved.
            raise_error (bool): to raise error if document is not found.

        Raises:
            ValueError: If a node doesn't exist and `raise_error` is set to True.

        Returns:
            Optional[BaseNode]: Returns a `BaseNode` object if the document is found
        """
        result = self._engine._run_as_sync(
            self.__document_store.aget_document(doc_id, raise_error)
        )
        return result

    async def adelete_document(self, doc_id: str, raise_error: bool = True) -> None:
        """Delete a document from the store.

        Args:
            doc_id (str): Id of the document / node to be deleted.
            raise_error (bool): to raise error if document is not found.

        Returns:
            None

        Raises:
            ValueError: If a node is not found and `raise_error` is set to True.
        """
        return await self._engine._run_as_async(
            self.__document_store.adelete_document(doc_id, raise_error)
        )

    def delete_document(self, doc_id: str, raise_error: bool = True) -> None:
        """Delete a document from the store.

        Args:
            doc_id (str): Id of the document / node to be deleted.
            raise_error (bool): to raise error if document is not found.

        Returns:
            None

        Raises:
            ValueError: If a node is not found and `raise_error` is set to True.
        """
        return self._engine._run_as_sync(
            self.__document_store.adelete_document(doc_id, raise_error)
        )

    async def adocument_exists(self, doc_id: str) -> bool:
        """Check if document exists.

        Args:
            doc_id (str): The document / node id which needs to be found.

        Returns:
            bool : True if document exists in the table.
        """
        return await self._engine._run_as_async(
            self.__document_store.adocument_exists(doc_id)
        )

    def document_exists(self, doc_id: str) -> bool:
        """Check if document exists.

        Args:
            doc_id (str): The document / node id which needs to be found.

        Returns:
            bool : True if document exists in the table.
        """
        return self._engine._run_as_sync(self.__document_store.adocument_exists(doc_id))

    async def aset_document_hash(self, doc_id: str, doc_hash: str) -> None:
        """Set the hash for a given doc_id.

        Args:
            doc_id (str): Id to be updated with the doc_hash.
            doc_hash (str): Doc_hash to be updated into the table.

        Returns:
            None
        """
        return await self._engine._run_as_async(
            self.__document_store.aset_document_hash(doc_id, doc_hash)
        )

    def set_document_hash(self, doc_id: str, doc_hash: str) -> None:
        """Set the hash for a given doc_id.

        Args:
            doc_id (str): Id to be updated with the doc_hash.
            doc_hash (str): Doc_hash to be updated into the table.

        Returns:
            None
        """
        return self._engine._run_as_sync(
            self.__document_store.aset_document_hash(doc_id, doc_hash)
        )

    async def aset_document_hashes(self, doc_hashes: dict[str, str]) -> None:
        """Set the hash for a given doc_id.

        Args:
            doc_hashes (dict[str, str]): Dictionary with doc_id as key and doc_hash as value.

        Returns:
            None
        """
        return await self._engine._run_as_async(
            self.__document_store.aset_document_hashes(doc_hashes)
        )

    def set_document_hashes(self, doc_hashes: dict[str, str]) -> None:
        """Set the hash for a given doc_id.

        Args:
            doc_hashes (dict[str, str]): Dictionary with doc_id as key and doc_hash as value.

        Returns:
            None
        """
        return self._engine._run_as_sync(
            self.__document_store.aset_document_hashes(doc_hashes)
        )

    async def aget_document_hash(self, doc_id: str) -> Optional[str]:
        """Get the stored hash for a document, if it exists.

        Args:
          doc_id: Node / Document id whose hash is to be retrieved.

        Returns:
            Optional[str]: The hash for the given doc_id, if available.
        """
        return await self._engine._run_as_async(
            self.__document_store.aget_document_hash(doc_id)
        )

    def get_document_hash(self, doc_id: str) -> Optional[str]:
        """Get the stored hash for a document, if it exists.

        Args:
          doc_id: Node / Document id whose hash is to be retrieved.

        Returns:
            Optional[str]: The hash for the given doc_id, if available.
        """
        return self._engine._run_as_sync(
            self.__document_store.aget_document_hash(doc_id)
        )

    async def aget_all_document_hashes(self) -> dict[str, str]:
        """Get the stored hash for all documents.

        Returns:
            dict[
              str,   # doc_hash
              str    # doc_id
            ]
        """
        return await self._engine._run_as_async(
            self.__document_store.aget_all_document_hashes()
        )

    def get_all_document_hashes(self) -> dict[str, str]:
        """Get the stored hash for all documents.

        Returns:
            dict[
              str,   # doc_hash
              str    # doc_id
            ]
        """
        return self._engine._run_as_sync(
            self.__document_store.aget_all_document_hashes()
        )

    async def aget_all_ref_doc_info(self) -> Optional[dict[str, RefDocInfo]]:
        """Get a mapping of ref_doc_id -> RefDocInfo for all ingested documents.

        Returns:
            Optional[
              dict[
                str,          #Ref_doc_id
                RefDocInfo,   #Ref_doc_info of the id
              ]
            ]
        """
        return await self._engine._run_as_async(
            self.__document_store.aget_all_ref_doc_info()
        )

    def get_all_ref_doc_info(self) -> Optional[dict[str, RefDocInfo]]:
        """Get a mapping of ref_doc_id -> RefDocInfo for all ingested documents.

        Returns:
            Optional[
              dict[
                str,          #Ref_doc_id
                RefDocInfo,   #Ref_doc_info of the id
              ]
            ]
        """
        return self._engine._run_as_sync(self.__document_store.aget_all_ref_doc_info())

    async def aget_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get the RefDocInfo for a given ref_doc_id.

        Args:
            ref_doc_id (str): The ref_doc_id whose ref_doc_info is to be retrieved.

        Returns:
            Optional[RefDocInfo]: Returns a `RefDocInfo` object if it exists.
        """
        return await self._engine._run_as_async(
            self.__document_store.aget_ref_doc_info(ref_doc_id)
        )

    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get the RefDocInfo for a given ref_doc_id.

        Args:
            ref_doc_id (str): The ref_doc_id whose ref_doc_info is to be retrieved.

        Returns:
            Optional[RefDocInfo]: Returns a `RefDocInfo` object if it exists.
        """
        return self._engine._run_as_sync(
            self.__document_store.aget_ref_doc_info(ref_doc_id)
        )

    async def aref_doc_exists(self, ref_doc_id: str) -> bool:
        """Check if a ref_doc_id has been ingested.

        Args:
            ref_doc_id (str): The ref_doc_id whose ref_doc_info is to be found.

        Returns:
            bool : True if document exists as a ref doc in the table.
        """
        return await self._engine._run_as_async(
            self.__document_store.aref_doc_exists(ref_doc_id)
        )

    def ref_doc_exists(self, ref_doc_id: str) -> bool:
        """Check if a ref_doc_id has been ingested.

        Args:
            ref_doc_id (str): The ref_doc_id whose ref_doc_info is to be found.

        Returns:
            bool : True if document exists as a ref doc in the table.
        """
        return self._engine._run_as_sync(
            self.__document_store.aref_doc_exists(ref_doc_id)
        )

    async def adelete_ref_doc(self, ref_doc_id: str, raise_error: bool = True) -> None:
        """Delete a ref_doc and all it's associated nodes.

        Args:
            ref_doc_id (str): Ref_doc_id which needs to be deleted.
            raise_error (bool): to raise error if ref_doc_info for the ref_doc_id is not found.

        Returns:
            None

        Raises:
            ValueError: If ref_doc_info for the ref_doc_id doesn't exist and `raise_error` is set to True.
        """
        return await self._engine._run_as_async(
            self.__document_store.adelete_ref_doc(ref_doc_id, raise_error)
        )

    def delete_ref_doc(self, ref_doc_id: str, raise_error: bool = True) -> None:
        """Delete a ref_doc and all it's associated nodes.

        Args:
            ref_doc_id (str): Ref_doc_id which needs to be deleted.
            raise_error (bool): to raise error if ref_doc_info for the ref_doc_id is not found.

        Returns:
            None

        Raises:
            ValueError: If ref_doc_info for the ref_doc_id doesn't exist and `raise_error` is set to True.
        """
        return self._engine._run_as_sync(
            self.__document_store.adelete_ref_doc(ref_doc_id, raise_error)
        )
