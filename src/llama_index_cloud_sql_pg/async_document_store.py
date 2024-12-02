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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llama_index.core.constants import DATA_KEY
from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore import BaseDocumentStore
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.docstore.utils import doc_to_json, json_to_doc
from llama_index.core.storage.kvstore.types import DEFAULT_BATCH_SIZE
from sqlalchemy import RowMapping, text
from sqlalchemy.ext.asyncio import AsyncEngine

from .engine import PostgresEngine


class AsyncPostgresDocumentStore(BaseDocumentStore):
    """Document Store Table stored in an Cloud SQL for PostgreSQL database."""

    __create_key = object()

    def __init__(
        self,
        key: object,
        engine: AsyncEngine,
        table_name: str,
        schema_name: str = "public",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """AsyncPostgresDocumentStore constructor.

        Args:
            key (object): Key to prevent direct constructor usage.
            engine (PostgresEngine): Database connection pool.
            table_name (str): Table name that stores the documents.
            schema_name (str): The schema name where the table is located. Defaults to "public"
            batch_size (int): The default batch size for bulk inserts. Defaults to 1.

        Raises:
            Exception: If constructor is directly called by the user.
        """
        if key != AsyncPostgresDocumentStore.__create_key:
            raise Exception("Only create class through 'create' method!")
        self._engine = engine
        self._table_name = table_name
        self._schema_name = schema_name
        self._batch_size = batch_size

    @classmethod
    async def create(
        cls,
        engine: PostgresEngine,
        table_name: str,
        schema_name: str = "public",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> AsyncPostgresDocumentStore:
        """Create a new AsyncPostgresDocumentStore instance.

        Args:
            engine (PostgresEngine): Postgres engine to use.
            table_name (str): Table name that stores the documents.
            schema_name (str): The schema name where the table is located. Defaults to "public"
            batch_size (int): The default batch size for bulk inserts. Defaults to 1.

        Raises:
            ValueError: If the table provided does not contain required schema.

        Returns:
            AsyncPostgresDocumentStore: A newly created instance of AsyncPostgresDocumentStore.
        """
        table_schema = await engine._aload_table_schema(table_name, schema_name)
        column_names = table_schema.columns.keys()

        required_columns = ["id", "doc_hash", "ref_doc_id", "node_data"]

        if not (all(x in column_names for x in required_columns)):
            raise ValueError(
                f"Table '{schema_name}'.'{table_name}' has incorrect schema. Got "
                f"column names '{column_names}' but required column names "
                f"'{required_columns}'.\nPlease create table with following schema:"
                f"CREATE TABLE {schema_name}.{table_name} ("
                "    id VARCHAR PRIMARY KEY,"
                "    doc_hash VARCHAR NOT NULL,"
                "    ref_doc_id VARCHAR,"
                "    node_data JSONB NOT NULL"
                ");"
            )

        return cls(cls.__create_key, engine._pool, table_name, schema_name, batch_size)

    async def __aexecute_query(self, query, params):
        async with self._engine.connect() as conn:
            await conn.execute(text(query), params)
            await conn.commit()
        return None

    async def __afetch_query(self, query):
        async with self._engine.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            results = result_map.fetchall()
            await conn.commit()
        return results

    async def _put_all_doc_hashes_to_table(
        self, rows: List[Tuple[str, str]], batch_size: int = int(DEFAULT_BATCH_SIZE)
    ) -> None:
        """Puts a multiple rows of node ids with their doc_hash into the document table.
        Incase a row with the id already exists, it updates the row with the new doc_hash.

        Args:
            rows (List[Tuple[str, str]]): List of tuples of id and doc_hash
            batch_size (int): batch_size to insert the rows. Defaults to 1.

        Returns:
            None
        """
        if batch_size < 1:
            batch_size = 1
            warnings.warn("Provided batch size less than 1. Defaulting to 1.")

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            params = [{"id": id, "doc_hash": doc_hash} for id, doc_hash in batch]

            # Insert statement
            stmt = f"""
              INSERT INTO "{self._schema_name}"."{self._table_name}" (id, doc_hash)
              VALUES (:id, :doc_hash)
              ON CONFLICT (id)
              DO UPDATE SET
              doc_hash = EXCLUDED.doc_hash;
              """

            await self.__aexecute_query(stmt, params)

    async def _delete_from_table(self, id: str) -> Sequence[RowMapping]:
        """Delete a value from the store.

        Args:
            id (str): id to be deleted

        Returns:
            List of deleted rows.
        """
        query = f"""DELETE FROM "{self._schema_name}"."{self._table_name}" WHERE id = '{id}' RETURNING *; """
        result = await self.__afetch_query(query)
        return result

    async def async_add_documents(
        self,
        docs: Sequence[BaseNode],
        allow_update: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        store_text: bool = True,
    ) -> None:
        """Adds a document to the store.

        Args:
            docs (List[BaseDocument]): documents
            allow_update (bool): allow update of docstore from document
            batch_size (int): batch_size to insert the rows. Defaults to 1.
            store_text (bool): allow the text content of the node to stored.

        Returns:
            None
        """
        batch_size = batch_size or self._batch_size

        if batch_size < 1:
            batch_size = 1
            warnings.warn("Provided batch size less than 1. Defaulting to 1.")

        node_rows = []

        for node in docs:
            # NOTE: doc could already exist in the store, but we overwrite it
            if not allow_update and await self.adocument_exists(node.node_id):
                raise ValueError(
                    f"node_id {node.node_id} already exists. "
                    "Set allow_update to True to overwrite."
                )

            id = node.node_id
            data = doc_to_json(node)

            if store_text:
                node_data = data
            ref_doc_id = node.ref_doc_id
            doc_hash = node.hash

            node_row = {
                "id": id,
                "doc_hash": doc_hash,
                "ref_doc_id": ref_doc_id,
                "node_data": json.dumps(node_data),
            }

            node_rows.append(node_row)

        for i in range(0, len(node_rows), batch_size):
            batch = node_rows[i : i + batch_size]

            insert_query = f'INSERT INTO "{self._schema_name}"."{self._table_name}"(id, doc_hash, ref_doc_id, node_data) '
            values_statement = f"VALUES (:id, :doc_hash, :ref_doc_id, :node_data)"
            upsert_statement = " ON CONFLICT (id) DO UPDATE SET node_data = EXCLUDED.node_data, ref_doc_id = EXCLUDED.ref_doc_id, doc_hash = EXCLUDED.doc_hash;"

            query = insert_query + values_statement + upsert_statement
            await self.__aexecute_query(query, batch)

    @property
    async def adocs(self) -> Dict[str, BaseNode]:
        """Get all documents.

        Returns:
            Dict[str, BaseDocument]: documents
        """
        query = f"""SELECT * from "{self._schema_name}"."{self._table_name}";"""
        list_docs = await self.__afetch_query(query)

        if list_docs is None:
            return {}

        return {doc["id"]: json_to_doc(doc["node_data"]) for doc in list_docs}

    async def aget_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseNode]:
        """Asynchronously retrieves a document from the table by its doc_id.

        Args:
            doc_id (str): Id of the document / node to be retrieved.
            raise_error (bool): to raise error if document is not found.

        Raises:
            ValueError: If a node doesn't exist and `raise_error` is set to True.

        Returns:
            Optional[BaseNode]: Returns a `BaseNode` object if the document is found
        """
        query = f"""SELECT node_data from "{self._schema_name}"."{self._table_name}" WHERE id = '{doc_id}';"""
        result = await self.__afetch_query(query)

        if result:
            result = result[0]
            json = result.get("node_data")
            if json is not None:
                return json_to_doc(json)
        if raise_error:
            raise ValueError(f"doc_id {doc_id} not found.")
        else:
            return None

    async def aget_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get the RefDocInfo for a given ref_doc_id.

        Args:
            ref_doc_id (str): The ref_doc_id whose ref_doc_info is to be retrieved.

        Returns:
            Optional[RefDocInfo]: Returns a `RefDocInfo` object if it exists.
        """
        query = f"""select id, node_data from "{self._schema_name}"."{self._table_name}" where ref_doc_id = '{ref_doc_id}'"""

        rows = await self.__afetch_query(query)
        node_ids = []
        merged_metadata = {}

        if not rows:
            return None

        for row in rows:
            id = row.get("id")
            node_data = row.get("node_data")

            # Extract node_id and add it to the list
            node_ids.append(id)

            # Extract and merge metadata
            metadata = node_data.get(DATA_KEY).get("metadata", {})
            for key, value in metadata.items():
                # Upsert logic: if key exists, the value will be overwritten
                merged_metadata[key] = value

        return RefDocInfo(node_ids=node_ids, metadata=merged_metadata)

    async def aget_all_ref_doc_info(self) -> Optional[Dict[str, RefDocInfo]]:
        """Get a mapping of ref_doc_id -> RefDocInfo for all ingested documents.

        Returns:
            Optional[
              Dict[
                str,          #Ref_doc_id
                RefDocInfo,   #Ref_doc_info of the id
              ]
            ]
        """

        ref_doc_infos = {}
        query = f"""SELECT distinct on (ref_doc_id) ref_doc_id from "{self._schema_name}"."{self._table_name}";"""
        ref_doc_ids = await self.__afetch_query(query)

        if ref_doc_ids is None:
            return None

        for id in ref_doc_ids:
            ref_doc_info = await self.aget_ref_doc_info(id["ref_doc_id"])
            if ref_doc_info is not None:
                ref_doc_infos[id["ref_doc_id"]] = ref_doc_info

        all_ref_doc_infos = {}
        for doc_id, ref_doc_info in ref_doc_infos.items():
            all_ref_doc_infos[doc_id] = ref_doc_info
        return all_ref_doc_infos

    # This method is not part of the base class, but it has been introduced for the user's convenience.
    async def aref_doc_exists(self, ref_doc_id: str) -> bool:
        """Check if a ref_doc_id has been ingested.

        Args:
            ref_doc_id (str): The ref_doc_id whose ref_doc_info is to be found.

        Returns:
            bool : True if document exists as a ref doc in the table.
        """
        return bool(await self._get_ref_doc_child_node_ids(ref_doc_id))

    async def adocument_exists(self, doc_id: str) -> bool:
        """Check if document exists.

        Args:
            doc_id (str): The document / node id which needs to be found.

        Returns:
            bool : True if document exists in the table.
        """
        query = f"""SELECT id from "{self._schema_name}"."{self._table_name}" WHERE id = '{doc_id}' LIMIT 1;"""
        result = await self.__afetch_query(query)
        return bool(result)

    async def _get_ref_doc_child_node_ids(
        self, ref_doc_id: str
    ) -> Optional[Dict[str, List[str]]]:
        """Helper function to find the child node mappings of a ref_doc_id.

        Returns:
            Optional[
              Dict[
                str,    # Ref_doc_id
                List    # List of all nodes that refer to ref_doc_id
              ]
            ]"""
        query = f"""select id from "{self._schema_name}"."{self._table_name}" where ref_doc_id = '{ref_doc_id}';"""
        results = await self.__afetch_query(query)
        result = {"node_ids": [item["id"] for item in results]}
        return result

    async def adelete_document(self, doc_id: str, raise_error: bool = True) -> None:
        """Delete a document from the store but preserve its child nodes.

        Args:
            doc_id (str): Id of the document / node to be deleted.
            raise_error (bool): to raise error if document is not found.

        Returns:
            None

        Raises:
            ValueError: If a node is not found and `raise_error` is set to True.
        """

        deleted_docs = await self._delete_from_table(doc_id)
        if not deleted_docs and raise_error:
            raise ValueError(f"doc_id {doc_id} not found.")

        if deleted_docs:
            ref_doc_id = deleted_docs[0].get("ref_doc_id")

            if ref_doc_id:
                results = await self._get_ref_doc_child_node_ids(ref_doc_id)

                if results and not results.get("node_ids"):
                    await self._delete_from_table(ref_doc_id)

        return None

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

        child_node_ids = await self._get_ref_doc_child_node_ids(ref_doc_id)

        if child_node_ids is None:
            if raise_error:
                raise ValueError(f"ref_doc_id {ref_doc_id} not found.")
            else:
                return

        query = f"""DELETE FROM "{self._schema_name}"."{self._table_name}" WHERE ref_doc_id = :ref_doc_id;"""
        await self.__aexecute_query(query, {"ref_doc_id": ref_doc_id})

        await self._delete_from_table(ref_doc_id)

        return None

    async def aset_document_hash(self, doc_id: str, doc_hash: str) -> None:
        """Set the hash for a given doc_id.

        Args:
            doc_id (str): Id to be updated with the doc_hash.
            doc_hash (str): Doc_hash to be updated into the table.

        Returns:
            None
        """

        await self._put_all_doc_hashes_to_table(rows=[(doc_id, doc_hash)])

    async def aset_document_hashes(self, doc_hashes: Dict[str, str]) -> None:
        """Set the hash for a given doc_id.

        Args:
            doc_hashes (Dict[str, str]): Dictionary with doc_id as key and doc_hash as value.

        Returns:
            None
        """
        doc_hash_pairs = []
        for doc_id, doc_hash in doc_hashes.items():
            doc_hash_pairs.append((doc_id, doc_hash))

        await self._put_all_doc_hashes_to_table(doc_hash_pairs)

    async def aget_document_hash(self, doc_id: str) -> Optional[str]:
        """Get the stored hash for a document, if it exists.

        Returns:
            Optional[
              str   # hash for the given doc_id
            ]
        """
        query = f"""SELECT id, doc_hash from "{self._schema_name}"."{self._table_name}" WHERE id = '{doc_id}' LIMIT 1;"""
        row = await self.__afetch_query(query)

        if row:
            return row[0].get("doc_hash", None)
        else:
            return None

    async def aget_all_document_hashes(self) -> Dict[str, str]:
        """Get the stored hash for all documents.

        Returns:
            Dict[
              str,   # doc_hash
              str    # doc_id
            ]
        """
        hashes = {}

        query = (
            f"""SELECT id, doc_hash from "{self._schema_name}"."{self._table_name}";"""
        )
        rows = await self.__afetch_query(query)

        if rows:
            for row in rows:
                doc_hash = str(row.get("doc_hash"))
                doc_id = str(row.get("id"))
                if doc_hash:
                    hashes[doc_hash] = doc_id
        return hashes

    @property
    def docs(self) -> Dict[str, BaseNode]:
        """Get all documents.

        Returns:
            Dict[str, BaseDocument]: documents

        """
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )

    def add_documents(
        self,
        docs: Sequence[BaseNode],
        allow_update: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        store_text: bool = True,
    ) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )

    def get_document(self, doc_id: str, raise_error: bool = True) -> Optional[BaseNode]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )

    def delete_document(self, doc_id: str, raise_error: bool = True) -> None:
        """Delete a document from the store."""
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )

    def document_exists(self, doc_id: str) -> bool:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )

    # This method is not part of the base class, but it has been introduced for the user's convenience.
    def ref_doc_exists(self, ref_doc_id: str) -> bool:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )

    def set_document_hash(self, doc_id: str, doc_hash: str) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )

    def set_document_hashes(self, doc_hashes: Dict[str, str]) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )

    def get_document_hash(self, doc_id: str) -> Optional[str]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )

    def get_all_document_hashes(self) -> Dict[str, str]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )

    def get_all_ref_doc_info(self) -> Optional[Dict[str, RefDocInfo]]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )

    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )

    def delete_ref_doc(self, ref_doc_id: str, raise_error: bool = True) -> None:
        raise NotImplementedError(
            "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."
        )
