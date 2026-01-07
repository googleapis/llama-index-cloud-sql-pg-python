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

import asyncio
import os
import uuid
import warnings
from typing import Any, Coroutine, Sequence

import pytest
import pytest_asyncio
from llama_index.core.constants import DATA_KEY
from llama_index.core.schema import Document, NodeRelationship, TextNode
from sqlalchemy import RowMapping, text

from llama_index_cloud_sql_pg import PostgresEngine
from llama_index_cloud_sql_pg.async_document_store import AsyncPostgresDocumentStore

default_table_name_async = "document_store_" + str(uuid.uuid4())
custom_table_name_async = "document_store_" + str(uuid.uuid4())
sync_method_exception_str = "Sync methods are not implemented for AsyncPostgresDocumentStore. Use PostgresDocumentStore  interface instead."


# Helper to bridge the Main Test Loop and the Engine Background Loop
async def run_on_background(engine: PostgresEngine, coro: Coroutine) -> Any:
    """Runs a coroutine on the engine's background loop."""
    if engine._loop:
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, engine._loop)
        )
    return await coro


async def aexecute(engine: PostgresEngine, query: str) -> None:
    async def _impl():
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await run_on_background(engine, _impl())


async def afetch(engine: PostgresEngine, query: str) -> Sequence[RowMapping]:
    async def _impl():
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()
        return result_fetch

    result = await run_on_background(engine, _impl())
    return result


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncPostgresDocumentStore:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for Cloud SQL instance")

    @pytest.fixture(scope="module")
    def db_instance(self) -> str:
        return get_env_var("INSTANCE_ID", "instance for Cloud SQL")

    @pytest.fixture(scope="module")
    def db_name(self) -> str:
        return get_env_var("DATABASE_ID", "database name on Cloud SQL instance")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for Cloud SQL")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for Cloud SQL")

    @pytest_asyncio.fixture(scope="class")
    async def async_engine(self, db_project, db_region, db_instance, db_name):
        async_engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )

        yield async_engine

        await async_engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def doc_store(self, async_engine):
        await run_on_background(
            async_engine,
            async_engine._ainit_doc_store_table(table_name=default_table_name_async),
        )

        doc_store = await run_on_background(
            async_engine,
            AsyncPostgresDocumentStore.create(
                engine=async_engine, table_name=default_table_name_async
            ),
        )

        yield doc_store

        query = f'DROP TABLE IF EXISTS "{default_table_name_async}"'
        await aexecute(async_engine, query)

    @pytest_asyncio.fixture(scope="class")
    async def custom_doc_store(self, async_engine):
        await run_on_background(
            async_engine,
            async_engine._ainit_doc_store_table(table_name=custom_table_name_async),
        )

        custom_doc_store = await run_on_background(
            async_engine,
            AsyncPostgresDocumentStore.create(
                engine=async_engine, table_name=custom_table_name_async, batch_size=0
            ),
        )
        yield custom_doc_store

        query = f'DROP TABLE IF EXISTS "{custom_table_name_async}"'
        await aexecute(async_engine, query)

    async def test_init_with_constructor(self, async_engine):
        key = object()
        with pytest.raises(Exception):
            AsyncPostgresDocumentStore(
                key, engine=async_engine, table_name=default_table_name_async
            )

    async def test_create_without_table(self, async_engine):
        with pytest.raises(ValueError):
            await run_on_background(
                async_engine,
                AsyncPostgresDocumentStore.create(
                    engine=async_engine, table_name="non-existent-table"
                ),
            )

    async def test_warning(self, custom_doc_store):
        # Create and add documents into the docstore with batch size set to 0.
        document_text = "warning test doc"
        doc = Document(
            text=document_text, id_="warning_test_doc", metadata={"doc": "info"}
        )
        with warnings.catch_warnings(record=True) as w:
            await custom_doc_store.async_add_documents([doc], batch_size=0)

            assert len(w) == 1
            assert "Provided batch size less than 1. Defaulting to 1." in str(
                w[-1].message
            )

    async def test_adocs(self, doc_store, async_engine):
        # Create and add document into the doc store.
        document_text = "add document test"
        doc = Document(text=document_text, id_="add_doc_test", metadata={"doc": "info"})

        # Add document into the store
        await run_on_background(async_engine, doc_store.async_add_documents([doc]))

        # Assert document is found using the docs property.
        docs = await doc_store.adocs

        assert doc.doc_id in docs

    async def test_async_add_document(self, async_engine, doc_store):
        # Create and add document into the doc store.
        document_text = "add document test"
        doc = Document(text=document_text, id_="add_doc_test", metadata={"doc": "info"})

        await run_on_background(async_engine, doc_store.async_add_documents([doc]))

        # Query the table to confirm the inserted document is present.
        query = f"""select * from "public"."{default_table_name_async}" where id = '{doc.doc_id}';"""
        results = await afetch(async_engine, query)
        result = results[0]
        assert result["node_data"][DATA_KEY]["text_resource"]["text"] == document_text

    async def test_add_hash_before_data(self, async_engine, doc_store):
        # Create a document
        document_text = "add document test"
        doc = Document(text=document_text, id_="add_doc_test", metadata={"doc": "info"})

        # Insert the document id with it's doc_hash.
        await run_on_background(
            async_engine,
            doc_store.aset_document_hash(doc_id=doc.doc_id, doc_hash=doc.hash),
        )

        # Insert the document's data
        await run_on_background(async_engine, doc_store.async_add_documents([doc]))

        # Confirm the overwrite was successful.
        query = f"""select * from "public"."{default_table_name_async}" where id = '{doc.doc_id}';"""
        results = await afetch(async_engine, query)
        result = results[0]
        assert result["node_data"][DATA_KEY]["text_resource"]["text"] == document_text

    async def test_aref_doc_exists(self, doc_store, async_engine):
        # Create a ref_doc & a doc and add them to the store.
        ref_doc = Document(
            text="first doc", id_="doc_exists_doc_1", metadata={"doc": "info"}
        )
        doc = Document(
            text="second doc", id_="doc_exists_doc_2", metadata={"doc": "info"}
        )
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()

        await run_on_background(
            async_engine, doc_store.async_add_documents([ref_doc, doc])
        )

        # Confirm that ref_doc_id is recorded for the doc.
        result = await run_on_background(
            async_engine, doc_store.aref_doc_exists(ref_doc_id=ref_doc.doc_id)
        )
        assert result == True

    async def test_fetch_ref_doc_info(self, doc_store, async_engine):
        # Create a ref_doc & doc and add them to the store.
        ref_doc = Document(
            text="first doc", id_="ref_parent_doc", metadata={"doc": "info"}
        )
        doc = Document(text="second doc", id_="ref_child_doc", metadata={"doc": "info"})
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()
        await run_on_background(
            async_engine, doc_store.async_add_documents([ref_doc, doc])
        )

        # Fetch to see if ref_doc_info is found.
        result = await run_on_background(
            async_engine, doc_store.aget_ref_doc_info(ref_doc_id=ref_doc.doc_id)
        )
        assert result is not None

        # Add a new_doc with reference to doc.
        new_doc = Document(
            text="third_doc", id_="ref_new_doc", metadata={"doc": "info"}
        )
        new_doc.relationships[NodeRelationship.SOURCE] = doc.as_related_node_info()
        await run_on_background(async_engine, doc_store.async_add_documents([new_doc]))

        # Fetch to see if ref_doc_info is found for both ref_doc and doc.
        results = await run_on_background(
            async_engine, doc_store.aget_all_ref_doc_info()
        )
        assert ref_doc.doc_id in results
        assert doc.doc_id in results

    async def test_adelete_ref_doc(self, doc_store, async_engine):
        # Create a ref_doc & doc and add them to the store.
        ref_doc = Document(
            text="first doc", id_="ref_parent_doc", metadata={"doc": "info"}
        )
        doc = Document(text="second doc", id_="ref_child_doc", metadata={"doc": "info"})
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()
        await run_on_background(
            async_engine, doc_store.async_add_documents([ref_doc, doc])
        )

        # Delete the reference doc
        await run_on_background(
            async_engine, doc_store.adelete_ref_doc(ref_doc_id=ref_doc.doc_id)
        )

        # Confirm the reference doc along with it's child nodes are deleted.
        assert (
            await run_on_background(
                async_engine,
                doc_store.aget_document(doc_id=doc.doc_id, raise_error=False),
            )
            is None
        )
        # Confirm deleting an non-existent reference doc returns None.
        assert (
            await run_on_background(
                async_engine, doc_store.adelete_ref_doc(ref_doc_id=ref_doc.doc_id)
            )
            is None
        )

    async def test_set_and_get_document_hash(self, doc_store, async_engine):
        # Set a doc hash for a document
        doc_id = "document_id"
        doc_hash = "document_hash"
        await run_on_background(
            async_engine, doc_store.aset_document_hash(doc_id=doc_id, doc_hash=doc_hash)
        )

        # Assert with get that the hash is same as the one set.
        assert (
            await run_on_background(
                async_engine, doc_store.aget_document_hash(doc_id=doc_id)
            )
            == doc_hash
        )

    async def test_aget_document_hash(self, doc_store, async_engine):
        assert (
            await run_on_background(
                async_engine, doc_store.aget_document_hash(doc_id="non-existent-doc")
            )
            is None
        )

    async def test_set_and_get_document_hashes(self, doc_store, async_engine):
        # Create a dictionary of doc_id -> doc_hash mappings and add it to the table.
        document_dict = {
            "document one": "document one hash",
            "document two": "document two hash",
        }
        expected_dict = {v: k for k, v in document_dict.items()}
        await run_on_background(
            async_engine, doc_store.aset_document_hashes(doc_hashes=document_dict)
        )

        # Get all the doc hashes and assert it is same as the one set.
        results = await run_on_background(
            async_engine, doc_store.aget_all_document_hashes()
        )
        assert "document one hash" in results
        assert "document two hash" in results
        assert results["document one hash"] == expected_dict["document one hash"]
        assert results["document two hash"] == expected_dict["document two hash"]

    async def test_doc_store_basic(self, doc_store, async_engine):
        # Create a doc and a node and add them to the store.
        doc = Document(text="document_1", id_="doc_id_1", metadata={"doc": "info"})
        node = TextNode(text="node_1", id_="node_id_1", metadata={"node": "info"})

        await run_on_background(
            async_engine, doc_store.async_add_documents([doc, node])
        )

        # Assert if document exists
        assert (
            await run_on_background(
                async_engine, doc_store.adocument_exists(doc_id=doc.doc_id)
            )
            == True
        )

        # Assert if retrieved doc is the same as the one inserted.
        retrieved_doc = await run_on_background(
            async_engine, doc_store.aget_document(doc_id=doc.doc_id)
        )
        assert retrieved_doc == doc

        # Assert if retrieved node is the same as the one inserted.
        retrieved_node = await run_on_background(
            async_engine, doc_store.aget_document(doc_id=node.node_id)
        )
        assert retrieved_node == node

    async def test_adelete_document(self, async_engine, doc_store):
        # Create a doc and add it to the store.
        doc = Document(text="document_2", id_="doc_id_2", metadata={"doc": "info"})
        await run_on_background(async_engine, doc_store.async_add_documents([doc]))

        # Delete the document from the store.
        await run_on_background(
            async_engine, doc_store.adelete_document(doc_id=doc.doc_id)
        )

        # Assert the document is deleted by querying the table.
        query = f"""select * from "public"."{default_table_name_async}" where id = '{doc.doc_id}';"""
        result = await afetch(async_engine, query)
        assert len(result) == 0

    async def test_delete_non_existent_document(self, doc_store, async_engine):
        await run_on_background(
            async_engine,
            doc_store.adelete_document(doc_id="non-existent-doc", raise_error=False),
        )
        with pytest.raises(ValueError):
            await run_on_background(
                async_engine, doc_store.adelete_document(doc_id="non-existent-doc")
            )

    async def test_doc_store_ref_doc_not_added(self, async_engine, doc_store):
        # Create a ref_doc & doc.
        ref_doc = Document(
            text="first doc", id_="doc_id_parent", metadata={"doc": "info"}
        )
        doc = Document(text="second doc", id_="doc_id_child", metadata={"doc": "info"})
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()

        # Insert only the node into the document store.
        await run_on_background(async_engine, doc_store.async_add_documents([doc]))

        query = f"""select id as node_ids from "public"."{default_table_name_async}" where ref_doc_id = '{ref_doc.doc_id}';"""
        result = await afetch(async_engine, query)

        # Assert document has been added
        assert len(result) != 0

        # Delete the document
        await run_on_background(
            async_engine, doc_store.adelete_ref_doc(ref_doc_id=ref_doc.doc_id)
        )

        # Assert if parent doc is deleted
        query = f"""select * from "public"."{default_table_name_async}" where id = '{ref_doc.doc_id}';"""
        result = await afetch(async_engine, query)
        assert len(result) == 0

        # Assert if child (related) doc is deleted
        query = f"""select * from "public"."{default_table_name_async}" where id = '{doc.doc_id}';"""
        result = await afetch(async_engine, query)
        assert len(result) == 0

    async def test_doc_store_delete_all_ref_doc_nodes(self, async_engine, doc_store):
        # Create a ref_doc, which is the parent doc for a doc and a node.
        ref_doc = Document(text="document", id_="parent_doc", metadata={"doc1": "info"})
        doc = Document(text="document", id_="child_doc", metadata={"doc": "info"})
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()
        node = TextNode(
            text="node", id_="child_node", metadata={"doc": "from_node_info"}
        )
        node.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()

        # Add all the structures into the store.
        await run_on_background(
            async_engine, doc_store.async_add_documents([ref_doc, doc, node])
        )

        query = f"""select id as node_ids from "public"."{default_table_name_async}" where ref_doc_id = '{ref_doc.doc_id}';"""
        result = await afetch(async_engine, query)
        result = {"node_ids": [item["node_ids"] for item in result]}

        # Assert the ref_doc has mappings to both the child doc and child node
        assert result["node_ids"] == [
            doc.doc_id,
            node.node_id,
        ]

        # Delete the child document.
        await run_on_background(async_engine, doc_store.adelete_document(doc.doc_id))

        # Assert the ref_doc still  exists.
        query = f"""select * from "public"."{default_table_name_async}" where id = '{ref_doc.doc_id}';"""
        result = await afetch(async_engine, query)
        assert len(result) != 0

        # Assert the ref_doc still has a mapping to the child_node.
        query = f"""select id as node_ids from "public"."{default_table_name_async}" where ref_doc_id = '{ref_doc.doc_id}';"""
        result = await afetch(async_engine, query)
        result = {"node_ids": [item["node_ids"] for item in result]}

        assert result["node_ids"] == [node.node_id]

        # Delete the child node
        await run_on_background(async_engine, doc_store.adelete_document(node.node_id))

        # Assert the ref_doc is also deleted from the store.
        query = f"""select * from "public"."{default_table_name_async}" where id = '{ref_doc.doc_id}';"""
        result = await afetch(async_engine, query)
        assert len(result) == 0

    async def test_docs(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.docs()

    async def test_add_documents(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.add_documents([])

    async def test_get_document(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.get_document("test_doc_id", raise_error=True)
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.get_document("test_doc_id", raise_error=False)

    async def test_delete_document(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.delete_document("test_doc_id", raise_error=True)
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.delete_document("test_doc_id", raise_error=False)

    async def test_document_exists(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.document_exists("test_doc_id")

    async def test_ref_doc_exists(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.ref_doc_exists(ref_doc_id="test_ref_doc_id")

    async def test_set_document_hash(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.set_document_hash("test_doc_id", "test_doc_hash")

    async def test_set_document_hashes(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.set_document_hashes({"test_doc_id": "test_doc_hash"})

    async def test_get_document_hash(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.get_document_hash(doc_id="test_doc_id")

    async def test_get_all_document_hashes(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.get_all_document_hashes()

    async def test_get_all_ref_doc_info(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.get_all_ref_doc_info()

    async def test_get_ref_doc_info(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.get_ref_doc_info(ref_doc_id="test_doc_id")

    async def test_delete_ref_doc(self, doc_store):
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.delete_ref_doc(ref_doc_id="test_doc_id", raise_error=False)
        with pytest.raises(Exception, match=sync_method_exception_str):
            doc_store.delete_ref_doc(ref_doc_id="test_doc_id", raise_error=True)
