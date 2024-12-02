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

import os
import uuid
from typing import Sequence

import pytest
import pytest_asyncio
from llama_index.core.constants import DATA_KEY
from llama_index.core.schema import Document, NodeRelationship, TextNode
from sqlalchemy import RowMapping, text

from llama_index_cloud_sql_pg import PostgresDocumentStore, PostgresEngine

default_table_name_async = "document_store_" + str(uuid.uuid4())
default_table_name_sync = "document_store_" + str(uuid.uuid4())


async def aexecute(
    engine: PostgresEngine,
    query: str,
) -> None:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await engine._run_as_async(run(engine, query))


async def afetch(engine: PostgresEngine, query: str) -> Sequence[RowMapping]:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()
        return result_fetch

    return await engine._run_as_async(run(engine, query))


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio(loop_scope="class")
class TestPostgresDocumentStoreAsync:
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
    async def async_engine(
        self,
        db_project,
        db_region,
        db_instance,
        db_name,
    ):
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
        await async_engine.ainit_doc_store_table(table_name=default_table_name_async)

        doc_store = await PostgresDocumentStore.create(
            engine=async_engine, table_name=default_table_name_async
        )

        yield doc_store

        query = f'DROP TABLE IF EXISTS "{default_table_name_async}"'
        await aexecute(async_engine, query)

    async def test_init_with_constructor(self, async_engine):
        with pytest.raises(Exception):
            PostgresDocumentStore(
                engine=async_engine, table_name=default_table_name_async
            )

    async def test_async_add_document(self, async_engine, doc_store):
        # Create and add document into the doc store.
        document_text = "add document test"
        doc = Document(text=document_text, id_="add_doc_test", metadata={"doc": "info"})

        await doc_store.async_add_documents([doc])

        # Query the table to confirm the inserted document is present.
        query = f"""select * from "public"."{default_table_name_async}" where id = '{doc.doc_id}';"""
        results = await afetch(async_engine, query)
        result = results[0]
        assert result["node_data"][DATA_KEY]["text"] == document_text

    async def test_add_hash_before_data(self, async_engine, doc_store):
        # Create a document
        document_text = "add document test"
        doc = Document(text=document_text, id_="add_doc_test", metadata={"doc": "info"})

        # Insert the document id with it's doc_hash.
        await doc_store.aset_document_hash(doc_id=doc.doc_id, doc_hash=doc.hash)

        # Insert the document's data
        await doc_store.async_add_documents([doc])

        # Confirm the overwrite was successful.
        query = f"""select * from "public"."{default_table_name_async}" where id = '{doc.doc_id}';"""
        results = await afetch(async_engine, query)
        result = results[0]
        assert result["node_data"][DATA_KEY]["text"] == document_text

    async def test_ref_doc_exists(self, doc_store):
        # Create a ref_doc & a doc and add them to the store.
        ref_doc = Document(
            text="first doc", id_="doc_exists_doc_1", metadata={"doc": "info"}
        )
        doc = Document(
            text="second doc", id_="doc_exists_doc_2", metadata={"doc": "info"}
        )
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()

        await doc_store.async_add_documents([ref_doc, doc])

        # Confirm that ref_doc_id is recorded for the doc.
        result = await doc_store.aref_doc_exists(ref_doc_id=ref_doc.doc_id)
        assert result == True

    async def test_fetch_ref_doc_info(self, doc_store):
        # Create a ref_doc & doc and add them to the store.
        ref_doc = Document(
            text="first doc", id_="ref_parent_doc", metadata={"doc": "info"}
        )
        doc = Document(text="second doc", id_="ref_child_doc", metadata={"doc": "info"})
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()
        await doc_store.async_add_documents([ref_doc, doc])

        # Fetch to see if ref_doc_info is found.
        result = await doc_store.aget_ref_doc_info(ref_doc_id=ref_doc.doc_id)
        assert result is not None

        # Add a new_doc with reference to doc.
        new_doc = Document(
            text="third_doc", id_="ref_new_doc", metadata={"doc": "info"}
        )
        new_doc.relationships[NodeRelationship.SOURCE] = doc.as_related_node_info()
        await doc_store.async_add_documents([new_doc])

        # Fetch to see if ref_doc_info is found for both ref_doc and doc.
        results = await doc_store.aget_all_ref_doc_info()
        assert ref_doc.doc_id in results
        assert doc.doc_id in results

    async def test_adelete_ref_doc(self, doc_store):
        # Create a ref_doc & doc and add them to the store.
        ref_doc = Document(
            text="first doc", id_="ref_parent_doc", metadata={"doc": "info"}
        )
        doc = Document(text="second doc", id_="ref_child_doc", metadata={"doc": "info"})
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()
        await doc_store.async_add_documents([ref_doc, doc])

        # Delete the reference doc
        await doc_store.adelete_ref_doc(ref_doc_id=ref_doc.doc_id)

        # Confirm the reference doc along with it's child nodes are deleted.
        assert (
            await doc_store.aget_document(doc_id=doc.doc_id, raise_error=False) is None
        )

    async def test_set_and_get_document_hash(self, doc_store):
        # Set a doc hash for a document
        doc_id = "document_id"
        doc_hash = "document_hash"
        await doc_store.aset_document_hash(doc_id=doc_id, doc_hash=doc_hash)

        # Assert with get that the hash is same as the one set.
        assert await doc_store.aget_document_hash(doc_id=doc_id) == doc_hash

    async def test_set_and_get_document_hashes(self, doc_store):
        # Create a dictionary of doc_id -> doc_hash mappings and add it to the table.
        document_dict = {
            "document one": "document one hash",
            "document two": "document two hash",
        }
        expected_dict = {v: k for k, v in document_dict.items()}
        await doc_store.aset_document_hashes(doc_hashes=document_dict)

        # Get all the doc hashes and assert it is same as the one set.
        results = await doc_store.aget_all_document_hashes()
        assert "document one hash" in results
        assert "document two hash" in results
        assert results["document one hash"] == expected_dict["document one hash"]
        assert results["document two hash"] == expected_dict["document two hash"]

    async def test_doc_store_basic(self, doc_store):
        # Create a doc and a node and add them to the store.
        doc = Document(text="document_1", id_="doc_id_1", metadata={"doc": "info"})
        node = TextNode(text="node_1", id_="node_id_1", metadata={"node": "info"})

        await doc_store.async_add_documents([doc, node])

        # Assert if document exists
        assert await doc_store.adocument_exists(doc_id=doc.doc_id) == True

        # Assert if retrieved doc is the same as the one inserted.
        retrieved_doc = await doc_store.aget_document(doc_id=doc.doc_id)
        assert retrieved_doc == doc

        # Assert if retrieved node is the same as the one inserted.
        retrieved_node = await doc_store.aget_document(doc_id=node.node_id)
        assert retrieved_node == node

    async def test_delete_document(self, async_engine, doc_store):
        # Create a doc and add it to the store.
        doc = Document(text="document_2", id_="doc_id_2", metadata={"doc": "info"})
        await doc_store.async_add_documents([doc])

        # Delete the document from the store.
        await doc_store.adelete_document(doc_id=doc.doc_id)

        # Assert the document is deleted by querying the table.
        query = f"""select * from "public"."{default_table_name_async}" where id = '{doc.doc_id}';"""
        result = await afetch(async_engine, query)
        assert len(result) == 0

    async def test_doc_store_ref_doc_not_added(self, async_engine, doc_store):
        # Create a ref_doc & doc.
        ref_doc = Document(
            text="first doc", id_="doc_id_parent", metadata={"doc": "info"}
        )
        doc = Document(text="second doc", id_="doc_id_child", metadata={"doc": "info"})
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()

        # Insert only the node into the document store.
        await doc_store.async_add_documents([doc])

        query = f"""select id as node_ids from "public"."{default_table_name_async}" where ref_doc_id = '{ref_doc.doc_id}';"""
        result = await afetch(async_engine, query)

        # Assert document has been added
        assert len(result) != 0

        # Delete the document
        await doc_store.adelete_ref_doc(ref_doc_id=ref_doc.doc_id)

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
        await doc_store.async_add_documents([ref_doc, doc, node])

        query = f"""select id as node_ids from "public"."{default_table_name_async}" where ref_doc_id = '{ref_doc.doc_id}';"""
        result = await afetch(async_engine, query)
        result = {"node_ids": [item["node_ids"] for item in result]}

        # Assert the ref_doc has mappings to both the child doc and child node
        assert result["node_ids"] == [
            doc.doc_id,
            node.node_id,
        ]

        # Delete the child document.
        await doc_store.adelete_document(doc.doc_id)

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
        await doc_store.adelete_document(node.node_id)

        # Assert the ref_doc is also deleted from the store.
        query = f"""select * from "public"."{default_table_name_async}" where id = '{ref_doc.doc_id}';"""
        result = await afetch(async_engine, query)
        assert len(result) == 0


@pytest.mark.asyncio(loop_scope="class")
class TestPostgresDocumentStoreSync:
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
    async def sync_engine(
        self,
        db_project,
        db_region,
        db_instance,
        db_name,
    ):
        sync_engine = PostgresEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )

        yield sync_engine

        await sync_engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def sync_doc_store(self, sync_engine):
        sync_engine.init_doc_store_table(table_name=default_table_name_sync)

        sync_doc_store = PostgresDocumentStore.create_sync(
            engine=sync_engine, table_name=default_table_name_sync
        )

        yield sync_doc_store

        query = f'DROP TABLE IF EXISTS "{default_table_name_sync}"'
        await aexecute(sync_engine, query)

    async def test_init_with_constructor(self, sync_engine):
        with pytest.raises(Exception):
            PostgresDocumentStore(
                engine=sync_engine, table_name=default_table_name_sync
            )

    async def test_docs(self, sync_doc_store):
        # Create and add document into the doc store.
        document_text = "add document test"
        doc = Document(text=document_text, id_="add_doc_test", metadata={"doc": "info"})

        # Add document into the store
        sync_doc_store.add_documents([doc])

        # Assert document is found using the docs property.
        docs = sync_doc_store.docs

        assert doc.doc_id in docs

    async def test_add_document(self, sync_engine, sync_doc_store):
        # Create and add document into the doc store.
        document_text = "add document test"
        doc = Document(text=document_text, id_="add_doc_test", metadata={"doc": "info"})

        sync_doc_store.add_documents([doc])

        # Query the table to confirm the inserted document is present.
        query = f"""select * from "public"."{default_table_name_sync}" where id = '{doc.doc_id}';"""
        results = await afetch(sync_engine, query)
        result = results[0]
        assert result["node_data"][DATA_KEY]["text"] == document_text

    async def test_add_hash_before_data(self, sync_engine, sync_doc_store):
        # Create a document
        document_text = "add document test"
        doc = Document(text=document_text, id_="add_doc_test", metadata={"doc": "info"})

        # Insert the document id with it's doc_hash.
        sync_doc_store.set_document_hash(doc_id=doc.doc_id, doc_hash=doc.hash)

        # Insert the document's data
        sync_doc_store.add_documents([doc])

        # Confirm the overwrite was successful.
        query = f"""select * from "public"."{default_table_name_sync}" where id = '{doc.doc_id}';"""
        results = await afetch(sync_engine, query)
        result = results[0]
        assert result["node_data"][DATA_KEY]["text"] == document_text

    async def test_ref_doc_exists(self, sync_doc_store):
        # Create a ref_doc & a doc and add them to the store.
        ref_doc = Document(
            text="first doc", id_="doc_exists_doc_1", metadata={"doc": "info"}
        )
        doc = Document(
            text="second doc", id_="doc_exists_doc_2", metadata={"doc": "info"}
        )
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()

        sync_doc_store.add_documents([ref_doc, doc])

        # Confirm that ref_doc_id is recorded for the doc.
        result = sync_doc_store.ref_doc_exists(ref_doc_id=ref_doc.doc_id)
        assert result == True

    async def test_fetch_ref_doc_info(self, sync_doc_store):
        # Create a ref_doc & doc and add them to the store.
        ref_doc = Document(
            text="first doc", id_="ref_parent_doc", metadata={"doc": "info"}
        )
        doc = Document(text="second doc", id_="ref_child_doc", metadata={"doc": "info"})
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()
        sync_doc_store.add_documents([ref_doc, doc])

        # Fetch to see if ref_doc_info is found.
        result = sync_doc_store.get_ref_doc_info(ref_doc_id=ref_doc.doc_id)
        assert result is not None

        # Add a new_doc with reference to doc.
        new_doc = Document(
            text="third_doc", id_="ref_new_doc", metadata={"doc": "info"}
        )
        new_doc.relationships[NodeRelationship.SOURCE] = doc.as_related_node_info()
        sync_doc_store.add_documents([new_doc])

        # Fetch to see if ref_doc_info is found for both ref_doc and doc.
        results = sync_doc_store.get_all_ref_doc_info()
        assert ref_doc.doc_id in results
        assert doc.doc_id in results

    async def test_delete_ref_doc(self, sync_doc_store):
        # Create a ref_doc & doc and add them to the store.
        ref_doc = Document(
            text="first doc", id_="ref_parent_doc", metadata={"doc": "info"}
        )
        doc = Document(text="second doc", id_="ref_child_doc", metadata={"doc": "info"})
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()
        sync_doc_store.add_documents([ref_doc, doc])

        # Delete the reference doc
        sync_doc_store.delete_ref_doc(ref_doc_id=ref_doc.doc_id)

        # Confirm the reference doc along with it's child nodes are deleted.
        assert sync_doc_store.get_document(doc_id=doc.doc_id, raise_error=False) is None

    async def test_set_and_get_document_hash(self, sync_doc_store):
        # Set a doc hash for a document
        doc_id = "document_id"
        doc_hash = "document_hash"
        sync_doc_store.set_document_hash(doc_id=doc_id, doc_hash=doc_hash)

        # Assert with get that the hash is same as the one set.
        assert sync_doc_store.get_document_hash(doc_id=doc_id) == doc_hash

    async def test_set_and_get_document_hashes(self, sync_doc_store):
        # Create a dictionary of doc_id -> doc_hash mappings and add it to the table.
        document_dict = {
            "document one": "document one hash",
            "document two": "document two hash",
        }
        expected_dict = {v: k for k, v in document_dict.items()}
        sync_doc_store.set_document_hashes(doc_hashes=document_dict)

        # Get all the doc hashes and assert it is same as the one set.
        results = sync_doc_store.get_all_document_hashes()
        assert "document one hash" in results
        assert "document two hash" in results
        assert results["document one hash"] == expected_dict["document one hash"]
        assert results["document two hash"] == expected_dict["document two hash"]

    async def test_sync_doc_store_basic(self, sync_doc_store):
        # Create a doc and a node and add them to the store.
        doc = Document(text="document_1", id_="doc_id_1", metadata={"doc": "info"})
        node = TextNode(text="node_1", id_="node_id_1", metadata={"node": "info"})

        sync_doc_store.add_documents([doc, node])

        # Assert if document exists
        assert sync_doc_store.document_exists(doc_id=doc.doc_id) == True

        # Assert if retrieved doc is the same as the one inserted.
        retrieved_doc = sync_doc_store.get_document(doc_id=doc.doc_id)
        assert retrieved_doc == doc

        # Assert if retrieved node is the same as the one inserted.
        retrieved_node = sync_doc_store.get_document(doc_id=node.node_id)
        assert retrieved_node == node

    async def test_delete_document(self, sync_engine, sync_doc_store):
        # Create a doc and add it to the store.
        doc = Document(text="document_2", id_="doc_id_2", metadata={"doc": "info"})
        sync_doc_store.add_documents([doc])

        # Delete the document from the store.
        sync_doc_store.delete_document(doc_id=doc.doc_id)

        # Assert the document is deleted by querying the table.
        query = f"""select * from "public"."{default_table_name_sync}" where id = '{doc.doc_id}';"""
        result = await afetch(sync_engine, query)
        assert len(result) == 0

    async def test_sync_doc_store_ref_doc_not_added(self, sync_engine, sync_doc_store):
        # Create a ref_doc & doc.
        ref_doc = Document(
            text="first doc", id_="doc_id_parent", metadata={"doc": "info"}
        )
        doc = Document(text="second doc", id_="doc_id_child", metadata={"doc": "info"})
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()

        # Insert only the node into the document store.
        sync_doc_store.add_documents([doc])

        query = f"""select id as node_ids from "public"."{default_table_name_sync}" where ref_doc_id = '{ref_doc.doc_id}';"""
        result = await afetch(sync_engine, query)

        # Assert document has been added
        assert len(result) != 0

        # Delete the document
        sync_doc_store.delete_ref_doc(ref_doc_id=ref_doc.doc_id)

        # Assert if parent doc is deleted
        query = f"""select * from "public"."{default_table_name_sync}" where id = '{ref_doc.doc_id}';"""
        result = await afetch(sync_engine, query)
        assert len(result) == 0

        # Assert if child (related) doc is deleted
        query = f"""select * from "public"."{default_table_name_sync}" where id = '{doc.doc_id}';"""
        result = await afetch(sync_engine, query)
        assert len(result) == 0

    async def test_sync_doc_store_delete_all_ref_doc_nodes(
        self, sync_engine, sync_doc_store
    ):
        # Create a ref_doc, which is the parent doc for a doc and a node.
        ref_doc = Document(text="document", id_="parent_doc", metadata={"doc1": "info"})
        doc = Document(text="document", id_="child_doc", metadata={"doc": "info"})
        doc.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()
        node = TextNode(
            text="node", id_="child_node", metadata={"doc": "from_node_info"}
        )
        node.relationships[NodeRelationship.SOURCE] = ref_doc.as_related_node_info()

        # Add all the structures into the store.
        sync_doc_store.add_documents([ref_doc, doc, node])

        query = f"""select id as node_ids from "public"."{default_table_name_sync}" where ref_doc_id = '{ref_doc.doc_id}';"""
        result = await afetch(sync_engine, query)
        result = {"node_ids": [item["node_ids"] for item in result]}

        # Assert the ref_doc has mappings to both the child doc and child node
        assert result["node_ids"] == [
            doc.doc_id,
            node.node_id,
        ]

        # Delete the child document.
        sync_doc_store.delete_document(doc.doc_id)

        # Assert the ref_doc still  exists.
        query = f"""select * from "public"."{default_table_name_sync}" where id = '{ref_doc.doc_id}';"""
        result = await afetch(sync_engine, query)
        assert len(result) != 0

        # Assert the ref_doc still has a mapping to the child_node.
        query = f"""select id as node_ids from "public"."{default_table_name_sync}" where ref_doc_id = '{ref_doc.doc_id}';"""
        result = await afetch(sync_engine, query)
        result = {"node_ids": [item["node_ids"] for item in result]}

        assert result["node_ids"] == [node.node_id]

        # Delete the child node
        sync_doc_store.delete_document(node.node_id)

        # Assert the ref_doc is also deleted from the store.
        query = f"""select * from "public"."{default_table_name_sync}" where id = '{ref_doc.doc_id}';"""
        result = await afetch(sync_engine, query)
        assert len(result) == 0
