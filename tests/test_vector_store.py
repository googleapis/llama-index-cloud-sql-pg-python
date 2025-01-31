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
from llama_index.core.schema import MetadataMode, NodeRelationship, TextNode
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from sqlalchemy import text
from sqlalchemy.engine.row import RowMapping

from llama_index_cloud_sql_pg import PostgresEngine
from llama_index_cloud_sql_pg.vector_store import PostgresVectorStore

DEFAULT_TABLE = "test_table" + str(uuid.uuid4())
VECTOR_SIZE = 5

texts = ["foo", "bar", "baz", "foobar", "foobarbaz"]
embedding = [1.0] * VECTOR_SIZE
nodes = [
    TextNode(
        id_=str(uuid.uuid4()),
        text=texts[i],
        embedding=[1 / (i + 1.0)] * VECTOR_SIZE,
        metadata={  # type: ignore
            "votes": [str(j) for j in range(i + 1)],
            "other_texts": texts[0:i],
        },
    )
    for i in range(len(texts))
]
# setting each node as their own parent
for node in nodes:
    node.relationships[NodeRelationship.SOURCE] = node.as_related_node_info()


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def aexecute(engine: PostgresEngine, query: str) -> None:
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


@pytest.mark.asyncio(loop_scope="class")
class TestVectorStoreAsync:
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
    def db_user(self) -> str:
        return get_env_var("DB_USER", "database name on Cloud SQL instance")

    @pytest.fixture(scope="module")
    def db_pwd(self) -> str:
        return get_env_var("DB_PASSWORD", "database name on Cloud SQL instance")

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_instance, db_name):
        sync_engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield sync_engine

        await aexecute(sync_engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await sync_engine.close()
        await sync_engine._connector.close_async()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await engine.ainit_vector_store_table(
            DEFAULT_TABLE, VECTOR_SIZE, overwrite_existing=True
        )
        vs = await PostgresVectorStore.create(engine, table_name=DEFAULT_TABLE)
        yield vs

    async def test_init_with_constructor(self, engine):
        key = object()
        with pytest.raises(Exception):
            PostgresVectorStore(key, engine, table_name=DEFAULT_TABLE)

    async def test_validate_id_column_create(self, engine, vs):
        test_id_column = "test_id_column"
        with pytest.raises(
            Exception, match=f"Id column, {test_id_column}, does not exist."
        ):
            await PostgresVectorStore.create(
                engine, table_name=DEFAULT_TABLE, id_column=test_id_column
            )

    async def test_validate_text_column_create(self, engine, vs):
        test_text_column = "test_text_column"
        with pytest.raises(
            Exception, match=f"Text column, {test_text_column}, does not exist."
        ):
            await PostgresVectorStore.create(
                engine, table_name=DEFAULT_TABLE, text_column=test_text_column
            )

    async def test_validate_embedding_column_create(self, engine, vs):
        test_embed_column = "test_embed_column"
        with pytest.raises(
            Exception, match=f"Embedding column, {test_embed_column}, does not exist."
        ):
            await PostgresVectorStore.create(
                engine, table_name=DEFAULT_TABLE, embedding_column=test_embed_column
            )

    async def test_validate_node_column_create(self, engine, vs):
        test_node_column = "test_node_column"
        with pytest.raises(
            Exception, match=f"Node column, {test_node_column}, does not exist."
        ):
            await PostgresVectorStore.create(
                engine, table_name=DEFAULT_TABLE, node_column=test_node_column
            )

    async def test_validate_ref_doc_id_column_create(self, engine, vs):
        test_ref_doc_id_column = "test_ref_doc_id_column"
        with pytest.raises(
            Exception,
            match=f"Reference Document Id column, {test_ref_doc_id_column}, does not exist.",
        ):
            await PostgresVectorStore.create(
                engine,
                table_name=DEFAULT_TABLE,
                ref_doc_id_column=test_ref_doc_id_column,
            )

    async def test_validate_metadata_json_column_create(self, engine, vs):
        test_metadata_json_column = "test_metadata_json_column"
        with pytest.raises(
            Exception,
            match=f"Metadata column, {test_metadata_json_column}, does not exist.",
        ):
            await PostgresVectorStore.create(
                engine,
                table_name=DEFAULT_TABLE,
                metadata_json_column=test_metadata_json_column,
            )

    async def test_add(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 5

    async def test_async_add(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        await vs.async_add(nodes)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 5

    async def test_delete(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        vs.delete(nodes[0].node_id)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 4

    async def test_adelete(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        await vs.adelete(nodes[0].node_id)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 4

    async def test_delete_nodes(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        vs.delete_nodes(
            node_ids=[nodes[0].node_id, nodes[1].node_id],
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="text", value="foo", operator=FilterOperator.TEXT_MATCH
                    ),
                    MetadataFilter(key="text", value="bar", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR,
            ),
        )

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

    async def test_adelete_nodes(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        await vs.adelete_nodes(
            node_ids=[nodes[0].node_id, nodes[1].node_id],
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="text", value="foo", operator=FilterOperator.TEXT_MATCH
                    ),
                    MetadataFilter(key="text", value="bar", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR,
            ),
        )

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

    async def test_get_nodes(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        results = vs.get_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="text", value="foo", operator=FilterOperator.TEXT_MATCH
                    ),
                    MetadataFilter(
                        key="text", value="bar", operator=FilterOperator.TEXT_MATCH
                    ),
                ],
                condition=FilterCondition.AND,
            )
        )

        assert len(results) == 2
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foobar"

    async def test_aget_nodes(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        results = await vs.aget_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="text", value="foo", operator=FilterOperator.TEXT_MATCH
                    ),
                    MetadataFilter(
                        key="text", value="bar", operator=FilterOperator.TEXT_MATCH
                    ),
                ],
                condition=FilterCondition.AND,
            )
        )

        assert len(results) == 2
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foobar"

    async def test_aget_nodes_filter_1(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        results = await vs.aget_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="text", value=["foo", "fooz"], operator=FilterOperator.IN
                    ),
                    MetadataFilter(
                        key="text", value=["bar", "baarz"], operator=FilterOperator.NIN
                    ),
                ],
                condition=FilterCondition.AND,
            )
        )

        assert len(results) == 1
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foo"

    async def test_get_nodes_filter_2(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        results = vs.get_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="other_texts",
                        value="",
                        operator=FilterOperator.IS_EMPTY,
                    ),
                ],
            )
        )

        assert len(results) == 1
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foo"

    async def test_aget_nodes_filter_3(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        nodes[0].excluded_embed_metadata_keys = ["abc", "def"]
        vs.add(nodes)
        results = await vs.aget_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="other_texts",
                        operator=FilterOperator.CONTAINS,
                        value="foobar",
                    ),
                ],
            )
        )

        assert len(results) == 1
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foobarbaz"

    async def test_get_nodes_filter_4(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        results = vs.get_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="votes", value=["3", "4"], operator=FilterOperator.ANY
                    ),
                    MetadataFilter(
                        key="votes", value=["3", "4"], operator=FilterOperator.ALL
                    ),
                ],
                condition=FilterCondition.OR,
            )
        )

        assert len(results) == 2
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foobar"
        assert results[1].get_content(metadata_mode=MetadataMode.NONE) == "foobarbaz"

    async def test_aquery(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        query = VectorStoreQuery(
            query_embedding=[1.0] * VECTOR_SIZE, similarity_top_k=3
        )
        results = await vs.aquery(query)

        assert results.nodes is not None
        assert results.ids is not None
        assert results.similarities is not None
        assert len(results.nodes) == 3
        assert results.nodes[0].get_content(metadata_mode=MetadataMode.NONE) == "foo"

    async def test_query(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        query = VectorStoreQuery(
            query_embedding=[1.0] * VECTOR_SIZE, similarity_top_k=3
        )
        results = vs.query(query)

        assert results.nodes is not None
        assert results.ids is not None
        assert results.similarities is not None
        assert len(results.nodes) == 3
        assert results.nodes[0].get_content(metadata_mode=MetadataMode.NONE) == "foo"

    async def test_aclear(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_adelete
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        await vs.aclear()

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 0

    async def test_clear(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_adelete
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        vs.clear()

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 0


@pytest.mark.asyncio(loop_scope="class")
class TestVectorStoreSync:
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
    def db_user(self) -> str:
        return get_env_var("DB_USER", "database name on Cloud SQL instance")

    @pytest.fixture(scope="module")
    def db_pwd(self) -> str:
        return get_env_var("DB_PASSWORD", "database name on Cloud SQL instance")

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_instance, db_name):
        sync_engine = PostgresEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield sync_engine

        await aexecute(sync_engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await sync_engine.close()
        await sync_engine._connector.close_async()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        engine.init_vector_store_table(
            DEFAULT_TABLE, VECTOR_SIZE, overwrite_existing=True
        )
        vs = PostgresVectorStore.create_sync(engine, table_name=DEFAULT_TABLE)
        yield vs

    async def test_init_with_constructor(self, engine):
        key = object()
        with pytest.raises(Exception):
            PostgresVectorStore(key, engine, table_name=DEFAULT_TABLE)

    async def test_validate_id_column_create(self, engine, vs):
        test_id_column = "test_id_column"
        with pytest.raises(
            Exception, match=f"Id column, {test_id_column}, does not exist."
        ):
            await PostgresVectorStore.create(
                engine, table_name=DEFAULT_TABLE, id_column=test_id_column
            )

    async def test_validate_text_column_create(self, engine, vs):
        test_text_column = "test_text_column"
        with pytest.raises(
            Exception, match=f"Text column, {test_text_column}, does not exist."
        ):
            await PostgresVectorStore.create(
                engine, table_name=DEFAULT_TABLE, text_column=test_text_column
            )

    async def test_validate_embedding_column_create(self, engine, vs):
        test_embed_column = "test_embed_column"
        with pytest.raises(
            Exception, match=f"Embedding column, {test_embed_column}, does not exist."
        ):
            await PostgresVectorStore.create(
                engine, table_name=DEFAULT_TABLE, embedding_column=test_embed_column
            )

    async def test_validate_node_column_create(self, engine, vs):
        test_node_column = "test_node_column"
        with pytest.raises(
            Exception, match=f"Node column, {test_node_column}, does not exist."
        ):
            await PostgresVectorStore.create(
                engine, table_name=DEFAULT_TABLE, node_column=test_node_column
            )

    async def test_validate_ref_doc_id_column_create(self, engine, vs):
        test_ref_doc_id_column = "test_ref_doc_id_column"
        with pytest.raises(
            Exception,
            match=f"Reference Document Id column, {test_ref_doc_id_column}, does not exist.",
        ):
            await PostgresVectorStore.create(
                engine,
                table_name=DEFAULT_TABLE,
                ref_doc_id_column=test_ref_doc_id_column,
            )

    async def test_validate_metadata_json_column_create(self, engine, vs):
        test_metadata_json_column = "test_metadata_json_column"
        with pytest.raises(
            Exception,
            match=f"Metadata column, {test_metadata_json_column}, does not exist.",
        ):
            await PostgresVectorStore.create(
                engine,
                table_name=DEFAULT_TABLE,
                metadata_json_column=test_metadata_json_column,
            )

    async def test_add(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 5

    async def test_async_add(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        await vs.async_add(nodes)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 5

    async def test_delete(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        vs.delete(nodes[0].node_id)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 4

    async def test_adelete(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        await vs.adelete(nodes[0].node_id)

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 4

    async def test_delete_nodes(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        vs.delete_nodes(
            node_ids=[nodes[0].node_id, nodes[1].node_id],
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="text", value="foo", operator=FilterOperator.TEXT_MATCH
                    ),
                    MetadataFilter(key="text", value="bar", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR,
            ),
        )

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

    async def test_adelete_nodes(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        await vs.adelete_nodes(
            node_ids=[nodes[0].node_id, nodes[1].node_id],
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="text", value="foo", operator=FilterOperator.TEXT_MATCH
                    ),
                    MetadataFilter(key="text", value="bar", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR,
            ),
        )

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

    async def test_get_nodes(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        results = vs.get_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="text", value="foo", operator=FilterOperator.TEXT_MATCH
                    ),
                    MetadataFilter(
                        key="text", value="bar", operator=FilterOperator.TEXT_MATCH
                    ),
                ],
                condition=FilterCondition.AND,
            )
        )

        assert len(results) == 2
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foobar"

    async def test_aget_nodes(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        results = await vs.aget_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="text", value="foo", operator=FilterOperator.TEXT_MATCH
                    ),
                    MetadataFilter(
                        key="text", value="bar", operator=FilterOperator.TEXT_MATCH
                    ),
                ],
                condition=FilterCondition.AND,
            )
        )

        assert len(results) == 2
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foobar"

    async def test_aget_nodes_filter_1(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        results = await vs.aget_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="text", value=["foo", "fooz"], operator=FilterOperator.IN
                    ),
                    MetadataFilter(
                        key="text", value=["bar", "baarz"], operator=FilterOperator.NIN
                    ),
                ],
                condition=FilterCondition.AND,
            )
        )

        assert len(results) == 1
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foo"

    async def test_get_nodes_filter_2(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        results = vs.get_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="other_texts",
                        value="",
                        operator=FilterOperator.IS_EMPTY,
                    ),
                ],
            )
        )

        assert len(results) == 1
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foo"

    async def test_aget_nodes_filter_3(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        nodes[0].excluded_embed_metadata_keys = ["abc", "def"]
        vs.add(nodes)
        results = await vs.aget_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="other_texts",
                        operator=FilterOperator.CONTAINS,
                        value="foobar",
                    ),
                ],
            )
        )

        assert len(results) == 1
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foobarbaz"

    async def test_get_nodes_filter_4(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        results = vs.get_nodes(
            filters=MetadataFilters(
                filters=[
                    MetadataFilter(
                        key="votes", value=["3", "4"], operator=FilterOperator.ANY
                    ),
                    MetadataFilter(
                        key="votes", value=["3", "4"], operator=FilterOperator.ALL
                    ),
                ],
                condition=FilterCondition.OR,
            )
        )

        assert len(results) == 2
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foobar"
        assert results[1].get_content(metadata_mode=MetadataMode.NONE) == "foobarbaz"

    async def test_aquery(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        query = VectorStoreQuery(
            query_embedding=[1.0] * VECTOR_SIZE, similarity_top_k=3
        )
        results = await vs.aquery(query)

        assert results.nodes is not None
        assert results.ids is not None
        assert results.similarities is not None
        assert len(results.nodes) == 3
        assert results.nodes[0].get_content(metadata_mode=MetadataMode.NONE) == "foo"

    async def test_query(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        query = VectorStoreQuery(
            query_embedding=[1.0] * VECTOR_SIZE, similarity_top_k=3
        )
        results = vs.query(query)

        assert results.nodes is not None
        assert results.ids is not None
        assert results.similarities is not None
        assert len(results.nodes) == 3
        assert results.nodes[0].get_content(metadata_mode=MetadataMode.NONE) == "foo"

    async def test_aclear(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_adelete
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        await vs.aclear()

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 0

    async def test_clear(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_adelete
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        vs.add(nodes)
        vs.clear()

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 0
