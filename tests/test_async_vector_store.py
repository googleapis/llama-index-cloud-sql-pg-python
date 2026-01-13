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

from llama_index_cloud_sql_pg import Column, PostgresEngine
from llama_index_cloud_sql_pg.async_vector_store import AsyncPostgresVectorStore

DEFAULT_TABLE = "test_table" + str(uuid.uuid4())
DEFAULT_TABLE_CUSTOM_VS = "test_table" + str(uuid.uuid4())
VECTOR_SIZE = 768

texts = ["foo", "bar", "baz", "foobar"]
embedding = [1.0] * VECTOR_SIZE
nodes = [
    TextNode(
        id_=str(uuid.uuid4()),
        text=texts[i],
        embedding=[1 / (i + 1.0)] * VECTOR_SIZE,
    )
    for i in range(len(texts))
]
# setting each node as their own parent
for node in nodes:
    node.relationships[NodeRelationship.SOURCE] = node.as_related_node_info()
sync_method_exception_str = "Sync methods are not implemented for AsyncPostgresVectorStore. Use PostgresVectorStore interface instead."


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


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


@pytest.mark.asyncio(loop_scope="class")
class TestVectorStore:
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
        engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )

        yield engine
        await aexecute(engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE}"')
        await aexecute(engine, f'DROP TABLE IF EXISTS "{DEFAULT_TABLE_CUSTOM_VS}"')
        await engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await run_on_background(
            engine,
            engine._ainit_vector_store_table(
                DEFAULT_TABLE, VECTOR_SIZE, overwrite_existing=True
            ),
        )
        vs = await run_on_background(
            engine, AsyncPostgresVectorStore.create(engine, table_name=DEFAULT_TABLE)
        )
        yield vs

    @pytest_asyncio.fixture(scope="class")
    async def custom_vs(self, engine):
        await run_on_background(
            engine,
            engine._ainit_vector_store_table(
                DEFAULT_TABLE_CUSTOM_VS,
                VECTOR_SIZE,
                overwrite_existing=True,
                metadata_columns=[
                    Column(name="len", data_type="INTEGER", nullable=False),
                    Column(
                        name="nullable_int_field",
                        data_type="INTEGER",
                        nullable=True,
                    ),
                    Column(
                        name="nullable_str_field",
                        data_type="VARCHAR",
                        nullable=True,
                    ),
                ],
            ),
        )
        vs = await run_on_background(
            engine,
            AsyncPostgresVectorStore.create(
                engine,
                table_name=DEFAULT_TABLE_CUSTOM_VS,
                metadata_columns=[
                    "len",
                    "nullable_int_field",
                    "nullable_str_field",
                ],
            ),
        )
        yield vs

    async def test_init_with_constructor(self, engine):
        key = object()
        with pytest.raises(Exception):
            AsyncPostgresVectorStore(key, engine, table_name=DEFAULT_TABLE)

    async def test_validate_id_column_create(self, engine, vs):
        test_id_column = "test_id_column"
        with pytest.raises(
            Exception, match=f"Id column, {test_id_column}, does not exist."
        ):
            await run_on_background(
                engine,
                AsyncPostgresVectorStore.create(
                    engine, table_name=DEFAULT_TABLE, id_column=test_id_column
                ),
            )

    async def test_validate_text_column_create(self, engine, vs):
        test_text_column = "test_text_column"
        with pytest.raises(
            Exception, match=f"Text column, {test_text_column}, does not exist."
        ):
            await run_on_background(
                engine,
                AsyncPostgresVectorStore.create(
                    engine, table_name=DEFAULT_TABLE, text_column=test_text_column
                ),
            )

    async def test_validate_embedding_column_create(self, engine, vs):
        test_embed_column = "test_embed_column"
        with pytest.raises(
            Exception,
            match=f"Embedding column, {test_embed_column}, does not exist.",
        ):
            await run_on_background(
                engine,
                AsyncPostgresVectorStore.create(
                    engine,
                    table_name=DEFAULT_TABLE,
                    embedding_column=test_embed_column,
                ),
            )

    async def test_validate_node_column_create(self, engine, vs):
        test_node_column = "test_node_column"
        with pytest.raises(
            Exception, match=f"Node column, {test_node_column}, does not exist."
        ):
            await run_on_background(
                engine,
                AsyncPostgresVectorStore.create(
                    engine, table_name=DEFAULT_TABLE, node_column=test_node_column
                ),
            )

    async def test_validate_ref_doc_id_column_create(self, engine, vs):
        test_ref_doc_id_column = "test_ref_doc_id_column"
        with pytest.raises(
            Exception,
            match=f"Reference Document Id column, {test_ref_doc_id_column}, does not exist.",
        ):
            await run_on_background(
                engine,
                AsyncPostgresVectorStore.create(
                    engine,
                    table_name=DEFAULT_TABLE,
                    ref_doc_id_column=test_ref_doc_id_column,
                ),
            )

    async def test_validate_metadata_json_column_create(self, engine, vs):
        test_metadata_json_column = "test_metadata_json_column"
        with pytest.raises(
            Exception,
            match=f"Metadata column, {test_metadata_json_column}, does not exist.",
        ):
            await run_on_background(
                engine,
                AsyncPostgresVectorStore.create(
                    engine,
                    table_name=DEFAULT_TABLE,
                    metadata_json_column=test_metadata_json_column,
                ),
            )

    async def test_async_add(self, engine, vs):
        await run_on_background(engine, vs.async_add(nodes))

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 4

    async def test_async_add_custom_vs(self, engine, custom_vs):
        # setting extra metadata to be indexed in separate column
        for node in nodes:
            node.metadata["len"] = len(node.text)

        await run_on_background(engine, custom_vs.async_add(nodes))

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE_CUSTOM_VS}"')
        assert len(results) == 4
        assert results[0]["len"] == 3
        assert results[0]["nullable_int_field"] == None
        assert results[0]["nullable_str_field"] == None

    async def test_adelete(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        await run_on_background(engine, vs.async_add(nodes))
        await run_on_background(engine, vs.adelete(nodes[0].node_id))

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 3

    async def test_adelete_nodes(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        await run_on_background(engine, vs.async_add(nodes))
        await run_on_background(
            engine,
            vs.adelete_nodes(
                node_ids=[nodes[0].node_id, nodes[1].node_id],
                filters=MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="text",
                            value="foo",
                            operator=FilterOperator.TEXT_MATCH,
                        ),
                        MetadataFilter(
                            key="text", value="bar", operator=FilterOperator.EQ
                        ),
                    ],
                    condition=FilterCondition.OR,
                ),
            ),
        )

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 2

    async def test_aget_nodes(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        await run_on_background(engine, vs.async_add(nodes))
        results = await run_on_background(
            engine,
            vs.aget_nodes(
                filters=MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="text",
                            value="foo",
                            operator=FilterOperator.TEXT_MATCH,
                        ),
                        MetadataFilter(
                            key="text",
                            value="bar",
                            operator=FilterOperator.TEXT_MATCH,
                        ),
                    ],
                    condition=FilterCondition.AND,
                )
            ),
        )

        assert len(results) == 1
        assert results[0].get_content(metadata_mode=MetadataMode.NONE) == "foobar"

    async def test_aquery(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        await run_on_background(engine, vs.async_add(nodes))
        query = VectorStoreQuery(
            query_embedding=[1.0] * VECTOR_SIZE, similarity_top_k=3
        )
        results = await run_on_background(engine, vs.aquery(query))

        assert results.nodes is not None
        assert results.ids is not None
        assert results.similarities is not None
        assert len(results.nodes) == 3
        assert results.nodes[0].get_content(metadata_mode=MetadataMode.NONE) == "foo"

    async def test_aquery_filters(self, engine, custom_vs):
        # Note: To be migrated to a pytest dependency on test_async_add
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE_CUSTOM_VS}"')
        # setting extra metadata to be indexed in separate column
        for node in nodes:
            node.metadata["len"] = len(node.text)

        await run_on_background(engine, custom_vs.async_add(nodes))

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="some_test_column",
                    value=["value_should_be_ignored"],
                    operator=FilterOperator.CONTAINS,
                ),
                MetadataFilter(
                    key="len",
                    value=3,
                    operator=FilterOperator.LTE,
                ),
                MetadataFilter(
                    key="len",
                    value=3,
                    operator=FilterOperator.GTE,
                ),
                MetadataFilter(
                    key="len",
                    value=2,
                    operator=FilterOperator.GT,
                ),
                MetadataFilter(
                    key="len",
                    value=4,
                    operator=FilterOperator.LT,
                ),
                MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="len",
                            value=6.0,
                            operator=FilterOperator.NE,
                        ),
                    ],
                    condition=FilterCondition.OR,
                ),
            ],
            condition=FilterCondition.AND,
        )
        query = VectorStoreQuery(
            query_embedding=[1.0] * VECTOR_SIZE, filters=filters, similarity_top_k=-1
        )
        with warnings.catch_warnings(record=True) as w:
            results = await run_on_background(engine, custom_vs.aquery(query))

            assert len(w) == 1
            assert "Expecting a scalar in the filter value" in str(w[-1].message)

        assert results.nodes is not None
        assert results.ids is not None
        assert results.similarities is not None
        assert len(results.nodes) == 3

    async def test_aclear(self, engine, vs):
        # Note: To be migrated to a pytest dependency on test_adelete
        # Blocked due to unexpected fixtures reloads while running integration test suite
        await aexecute(engine, f'TRUNCATE TABLE "{DEFAULT_TABLE}"')
        await run_on_background(engine, vs.async_add(nodes))
        await run_on_background(engine, vs.aclear())

        results = await afetch(engine, f'SELECT * FROM "{DEFAULT_TABLE}"')
        assert len(results) == 0

    async def test_add(self, vs):
        with pytest.raises(Exception, match=sync_method_exception_str):
            vs.add(nodes)

    async def test_get_nodes(self, vs):
        with pytest.raises(Exception, match=sync_method_exception_str):
            vs.get_nodes()

    async def test_query(self, vs):
        with pytest.raises(Exception, match=sync_method_exception_str):
            vs.query(VectorStoreQuery(query_str="foo"))

    async def test_delete(self, vs):
        with pytest.raises(Exception, match=sync_method_exception_str):
            vs.delete("test_ref_doc_id")

    async def test_delete_nodes(self, vs):
        with pytest.raises(Exception, match=sync_method_exception_str):
            vs.delete_nodes(["test_node_id"])

    async def test_clear(self, vs):
        with pytest.raises(Exception, match=sync_method_exception_str):
            vs.clear()
