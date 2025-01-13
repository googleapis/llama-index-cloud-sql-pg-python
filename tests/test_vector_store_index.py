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

import pytest
import pytest_asyncio
from llama_index.core.schema import TextNode
from sqlalchemy import text

from llama_index_cloud_sql_pg import PostgresEngine, PostgresVectorStore
from llama_index_cloud_sql_pg.indexes import (  # type: ignore
    DEFAULT_INDEX_NAME_SUFFIX,
    DistanceStrategy,
    HNSWIndex,
    IVFFlatIndex,
)
from llama_index_cloud_sql_pg.vector_store import PostgresVectorStore

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_TABLE_ASYNC = "test_table" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_INDEX_NAME = DEFAULT_TABLE + DEFAULT_INDEX_NAME_SUFFIX
DEFAULT_INDEX_NAME_ASYNC = DEFAULT_TABLE_ASYNC + DEFAULT_INDEX_NAME_SUFFIX
VECTOR_SIZE = 5


texts = ["foo", "bar", "baz", "foobar", "foobaz"]
ids = [str(uuid.uuid4()) for i in range(len(texts))]
metadatas = [{"page": str(i), "source": "google.com"} for i in range(len(texts))]
embeddings = [[1.0] * VECTOR_SIZE for i in range(len(texts))]
nodes = [
    TextNode(
        id_=ids[i],
        text=texts[i],
        embedding=embeddings[i],
        metadata=metadatas[i],  # type: ignore
    )
    for i in range(len(texts))
]


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


async def aexecute(
    engine: PostgresEngine,
    query: str,
) -> None:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            await conn.execute(text(query))
            await conn.commit()

    await engine._run_as_async(run(engine, query))


@pytest.mark.asyncio(loop_scope="class")
class TestIndexSync:
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
        return get_env_var("DATABASE_ID", "instance for Cloud SQL")

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
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE}")
        await engine.close()
        await engine._connector.close_async()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        engine.init_vector_store_table(
            DEFAULT_TABLE, VECTOR_SIZE, overwrite_existing=True
        )
        vs = PostgresVectorStore.create_sync(
            engine,
            table_name=DEFAULT_TABLE,
        )

        vs.add(nodes)
        vs.drop_vector_index()
        yield vs

    async def test_aapply_vector_index(self, vs):
        index = HNSWIndex()
        vs.apply_vector_index(index)
        assert vs.is_valid_index(DEFAULT_INDEX_NAME)
        vs.drop_vector_index(DEFAULT_INDEX_NAME)

    async def test_areindex(self, vs):
        if not vs.is_valid_index(DEFAULT_INDEX_NAME):
            index = HNSWIndex()
            vs.apply_vector_index(index)
        vs.reindex()
        vs.reindex(DEFAULT_INDEX_NAME)
        assert vs.is_valid_index(DEFAULT_INDEX_NAME)
        vs.drop_vector_index(DEFAULT_INDEX_NAME)

    async def test_dropindex(self, vs):
        vs.drop_vector_index()
        result = vs.is_valid_index(DEFAULT_INDEX_NAME)
        assert not result

    async def test_aapply_vector_index_ivfflat(self, vs):
        index = IVFFlatIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
        vs.apply_vector_index(index, concurrently=True)
        assert vs.is_valid_index(DEFAULT_INDEX_NAME)
        index = IVFFlatIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        vs.apply_vector_index(index)
        assert vs.is_valid_index("secondindex")
        vs.drop_vector_index("secondindex")
        vs.drop_vector_index(DEFAULT_INDEX_NAME)

    async def test_is_valid_index(self, vs):
        is_valid = vs.is_valid_index("invalid_index")
        assert is_valid == False


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncIndex:
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
        return get_env_var("DATABASE_ID", "instance for Cloud SQL")

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
        await aexecute(engine, f"DROP TABLE IF EXISTS {DEFAULT_TABLE_ASYNC}")
        await engine.close()
        await engine._connector.close_async()

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await engine.ainit_vector_store_table(
            DEFAULT_TABLE_ASYNC, VECTOR_SIZE, overwrite_existing=True
        )
        vs = await PostgresVectorStore.create(
            engine,
            table_name=DEFAULT_TABLE_ASYNC,
        )

        await vs.async_add(nodes)
        await vs.adrop_vector_index()
        yield vs

    async def test_aapply_vector_index(self, vs):
        index = HNSWIndex()
        await vs.aapply_vector_index(index)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME_ASYNC)

    async def test_areindex(self, vs):
        if not await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC):
            index = HNSWIndex()
            await vs.aapply_vector_index(index)
        await vs.areindex()
        await vs.areindex(DEFAULT_INDEX_NAME_ASYNC)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME_ASYNC)

    async def test_dropindex(self, vs):
        await vs.adrop_vector_index()
        result = await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        assert not result

    async def test_aapply_vector_index_ivfflat(self, vs):
        index = IVFFlatIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
        await vs.aapply_vector_index(index, concurrently=True)
        assert await vs.ais_valid_index(DEFAULT_INDEX_NAME_ASYNC)
        index = IVFFlatIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        await vs.aapply_vector_index(index)
        assert await vs.ais_valid_index("secondindex")
        await vs.adrop_vector_index("secondindex")
        await vs.adrop_vector_index()

    async def test_is_valid_index(self, vs):
        is_valid = await vs.ais_valid_index("invalid_index")
        assert is_valid == False
