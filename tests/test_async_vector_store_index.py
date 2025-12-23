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
from typing import Any, Coroutine

import pytest
import pytest_asyncio
from llama_index.core.schema import TextNode
from sqlalchemy import text

from llama_index_cloud_sql_pg import PostgresEngine
from llama_index_cloud_sql_pg.async_vector_store import AsyncPostgresVectorStore
from llama_index_cloud_sql_pg.indexes import (
    DEFAULT_INDEX_NAME_SUFFIX,
    DistanceStrategy,
    HNSWIndex,
    IVFFlatIndex,
)

DEFAULT_TABLE = "test_table" + str(uuid.uuid4()).replace("-", "_")
DEFAULT_INDEX_NAME = DEFAULT_TABLE + DEFAULT_INDEX_NAME_SUFFIX
VECTOR_SIZE = 5

texts = ["foo", "bar", "baz", "foobar"]
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


@pytest.mark.asyncio(loop_scope="class")
class TestIndex:
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

    @pytest_asyncio.fixture(scope="class")
    async def vs(self, engine):
        await run_on_background(
            engine,
            engine._ainit_vector_store_table(
                DEFAULT_TABLE, VECTOR_SIZE, overwrite_existing=True
            ),
        )
        vs = await AsyncPostgresVectorStore.create(
            engine,
            table_name=DEFAULT_TABLE,
        )

        await vs.async_add(nodes)
        await vs.adrop_vector_index()
        yield vs

    async def test_aapply_vector_index(self, vs):
        index = HNSWIndex()
        await vs.aapply_vector_index(index)
        assert await vs.is_valid_index(DEFAULT_INDEX_NAME)
        await vs.adrop_vector_index(DEFAULT_INDEX_NAME)

    async def test_areindex(self, vs):
        if not await vs.is_valid_index(DEFAULT_INDEX_NAME):
            index = HNSWIndex()
            await vs.aapply_vector_index(index)
        await vs.areindex()
        await vs.areindex(DEFAULT_INDEX_NAME)
        assert await vs.is_valid_index(DEFAULT_INDEX_NAME)
        await vs.adrop_vector_index()

    async def test_dropindex(self, vs):
        await vs.adrop_vector_index()
        result = await vs.is_valid_index(DEFAULT_INDEX_NAME)
        assert not result

    async def test_aapply_vector_index_ivfflat(self, vs):
        index = IVFFlatIndex(distance_strategy=DistanceStrategy.EUCLIDEAN)
        await vs.aapply_vector_index(index, concurrently=True)
        assert await vs.is_valid_index(DEFAULT_INDEX_NAME)
        index = IVFFlatIndex(
            name="secondindex",
            distance_strategy=DistanceStrategy.INNER_PRODUCT,
        )
        await vs.aapply_vector_index(index)
        assert await vs.is_valid_index("secondindex")
        await vs.adrop_vector_index()
        await vs.adrop_vector_index("secondindex")

    async def test_is_valid_index(self, vs):
        is_valid = await vs.is_valid_index("invalid_index")
        assert is_valid == False
