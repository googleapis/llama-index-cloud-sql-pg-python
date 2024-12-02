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
import warnings
from typing import Sequence

import pytest
import pytest_asyncio
from llama_index.core.data_structs.data_structs import IndexDict, IndexGraph, IndexList
from sqlalchemy import RowMapping, text

from llama_index_cloud_sql_pg import PostgresEngine, PostgresIndexStore

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
class TestPostgresIndexStoreAsync:
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
        self, db_project, db_region, db_instance, db_name, user, password
    ):
        async_engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            user=user,
            password=password,
        )

        yield async_engine

        await async_engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def index_store(self, async_engine):
        await async_engine.ainit_index_store_table(table_name=default_table_name_async)

        index_store = await PostgresIndexStore.create(
            engine=async_engine, table_name=default_table_name_async
        )

        yield index_store

        query = f'DROP TABLE IF EXISTS "{default_table_name_async}"'
        await aexecute(async_engine, query)

    async def test_init_with_constructor(self, async_engine):
        with pytest.raises(Exception):
            PostgresIndexStore(engine=async_engine, table_name=default_table_name_async)

    async def test_add_and_delete_index(self, index_store, async_engine):
        index_struct = IndexGraph()
        index_id = index_struct.index_id
        index_type = index_struct.get_type()
        await index_store.aadd_index_struct(index_struct)

        query = f"""select * from "public"."{default_table_name_async}" where index_id = '{index_id}';"""
        results = await afetch(async_engine, query)
        result = results[0]
        assert result.get("type") == index_type

        await index_store.adelete_index_struct(index_id)
        query = f"""select * from "public"."{default_table_name_async}" where index_id = '{index_id}';"""
        results = await afetch(async_engine, query)
        assert results == []

    async def test_get_index(self, index_store):
        index_struct = IndexGraph()
        index_id = index_struct.index_id
        index_type = index_struct.get_type()
        await index_store.aadd_index_struct(index_struct)

        ind_struct = await index_store.aget_index_struct(index_id)

        assert index_struct == ind_struct

    async def test_aindex_structs(self, index_store):
        index_dict_struct = IndexDict()
        index_list_struct = IndexList()
        index_graph_struct = IndexGraph()

        await index_store.aadd_index_struct(index_dict_struct)
        await index_store.aadd_index_struct(index_graph_struct)
        await index_store.aadd_index_struct(index_list_struct)

        indexes = await index_store.aindex_structs()

        index_store.add_index_struct(index_dict_struct)
        index_store.add_index_struct(index_graph_struct)
        index_store.add_index_struct(index_list_struct)

    async def test_warning(self, index_store):
        index_dict_struct = IndexDict()
        index_list_struct = IndexList()

        await index_store.aadd_index_struct(index_dict_struct)
        await index_store.aadd_index_struct(index_list_struct)

        with warnings.catch_warnings(record=True) as w:
            index_struct = await index_store.aget_index_struct()

            assert len(w) == 1
            assert "No struct_id specified and more than one struct exists." in str(
                w[-1].message
            )


@pytest.mark.asyncio(loop_scope="class")
class TestPostgresIndexStoreSync:
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
        self, db_project, db_region, db_instance, db_name, user, password
    ):
        async_engine = PostgresEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            user=user,
            password=password,
        )

        yield async_engine

        await async_engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def index_store(self, async_engine):
        async_engine.init_index_store_table(table_name=default_table_name_sync)

        index_store = PostgresIndexStore.create_sync(
            engine=async_engine, table_name=default_table_name_sync
        )

        yield index_store

        query = f'DROP TABLE IF EXISTS "{default_table_name_sync}"'
        await aexecute(async_engine, query)

    async def test_init_with_constructor(self, async_engine):
        with pytest.raises(Exception):
            PostgresIndexStore(engine=async_engine, table_name=default_table_name_sync)

    async def test_add_and_delete_index(self, index_store, async_engine):
        index_struct = IndexGraph()
        index_id = index_struct.index_id
        index_type = index_struct.get_type()
        index_store.add_index_struct(index_struct)

        query = f"""select * from "public"."{default_table_name_sync}" where index_id = '{index_id}';"""
        results = await afetch(async_engine, query)
        result = results[0]
        assert result.get("type") == index_type

        index_store.delete_index_struct(index_id)
        query = f"""select * from "public"."{default_table_name_sync}" where index_id = '{index_id}';"""
        results = await afetch(async_engine, query)
        assert results == []

    async def test_get_index(self, index_store):
        index_struct = IndexGraph()
        index_id = index_struct.index_id
        index_type = index_struct.get_type()
        index_store.add_index_struct(index_struct)

        ind_struct = index_store.get_index_struct(index_id)

        assert index_struct == ind_struct

    async def test_aindex_structs(self, index_store):
        index_dict_struct = IndexDict()
        index_list_struct = IndexList()
        index_graph_struct = IndexGraph()

        index_store.add_index_struct(index_dict_struct)
        index_store.add_index_struct(index_graph_struct)
        index_store.add_index_struct(index_list_struct)

        indexes = index_store.index_structs()

        index_store.add_index_struct(index_dict_struct)
        index_store.add_index_struct(index_graph_struct)
        index_store.add_index_struct(index_list_struct)

    async def test_warning(self, index_store):
        index_dict_struct = IndexDict()
        index_list_struct = IndexList()

        index_store.add_index_struct(index_dict_struct)
        index_store.add_index_struct(index_list_struct)

        with warnings.catch_warnings(record=True) as w:
            index_struct = index_store.get_index_struct()

            assert len(w) == 1
            assert "No struct_id specified and more than one struct exists." in str(
                w[-1].message
            )
