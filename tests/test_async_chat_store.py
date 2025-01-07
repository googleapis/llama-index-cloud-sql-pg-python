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

import os
import uuid
from typing import Sequence

import pytest
import pytest_asyncio
from llama_index.core.llms import ChatMessage
from sqlalchemy import RowMapping, text

from llama_index_cloud_sql_pg import PostgresEngine
from llama_index_cloud_sql_pg.async_chat_store import AsyncPostgresChatStore

default_table_name_async = "chat_store_" + str(uuid.uuid4())


async def aexecute(engine: PostgresEngine, query: str) -> None:
    async with engine._pool.connect() as conn:
        await conn.execute(text(query))
        await conn.commit()


async def afetch(engine: PostgresEngine, query: str) -> Sequence[RowMapping]:
    async with engine._pool.connect() as conn:
        result = await conn.execute(text(query))
        result_map = result.mappings()
        result_fetch = result_map.fetchall()
    return result_fetch


def get_env_var(key: str, desc: str) -> str:
    v = os.environ.get(key)
    if v is None:
        raise ValueError(f"Must set env var {key} to: {desc}")
    return v


@pytest.mark.asyncio(loop_scope="class")
class TestAsyncPostgresChatStores:
    @pytest.fixture(scope="module")
    def db_project(self) -> str:
        return get_env_var("PROJECT_ID", "project id for google cloud")

    @pytest.fixture(scope="module")
    def db_region(self) -> str:
        return get_env_var("REGION", "region for Cloud SQL instance")

    @pytest.fixture(scope="module")
    def db_cluster(self) -> str:
        return get_env_var("CLUSTER_ID", "cluster for Cloud SQL")

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
        db_cluster,
        db_instance,
        db_name,
    ):
        async_engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            cluster=db_cluster,
            region=db_region,
            database=db_name,
        )

        yield async_engine

        await async_engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def chat_store(self, async_engine):
        await async_engine._ainit_chat_store_table(table_name=default_table_name_async)

        chat_store = await AsyncPostgresChatStore.create(
            engine=async_engine, table_name=default_table_name_async
        )

        yield chat_store

        query = f'DROP TABLE IF EXISTS "{default_table_name_async}"'
        await aexecute(async_engine, query)

    async def test_init_with_constructor(self, async_engine):
        with pytest.raises(Exception):
            AsyncPostgresChatStore(
                engine=async_engine, table_name=default_table_name_async
            )

    async def test_async_add_message(self, async_engine, chat_store):
        key = "test_add_key"

        message = ChatMessage(content="add_message_test", role="user")
        await chat_store.async_add_message(key, message=message)

        query = f"""select * from "public"."{default_table_name_async}" where key = '{key}';"""
        results = await afetch(async_engine, query)
        result = results[0]
        assert result["message"] == message.dict()

    async def test_aset_and_aget_messages(self, chat_store):
        message_1 = ChatMessage(content="First message", role="user")
        message_2 = ChatMessage(content="Second message", role="user")
        messages = [message_1, message_2]
        key = "test_set_and_get_key"
        await chat_store.aset_messages(key, messages)

        results = await chat_store.aget_messages(key)

        assert len(results) == 2
        assert results[0].content == message_1.content
        assert results[1].content == message_2.content

    async def test_adelete_messages(self, async_engine, chat_store):
        messages = [ChatMessage(content="Message to delete", role="user")]
        key = "test_delete_key"
        await chat_store.aset_messages(key, messages)

        await chat_store.adelete_messages(key)
        query = f"""select * from "public"."{default_table_name_async}" where key = '{key}' ORDER BY id;"""
        results = await afetch(async_engine, query)

        assert len(results) == 0

    async def test_adelete_message(self, async_engine, chat_store):
        message_1 = ChatMessage(content="Keep me", role="user")
        message_2 = ChatMessage(content="Delete me", role="user")
        messages = [message_1, message_2]
        key = "test_delete_message_key"
        await chat_store.aset_messages(key, messages)

        await chat_store.adelete_message(key, 1)
        query = f"""select * from "public"."{default_table_name_async}" where key = '{key}' ORDER BY id;"""
        results = await afetch(async_engine, query)

        assert len(results) == 1
        assert results[0]["message"] == message_1.dict()

    async def test_adelete_last_message(self, async_engine, chat_store):
        message_1 = ChatMessage(content="Message 1", role="user")
        message_2 = ChatMessage(content="Message 2", role="user")
        message_3 = ChatMessage(content="Message 3", role="user")
        messages = [message_1, message_2, message_3]
        key = "test_delete_last_message_key"
        await chat_store.aset_messages(key, messages)

        await chat_store.adelete_last_message(key)
        query = f"""select * from "public"."{default_table_name_async}" where key = '{key}' ORDER BY id;"""
        results = await afetch(async_engine, query)

        assert len(results) == 2
        assert results[0]["message"] == message_1.dict()
        assert results[1]["message"] == message_2.dict()

    async def test_aget_keys(self, async_engine, chat_store):
        message_1 = [ChatMessage(content="First message", role="user")]
        message_2 = [ChatMessage(content="Second message", role="user")]
        key_1 = "key1"
        key_2 = "key2"
        await chat_store.aset_messages(key_1, message_1)
        await chat_store.aset_messages(key_2, message_2)

        keys = await chat_store.aget_keys()

        assert key_1 in keys
        assert key_2 in keys

    async def test_set_exisiting_key(self, async_engine, chat_store):
        message_1 = [ChatMessage(content="First message", role="user")]
        key = "test_set_exisiting_key"
        await chat_store.aset_messages(key, message_1)

        query = f"""select * from "public"."{default_table_name_async}" where key = '{key}';"""
        results = await afetch(async_engine, query)

        assert len(results) == 1
        result = results[0]
        assert result["message"] == message_1[0].dict()

        message_2 = ChatMessage(content="Second message", role="user")
        message_3 = ChatMessage(content="Third message", role="user")
        messages = [message_2, message_3]

        await chat_store.aset_messages(key, messages)

        query = f"""select * from "public"."{default_table_name_async}" where key = '{key}';"""
        results = await afetch(async_engine, query)

        # Assert the previous messages are deleted and only the newest ones exist.
        assert len(results) == 2

        assert results[0]["message"] == message_2.dict()
        assert results[1]["message"] == message_3.dict()
