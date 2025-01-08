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
import warnings
from typing import Sequence

import pytest
import pytest_asyncio
from llama_index.core.llms import ChatMessage
from sqlalchemy import RowMapping, text

from llama_index_cloud_sql_pg import PostgresChatStore, PostgresEngine

default_table_name_async = "chat_store_" + str(uuid.uuid4())
default_table_name_sync = "chat_store_" + str(uuid.uuid4())


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
class TestPostgresChatStoreAsync:
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
    async def async_chat_store(self, async_engine):
        await async_engine.ainit_chat_store_table(table_name=default_table_name_async)

        async_chat_store = await PostgresChatStore.create(
            engine=async_engine, table_name=default_table_name_async
        )

        yield async_chat_store

        query = f'DROP TABLE IF EXISTS "{default_table_name_async}"'
        await aexecute(async_engine, query)

    async def test_init_with_constructor(self, async_engine):
        with pytest.raises(Exception):
            PostgresChatStore(engine=async_engine, table_name=default_table_name_async)

    async def test_async_add_message(self, async_engine, async_chat_store):
        key = "test_add_key"

        message = ChatMessage(content="add_message_test", role="user")
        await async_chat_store.async_add_message(key, message=message)

        query = f"""select * from "public"."{default_table_name_async}" where key = '{key}';"""
        results = await afetch(async_engine, query)
        result = results[0]
        assert result["message"] == message.dict()

    async def test_aset_and_aget_messages(self, async_chat_store):
        message_1 = ChatMessage(content="First message", role="user")
        message_2 = ChatMessage(content="Second message", role="user")
        messages = [message_1, message_2]
        key = "test_set_and_get_key"
        await async_chat_store.aset_messages(key, messages)

        results = await async_chat_store.aget_messages(key)

        assert len(results) == 2
        assert results[0].content == message_1.content
        assert results[1].content == message_2.content

    async def test_adelete_messages(self, async_engine, async_chat_store):
        messages = [ChatMessage(content="Message to delete", role="user")]
        key = "test_delete_key"
        await async_chat_store.aset_messages(key, messages)

        await async_chat_store.adelete_messages(key)
        query = f"""select * from "public"."{default_table_name_async}" where key = '{key}' ORDER BY id;"""
        results = await afetch(async_engine, query)

        assert len(results) == 0

    async def test_adelete_message(self, async_engine, async_chat_store):
        message_1 = ChatMessage(content="Keep me", role="user")
        message_2 = ChatMessage(content="Delete me", role="user")
        messages = [message_1, message_2]
        key = "test_delete_message_key"
        await async_chat_store.aset_messages(key, messages)

        await async_chat_store.adelete_message(key, 1)
        query = f"""select * from "public"."{default_table_name_async}" where key = '{key}' ORDER BY id;"""
        results = await afetch(async_engine, query)

        assert len(results) == 1
        assert results[0]["message"] == message_1.dict()

    async def test_adelete_last_message(self, async_engine, async_chat_store):
        message_1 = ChatMessage(content="Message 1", role="user")
        message_2 = ChatMessage(content="Message 2", role="user")
        message_3 = ChatMessage(content="Message 3", role="user")
        messages = [message_1, message_2, message_3]
        key = "test_delete_last_message_key"
        await async_chat_store.aset_messages(key, messages)

        await async_chat_store.adelete_last_message(key)
        query = f"""select * from "public"."{default_table_name_async}" where key = '{key}' ORDER BY id;"""
        results = await afetch(async_engine, query)

        assert len(results) == 2
        assert results[0]["message"] == message_1.dict()
        assert results[1]["message"] == message_2.dict()

    async def test_aget_keys(self, async_engine, async_chat_store):
        message_1 = [ChatMessage(content="First message", role="user")]
        message_2 = [ChatMessage(content="Second message", role="user")]
        key_1 = "key1"
        key_2 = "key2"
        await async_chat_store.aset_messages(key_1, message_1)
        await async_chat_store.aset_messages(key_2, message_2)

        keys = await async_chat_store.aget_keys()

        assert key_1 in keys
        assert key_2 in keys

    async def test_set_exisiting_key(self, async_engine, async_chat_store):
        message_1 = [ChatMessage(content="First message", role="user")]
        key = "test_set_exisiting_key"
        await async_chat_store.aset_messages(key, message_1)

        query = f"""select * from "public"."{default_table_name_async}" where key = '{key}';"""
        results = await afetch(async_engine, query)

        assert len(results) == 1
        result = results[0]
        assert result["message"] == message_1[0].dict()

        message_2 = ChatMessage(content="Second message", role="user")
        message_3 = ChatMessage(content="Third message", role="user")
        messages = [message_2, message_3]

        await async_chat_store.aset_messages(key, messages)

        query = f"""select * from "public"."{default_table_name_async}" where key = '{key}';"""
        results = await afetch(async_engine, query)

        # Assert the previous messages are deleted and only the newest ones exist.
        assert len(results) == 2

        assert results[0]["message"] == message_2.dict()
        assert results[1]["message"] == message_3.dict()


@pytest.mark.asyncio(loop_scope="class")
class TestPostgresChatStoreSync:
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
    async def sync_engine(self, db_project, db_region, db_instance, db_name):
        sync_engine = PostgresEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )

        yield sync_engine

        await sync_engine.close()

    @pytest_asyncio.fixture(scope="class")
    async def sync_chat_store(self, sync_engine):
        sync_engine.init_chat_store_table(table_name=default_table_name_sync)

        sync_chat_store = PostgresChatStore.create_sync(
            engine=sync_engine, table_name=default_table_name_sync
        )

        yield sync_chat_store

        query = f'DROP TABLE IF EXISTS "{default_table_name_sync}"'
        await aexecute(sync_engine, query)

    async def test_init_with_constructor(self, sync_engine):
        with pytest.raises(Exception):
            PostgresChatStore(engine=sync_engine, table_name=default_table_name_sync)

    async def test_add_message(self, sync_engine, sync_chat_store):
        key = "test_add_key"

        message = ChatMessage(content="add_message_test", role="user")
        sync_chat_store.add_message(key, message=message)

        query = f"""select * from "public"."{default_table_name_sync}" where key = '{key}';"""
        results = await afetch(sync_engine, query)
        result = results[0]
        assert result["message"] == message.dict()

    async def test_set_and_get_messages(self, sync_chat_store):
        message_1 = ChatMessage(content="First message", role="user")
        message_2 = ChatMessage(content="Second message", role="user")
        messages = [message_1, message_2]
        key = "test_set_and_get_key"
        sync_chat_store.set_messages(key, messages)

        results = sync_chat_store.get_messages(key)

        assert len(results) == 2
        assert results[0].content == message_1.content
        assert results[1].content == message_2.content

    async def test_delete_messages(self, sync_engine, sync_chat_store):
        messages = [ChatMessage(content="Message to delete", role="user")]
        key = "test_delete_key"
        sync_chat_store.set_messages(key, messages)

        sync_chat_store.delete_messages(key)
        query = f"""select * from "public"."{default_table_name_sync}" where key = '{key}' ORDER BY id;"""
        results = await afetch(sync_engine, query)

        assert len(results) == 0

    async def test_delete_message(self, sync_engine, sync_chat_store):
        message_1 = ChatMessage(content="Keep me", role="user")
        message_2 = ChatMessage(content="Delete me", role="user")
        messages = [message_1, message_2]
        key = "test_delete_message_key"
        sync_chat_store.set_messages(key, messages)

        sync_chat_store.delete_message(key, 1)
        query = f"""select * from "public"."{default_table_name_sync}" where key = '{key}' ORDER BY id;"""
        results = await afetch(sync_engine, query)

        assert len(results) == 1
        assert results[0]["message"] == message_1.dict()

    async def test_delete_last_message(self, sync_engine, sync_chat_store):
        message_1 = ChatMessage(content="Message 1", role="user")
        message_2 = ChatMessage(content="Message 2", role="user")
        message_3 = ChatMessage(content="Message 3", role="user")
        messages = [message_1, message_2, message_3]
        key = "test_delete_last_message_key"
        sync_chat_store.set_messages(key, messages)

        sync_chat_store.delete_last_message(key)
        query = f"""select * from "public"."{default_table_name_sync}" where key = '{key}' ORDER BY id;"""
        results = await afetch(sync_engine, query)

        assert len(results) == 2
        assert results[0]["message"] == message_1.dict()
        assert results[1]["message"] == message_2.dict()

    async def test_get_keys(self, sync_engine, sync_chat_store):
        message_1 = [ChatMessage(content="First message", role="user")]
        message_2 = [ChatMessage(content="Second message", role="user")]
        key_1 = "key1"
        key_2 = "key2"
        sync_chat_store.set_messages(key_1, message_1)
        sync_chat_store.set_messages(key_2, message_2)

        keys = sync_chat_store.get_keys()

        assert key_1 in keys
        assert key_2 in keys

    async def test_set_exisiting_key(self, sync_engine, sync_chat_store):
        message_1 = [ChatMessage(content="First message", role="user")]
        key = "test_set_exisiting_key"
        sync_chat_store.set_messages(key, message_1)

        query = f"""select * from "public"."{default_table_name_sync}" where key = '{key}';"""
        results = await afetch(sync_engine, query)

        assert len(results) == 1
        result = results[0]
        assert result["message"] == message_1[0].dict()

        message_2 = ChatMessage(content="Second message", role="user")
        message_3 = ChatMessage(content="Third message", role="user")
        messages = [message_2, message_3]

        sync_chat_store.set_messages(key, messages)

        query = f"""select * from "public"."{default_table_name_sync}" where key = '{key}';"""
        results = await afetch(sync_engine, query)

        # Assert the previous messages are deleted and only the newest ones exist.
        assert len(results) == 2

        assert results[0]["message"] == message_2.dict()
        assert results[1]["message"] == message_3.dict()
