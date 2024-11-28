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
from typing import Sequence

import asyncpg  # type: ignore
import pytest
import pytest_asyncio
from google.cloud.sql.connector import Connector, IPTypes
from sqlalchemy import VARCHAR, text
from sqlalchemy.engine import URL
from sqlalchemy.engine.row import RowMapping
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from llama_index_cloud_sql_pg import Column, PostgresEngine


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


async def afetch(engine: PostgresEngine, query: str) -> Sequence[RowMapping]:
    async def run(engine, query):
        async with engine._pool.connect() as conn:
            result = await conn.execute(text(query))
            result_map = result.mappings()
            result_fetch = result_map.fetchall()
        return result_fetch

    return await engine._run_as_async(run(engine, query))


@pytest.mark.asyncio
class TestEngineAsync:
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
        return get_env_var("DATABASE_ID", "instance for Cloud SQL")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for Cloud SQL")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for Cloud SQL")

    @pytest.fixture(scope="module")
    def iam_account(self) -> str:
        return get_env_var("IAM_ACCOUNT", "Cloud SQL IAM account email")

    @pytest.fixture(scope="module")
    def host(self) -> str:
        return get_env_var("IP_ADDRESS", "IP Address for the connection string")

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_instance, db_name):
        engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine
        await engine.close()

    async def test_password(
        self,
        db_project,
        db_region,
        db_instance,
        db_name,
        user,
        password,
    ):
        PostgresEngine._connector = None
        engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            user=user,
            password=password,
        )
        assert engine
        await aexecute(engine, "SELECT 1")
        PostgresEngine._connector = None

    async def test_from_engine(
        self,
        db_project,
        db_region,
        db_instance,
        db_name,
        user,
        password,
    ):
        async with Connector() as connector:

            async def getconn() -> asyncpg.Connection:
                conn = await connector.connect_async(  # type: ignore
                    f"{db_project}:{db_region}:{db_instance}",
                    "asyncpg",
                    user=user,
                    password=password,
                    db=db_name,
                    enable_iam_auth=False,
                    ip_type=IPTypes.PUBLIC,
                )
                return conn

            engine = create_async_engine(
                "postgresql+asyncpg://",
                async_creator=getconn,
            )

            engine = PostgresEngine.from_engine(engine)
            await aexecute(engine, "SELECT 1")
            await engine.close()

    async def test_from_connection_string(self, db_name, user, password, host):
        port = "5432"
        url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"
        engine = PostgresEngine.from_engine_args(
            url,
            echo=True,
            poolclass=NullPool,
        )
        await aexecute(engine, "SELECT 1")
        await engine.close()

        engine = PostgresEngine.from_engine_args(
            URL.create("postgresql+asyncpg", user, password, host, port, db_name)
        )
        await aexecute(engine, "SELECT 1")
        await engine.close()

    async def test_from_connection_string_url_error(
        self,
        db_name,
        user,
        password,
        host,
    ):
        port = "5432"
        url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"
        with pytest.raises(TypeError):
            engine = PostgresEngine.from_engine_args(url, random=False)
        with pytest.raises(ValueError):
            PostgresEngine.from_engine_args(
                f"postgresql+pg8000://{user}:{password}@{host}:{port}/{db_name}",
            )
        with pytest.raises(ValueError):
            PostgresEngine.from_engine_args(
                URL.create("postgresql+pg8000", user, password, host, port, db_name)
            )

    async def test_column(self, engine):
        with pytest.raises(ValueError):
            Column("test", VARCHAR)
        with pytest.raises(ValueError):
            Column(1, "INTEGER")

    async def test_iam_account_override(
        self,
        db_project,
        db_cluster,
        db_instance,
        db_region,
        db_name,
        iam_account,
    ):
        engine = await PostgresEngine.afrom_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            iam_account_email=iam_account,
        )
        assert engine
        await aexecute(engine, "SELECT 1")
        await engine.close()


@pytest.mark.asyncio
class TestEngineSync:
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
        return get_env_var("DATABASE_ID", "instance for Cloud SQL")

    @pytest.fixture(scope="module")
    def user(self) -> str:
        return get_env_var("DB_USER", "database user for Cloud SQL")

    @pytest.fixture(scope="module")
    def password(self) -> str:
        return get_env_var("DB_PASSWORD", "database password for Cloud SQL")

    @pytest.fixture(scope="module")
    def iam_account(self) -> str:
        return get_env_var("IAM_ACCOUNT", "Cloud SQL IAM account email")

    @pytest.fixture(scope="module")
    def host(self) -> str:
        return get_env_var("IP_ADDRESS", "IP Address for the connection string")

    @pytest_asyncio.fixture(scope="class")
    async def engine(self, db_project, db_region, db_instance, db_name):
        engine = PostgresEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
        )
        yield engine
        await engine.close()

    async def test_password(
        self,
        db_project,
        db_region,
        db_instance,
        db_name,
        user,
        password,
    ):
        PostgresEngine._connector = None
        engine = PostgresEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            user=user,
            password=password,
            quota_project=db_project,
        )
        assert engine
        await aexecute(engine, "SELECT 1")
        PostgresEngine._connector = None

    async def test_engine_constructor_key(
        self,
        engine,
    ):
        key = object()
        with pytest.raises(Exception):
            PostgresEngine(key, engine)

    async def test_iam_account_override(
        self,
        db_project,
        db_cluster,
        db_instance,
        db_region,
        db_name,
        iam_account,
    ):
        engine = PostgresEngine.from_instance(
            project_id=db_project,
            instance=db_instance,
            region=db_region,
            database=db_name,
            iam_account_email=iam_account,
        )
        assert engine
        await aexecute(engine, "SELECT 1")
        await engine.close()
