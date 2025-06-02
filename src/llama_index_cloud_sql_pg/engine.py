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
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Thread
from typing import TYPE_CHECKING, Any, Awaitable, Optional, TypeVar, Union

import aiohttp
import google.auth  # type: ignore
import google.auth.transport.requests  # type: ignore
from google.cloud.sql.connector import Connector, IPTypes, RefreshStrategy
from sqlalchemy import MetaData, Table, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .version import __version__

if TYPE_CHECKING:
    import asyncpg  # type: ignore
    import google.auth.credentials  # type: ignore

T = TypeVar("T")

USER_AGENT = "llama-index-cloud-sql-pg-python/" + __version__


async def _get_iam_principal_email(
    credentials: google.auth.credentials.Credentials,
) -> str:
    """Get email address associated with current authenticated IAM principal.

    Email will be used for automatic IAM database authentication to Cloud SQL.

    Args:
        credentials (google.auth.credentials.Credentials):
            The credentials object to use in finding the associated IAM
            principal email address.

    Returns:
        email (str):
            The email address associated with the current authenticated IAM
            principal.
    """
    if not credentials.valid:
        request = google.auth.transport.requests.Request()
        credentials.refresh(request)
    if hasattr(credentials, "_service_account_email"):
        return credentials._service_account_email.replace(".gserviceaccount.com", "")
    # call OAuth2 api to get IAM principal email associated with OAuth2 token
    url = f"https://oauth2.googleapis.com/tokeninfo?access_token={credentials.token}"
    async with aiohttp.ClientSession() as client:
        response = await client.get(url, raise_for_status=True)
        response_json: dict = await response.json()
        email = response_json.get("email")
    if email is None:
        raise ValueError(
            "Failed to automatically obtain authenticated IAM principal's "
            "email address using environment's ADC credentials!"
        )
    return email.replace(".gserviceaccount.com", "")


@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool = True

    def __post_init__(self):
        """Check if initialization parameters are valid.

        Raises:
            ValueError: Raises error if Column name is not string.
            ValueError: Raises error if data_type is not type string.
        """
        if not isinstance(self.name, str):
            raise ValueError("Column name must be type string")
        if not isinstance(self.data_type, str):
            raise ValueError("Column data_type must be type string")


class PostgresEngine:
    """A class for managing connections to a Cloud SQL for Postgres database."""

    _connector: Optional[Connector] = None
    _default_loop: Optional[asyncio.AbstractEventLoop] = None
    _default_thread: Optional[Thread] = None
    __create_key = object()

    def __init__(
        self,
        key: object,
        pool: AsyncEngine,
        loop: Optional[asyncio.AbstractEventLoop],
        thread: Optional[Thread],
    ):
        """PostgresEngine constructor.

        Args:
            key (object): Prevent direct constructor usage.
            pool (AsyncEngine): Async engine connection pool.
            loop (Optional[asyncio.AbstractEventLoop]): Async event loop used to create the engine.
            thread (Optional[Thread]): Thread used to create the engine async.

        Raises:
            Exception: If the constructor is called directly by the user.
        """
        if key != PostgresEngine.__create_key:
            raise Exception(
                "Only create class through 'create' or 'create_sync' methods!"
            )
        self._pool = pool
        self._loop = loop
        self._thread = thread

    @classmethod
    async def _create(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
        ip_type: Union[str, IPTypes],
        user: Optional[str] = None,
        password: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        thread: Optional[Thread] = None,
        quota_project: Optional[str] = None,
        iam_account_email: Optional[str] = None,
    ) -> PostgresEngine:
        """Create a PostgresEngine instance.

        Args:
            project_id (str): GCP project ID.
            region (str): Postgres instance region.
            instance (str): Postgres instance name.
            database (str): Database name.
            ip_type (Union[str, IPTypes]): IP address type. Defaults to IPTypes.PUBLIC.
            user (Optional[str]): Postgres user name. Defaults to None.
            password (Optional[str]): Postgres user password. Defaults to None.
            loop (Optional[asyncio.AbstractEventLoop]): Async event loop used to create the engine.
            thread (Optional[Thread]): Thread used to create the engine async.
            quota_project (Optional[str]): Project that provides quota for API calls.
            iam_account_email (Optional[str]): IAM service account email. Defaults to None.

        Raises:
            ValueError: If only one of `user` and `password` is specified.

        Returns:
            PostgresEngine
        """
        if bool(user) ^ bool(password):
            raise ValueError(
                "Only one of 'user' or 'password' were specified. Either "
                "both should be specified to use basic user/password "
                "authentication or neither for IAM DB authentication."
            )
        if cls._connector is None:
            cls._connector = Connector(
                loop=loop,
                user_agent=USER_AGENT,
                quota_project=quota_project,
                refresh_strategy=RefreshStrategy.LAZY,
            )

        # if user and password are given, use basic auth
        if user and password:
            enable_iam_auth = False
            db_user = user
        # otherwise use automatic IAM database authentication
        else:
            enable_iam_auth = True
            if iam_account_email:
                db_user = iam_account_email
            else:
                # get application default credentials
                credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/userinfo.email"]
                )
                db_user = await _get_iam_principal_email(credentials)

        # anonymous function to be used for SQLAlchemy 'creator' argument
        async def getconn() -> asyncpg.Connection:
            conn = await cls._connector.connect_async(  # type: ignore
                f"{project_id}:{region}:{instance}",
                "asyncpg",
                user=db_user,
                password=password,
                db=database,
                enable_iam_auth=enable_iam_auth,
                ip_type=ip_type,
            )
            return conn

        engine = create_async_engine(
            "postgresql+asyncpg://",
            async_creator=getconn,
        )
        return cls(cls.__create_key, engine, loop, thread)

    @classmethod
    def __start_background_loop(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        ip_type: Union[str, IPTypes] = IPTypes.PUBLIC,
        quota_project: Optional[str] = None,
        iam_account_email: Optional[str] = None,
    ) -> Future:
        # Running a loop in a background thread allows us to support
        # async methods from non-async environments
        if cls._default_loop is None:
            cls._default_loop = asyncio.new_event_loop()
            cls._default_thread = Thread(
                target=cls._default_loop.run_forever, daemon=True
            )
            cls._default_thread.start()
        coro = cls._create(
            project_id,
            region,
            instance,
            database,
            ip_type,
            user,
            password,
            loop=cls._default_loop,
            thread=cls._default_thread,
            quota_project=quota_project,
            iam_account_email=iam_account_email,
        )
        return asyncio.run_coroutine_threadsafe(coro, cls._default_loop)

    @classmethod
    def from_instance(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        ip_type: Union[str, IPTypes] = IPTypes.PUBLIC,
        quota_project: Optional[str] = None,
        iam_account_email: Optional[str] = None,
    ) -> PostgresEngine:
        """Create a PostgresEngine from a Postgres instance.

        Args:
            project_id (str): GCP project ID.
            region (str): Postgres instance region.
            instance (str): Postgres instance name.
            database (str): Database name.
            user (Optional[str], optional): Postgres user name. Defaults to None.
            password (Optional[str], optional): Postgres user password. Defaults to None.
            ip_type (Union[str, IPTypes], optional): IP address type. Defaults to IPTypes.PUBLIC.
            quota_project (Optional[str]): Project that provides quota for API calls.
            iam_account_email (Optional[str], optional): IAM service account email. Defaults to None.

        Returns:
            PostgresEngine: A newly created PostgresEngine instance.
        """
        future = cls.__start_background_loop(
            project_id,
            region,
            instance,
            database,
            user,
            password,
            ip_type,
            quota_project=quota_project,
            iam_account_email=iam_account_email,
        )
        return future.result()

    @classmethod
    async def afrom_instance(
        cls,
        project_id: str,
        region: str,
        instance: str,
        database: str,
        user: Optional[str] = None,
        password: Optional[str] = None,
        ip_type: Union[str, IPTypes] = IPTypes.PUBLIC,
        quota_project: Optional[str] = None,
        iam_account_email: Optional[str] = None,
    ) -> PostgresEngine:
        """Create a PostgresEngine from a Postgres instance.

        Args:
            project_id (str): GCP project ID.
            region (str): Postgres instance region.
            instance (str): Postgres instance name.
            database (str): Database name.
            user (Optional[str], optional): Postgres user name. Defaults to None.
            password (Optional[str], optional): Postgres user password. Defaults to None.
            ip_type (Union[str, IPTypes], optional): IP address type. Defaults to IPTypes.PUBLIC.
            quota_project (Optional[str]): Project that provides quota for API calls.
            iam_account_email (Optional[str], optional): IAM service account email. Defaults to None.

        Returns:
            PostgresEngine: A newly created PostgresEngine instance.
        """
        future = cls.__start_background_loop(
            project_id,
            region,
            instance,
            database,
            user,
            password,
            ip_type,
            quota_project=quota_project,
            iam_account_email=iam_account_email,
        )
        return await asyncio.wrap_future(future)

    @classmethod
    def from_engine(
        cls,
        engine: AsyncEngine,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> PostgresEngine:
        """Create an PostgresEngine instance from an AsyncEngine."""
        return cls(cls.__create_key, engine, loop, None)

    @classmethod
    def from_engine_args(
        cls,
        url: Union[str | URL],
        **kwargs: Any,
    ) -> PostgresEngine:
        """Create an PostgresEngine instance from arguments. These parameters are pass directly into sqlalchemy's create_async_engine function.

        Args:
            url (Union[str | URL]): the URL used to connect to a database
            **kwargs (Any, optional): sqlalchemy `create_async_engine` arguments

        Raises:
            ValueError: If `postgresql+asyncpg` is not specified as the PG driver

        Returns:
            PostgresEngine
        """
        # Running a loop in a background thread allows us to support
        # async methods from non-async environments
        if cls._default_loop is None:
            cls._default_loop = asyncio.new_event_loop()
            cls._default_thread = Thread(
                target=cls._default_loop.run_forever, daemon=True
            )
            cls._default_thread.start()

        driver = "postgresql+asyncpg"
        if (isinstance(url, str) and not url.startswith(driver)) or (
            isinstance(url, URL) and url.drivername != driver
        ):
            raise ValueError("Driver must be type 'postgresql+asyncpg'")

        engine = create_async_engine(url, **kwargs)
        return cls(cls.__create_key, engine, cls._default_loop, cls._default_thread)

    async def _run_as_async(self, coro: Awaitable[T]) -> T:
        """Run an async coroutine asynchronously"""
        # If a loop has not been provided, attempt to run in current thread
        if not self._loop:
            return await coro
        # Otherwise, run in the background thread
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, self._loop)  # type: ignore
        )

    def _run_as_sync(self, coro: Awaitable[T]) -> T:
        """Run an async coroutine synchronously"""
        if not self._loop:
            raise Exception(
                "Engine was initialized without a background loop and cannot call sync methods."
            )
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()  # type: ignore

    async def close(self) -> None:
        """Dispose of connection pool"""
        await self._run_as_async(self._pool.dispose())

    async def _ainit_doc_store_table(
        self,
        table_name: str,
        schema_name: str = "public",
        overwrite_existing: bool = False,
    ) -> None:
        """
        Create an  table for the DocumentStore.

        Args:
            table_name (str): The table name to store documents.
            schema_name (str): The schema name to store the documents table.
                Default: "public".
            overwrite_existing (bool): Whether to drop existing table.
                Default: False.

        Returns:
            None
        """
        if overwrite_existing:
            async with self._pool.connect() as conn:
                await conn.execute(
                    text(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"')
                )
                await conn.commit()

        node_data_default = r"{}"

        create_table_query = f"""CREATE TABLE "{schema_name}"."{table_name}"(
            id VARCHAR PRIMARY KEY,
            doc_hash VARCHAR NOT NULL,
            ref_doc_id VARCHAR,
            node_data JSONB NOT NULL DEFAULT '{node_data_default}'::jsonb
        );"""
        create_index_query = f"""CREATE INDEX "{table_name}_idx_ref_doc_id" ON "{schema_name}"."{table_name}" (ref_doc_id);"""
        async with self._pool.connect() as conn:
            await conn.execute(text(create_table_query))
            await conn.execute(text(create_index_query))
            await conn.commit()

    async def ainit_doc_store_table(
        self,
        table_name: str,
        schema_name: str = "public",
        overwrite_existing: bool = False,
    ) -> None:
        """Create a table for the DocumentStore.

        Args:
            table_name (str): The table name to store documents.
            schema_name (str): The schema name to store the documents table.
                Default: "public".
            overwrite_existing (bool): Whether to drop existing table.
                Default: False.

        Returns:
            None
        """
        await self._run_as_async(
            self._ainit_doc_store_table(
                table_name,
                schema_name,
                overwrite_existing,
            )
        )

    def init_doc_store_table(
        self,
        table_name: str,
        schema_name: str = "public",
        overwrite_existing: bool = False,
    ) -> None:
        """Create a table for the DocumentStore.

        Args:
            table_name (str): The table name to store documents.
            schema_name (str): The schema name to store the documents table.
                Default: "public".
            overwrite_existing (bool): Whether to drop existing table.
                Default: False.

        Returns:
            None
        """
        self._run_as_sync(
            self._ainit_doc_store_table(
                table_name,
                schema_name,
                overwrite_existing,
            )
        )

    async def _ainit_vector_store_table(
        self,
        table_name: str,
        vector_size: int,
        schema_name: str = "public",
        id_column: Union[str, Column] = "node_id",
        text_column: str = "text",
        embedding_column: str = "embedding",
        metadata_json_column: str = "li_metadata",
        metadata_columns: list[Column] = [],
        ref_doc_id_column: str = "ref_doc_id",
        node_column: str = "node_data",
        stores_text: bool = True,
        overwrite_existing: bool = False,
    ) -> None:
        """Create a table for the VectorStore.

        Args:
            table_name (str): The table name to store nodes with embedding vectors.
            vector_size (int): Vector size for the embedding model to be used.
            schema_name (str): The schema name to store the vector store table. Default: "public".
            id_column (str): Column that represents if of a Node. Defaults to "node_id".
            text_column (str): Column that represent text content of a Node. Defaults to "text".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the content of Node. Defaults to "embedding".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "li_metadata".
            metadata_columns (list[str]): Column(s) that represent extracted metadata keys in their own columns.
            ref_doc_id_column (str): Column that represents id of a node's parent document. Defaults to "ref_doc_id".
            node_column (str): Column that represents the whole JSON node. Defaults to "node_data".
            stores_text (bool): Whether the table stores text. Defaults to "True".
            overwrite_existing (bool): Whether to drop existing table. Default: False.

        Returns:
            None

        Raises:
            :class:`DuplicateTableError <asyncpg.exceptions.DuplicateTableError>`: if table already exists.
            :class:`UndefinedObjectError <asyncpg.exceptions.UndefinedObjectError>`: if the data type of the id column is not a postgreSQL data type.
        """
        async with self._pool.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()

        if overwrite_existing:
            async with self._pool.connect() as conn:
                await conn.execute(
                    text(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"')
                )
                await conn.commit()

        id_data_type = "VARCHAR" if isinstance(id_column, str) else id_column.data_type
        id_column_name = id_column if isinstance(id_column, str) else id_column.name

        create_table_query = f"""CREATE TABLE "{schema_name}"."{table_name}"(
            "{id_column_name}" {id_data_type} PRIMARY KEY,
            "{metadata_json_column}" JSONB NOT NULL,
            "{embedding_column}" vector({vector_size}),
            "{node_column}" JSON NOT NULL,
            "{ref_doc_id_column}" VARCHAR
            """

        for column in metadata_columns:
            nullable = "NOT NULL" if not column.nullable else ""
            create_table_query += f',\n"{column.name}" {column.data_type} {nullable}'
        nullable_text = "NOT NULL" if stores_text else ""
        create_table_query += f""",\n"{text_column}" TEXT {nullable_text}"""
        create_table_query += "\n);"
        create_index_query = f"""CREATE INDEX "{table_name}_idx_ref_doc_id" ON "{schema_name}"."{table_name}" ("{ref_doc_id_column}");"""

        async with self._pool.connect() as conn:
            await conn.execute(text(create_table_query))
            await conn.execute(text(create_index_query))
            await conn.commit()

    async def ainit_vector_store_table(
        self,
        table_name: str,
        vector_size: int,
        schema_name: str = "public",
        id_column: Union[str, Column] = "node_id",
        text_column: str = "text",
        embedding_column: str = "embedding",
        metadata_json_column: str = "li_metadata",
        metadata_columns: list[Column] = [],
        ref_doc_id_column: str = "ref_doc_id",
        node_column: str = "node_data",
        stores_text: bool = True,
        overwrite_existing: bool = False,
    ) -> None:
        """Create a table for the VectorStore.

        Args:
            table_name (str): The table name to store nodes with embedding vectors.
            vector_size (int): Vector size for the embedding model to be used.
            schema_name (str): The schema name to store the vector store table. Default: "public".
            id_column (str): Column that represents if of a Node. Defaults to "node_id".
            text_column (str): Column that represent text content of a Node. Defaults to "text".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the content of Node. Defaults to "embedding".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "li_metadata".
            metadata_columns (list[str]): Column(s) that represent extracted metadata keys in their own columns.
            ref_doc_id_column (str): Column that represents id of a node's parent document. Defaults to "ref_doc_id".
            node_column (str): Column that represents the whole JSON node. Defaults to "node_data".
            stores_text (bool): Whether the table stores text. Defaults to "True".
            overwrite_existing (bool): Whether to drop existing table. Default: False.

        Returns:
            None
        """
        await self._run_as_async(
            self._ainit_vector_store_table(
                table_name,
                vector_size,
                schema_name,
                id_column,
                text_column,
                embedding_column,
                metadata_json_column,
                metadata_columns,
                ref_doc_id_column,
                node_column,
                stores_text,
                overwrite_existing,
            )
        )

    def init_vector_store_table(
        self,
        table_name: str,
        vector_size: int,
        schema_name: str = "public",
        id_column: Union[str, Column] = "node_id",
        text_column: str = "text",
        embedding_column: str = "embedding",
        metadata_json_column: str = "li_metadata",
        metadata_columns: list[Column] = [],
        ref_doc_id_column: str = "ref_doc_id",
        node_column: str = "node_data",
        stores_text: bool = True,
        overwrite_existing: bool = False,
    ) -> None:
        """Create a table for the VectorStore.

        Args:
            table_name (str): The table name to store nodes with embedding vectors.
            vector_size (int): Vector size for the embedding model to be used.
            schema_name (str): The schema name to store the vector store table. Default: "public".
            id_column (str): Column that represents if of a Node. Defaults to "node_id".
            text_column (str): Column that represent text content of a Node. Defaults to "text".
            embedding_column (str): Column for embedding vectors. The embedding is generated from the content of Node. Defaults to "embedding".
            metadata_json_column (str): Column to store metadata as JSON. Defaults to "li_metadata".
            metadata_columns (list[str]): Column(s) that represent extracted metadata keys in their own columns.
            ref_doc_id_column (str): Column that represents id of a node's parent document. Defaults to "ref_doc_id".
            node_column (str): Column that represents the whole JSON node. Defaults to "node_data".
            stores_text (bool): Whether the table stores text. Defaults to "True".
            overwrite_existing (bool): Whether to drop existing table. Default: False.

        Returns:
            None
        """
        self._run_as_sync(
            self._ainit_vector_store_table(
                table_name,
                vector_size,
                schema_name,
                id_column,
                text_column,
                embedding_column,
                metadata_json_column,
                metadata_columns,
                ref_doc_id_column,
                node_column,
                stores_text,
                overwrite_existing,
            )
        )

    async def _ainit_index_store_table(
        self,
        table_name: str,
        schema_name: str = "public",
        overwrite_existing: bool = False,
    ) -> None:
        """
        Create a table to save Index metadata.

        Args:
            table_name (str): The table name to store index metadata.
            schema_name (str): The schema name to store the index metadata table.
                Default: "public".
            overwrite_existing (bool): Whether to drop existing table.
                Default: False.

        Returns:
            None
        """
        if overwrite_existing:
            async with self._pool.connect() as conn:
                await conn.execute(
                    text(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"')
                )
                await conn.commit()

        create_table_query = f"""CREATE TABLE "{schema_name}"."{table_name}"(
            index_id VARCHAR PRIMARY KEY,
            type VARCHAR NOT NULL,
            index_data JSONB NOT NULL
        );"""
        async with self._pool.connect() as conn:
            await conn.execute(text(create_table_query))
            await conn.commit()

    async def ainit_index_store_table(
        self,
        table_name: str,
        schema_name: str = "public",
        overwrite_existing: bool = False,
    ) -> None:
        """
        Create a table to save Index metadata.

        Args:
            table_name (str): The table name to store index metadata.
            schema_name (str): The schema name to store the index metadata table.
                Default: "public".
            overwrite_existing (bool): Whether to drop existing table.
                Default: False.

        Returns:
            None
        """
        await self._run_as_async(
            self._ainit_index_store_table(
                table_name,
                schema_name,
                overwrite_existing,
            )
        )

    def init_index_store_table(
        self,
        table_name: str,
        schema_name: str = "public",
        overwrite_existing: bool = False,
    ) -> None:
        """
        Create a table to save Index metadata.

        Args:
            table_name (str): The table name to store index metadata.
            schema_name (str): The schema name to store the index metadata table.
                Default: "public".
            overwrite_existing (bool): Whether to drop existing table.
                Default: False.

        Returns:
            None
        """
        self._run_as_sync(
            self._ainit_index_store_table(
                table_name,
                schema_name,
                overwrite_existing,
            )
        )

    async def _ainit_chat_store_table(
        self,
        table_name: str,
        schema_name: str = "public",
        overwrite_existing: bool = False,
    ) -> None:
        """Create a table to save chat store.

        Args:
            table_name (str): The table name to store chat history.
            schema_name (str): The schema name to store the chat store table.
                Default: "public".
            overwrite_existing (bool): Whether to drop existing table.
                Default: False.

        Returns: None
        """
        if overwrite_existing:
            async with self._pool.connect() as conn:
                await conn.execute(
                    text(f'DROP TABLE IF EXISTS "{schema_name}"."{table_name}"')
                )
                await conn.commit()

        create_table_query = f"""CREATE TABLE "{schema_name}"."{table_name}"(
            id SERIAL PRIMARY KEY,
            key VARCHAR NOT NULL,
            message JSON NOT NULL
        );"""
        create_index_query = f"""CREATE INDEX "{table_name}_idx_key" ON "{schema_name}"."{table_name}" (key);"""
        async with self._pool.connect() as conn:
            await conn.execute(text(create_table_query))
            await conn.execute(text(create_index_query))
            await conn.commit()

    async def ainit_chat_store_table(
        self,
        table_name: str,
        schema_name: str = "public",
        overwrite_existing: bool = False,
    ) -> None:
        """Create a table to save chat store.

        Args:
            table_name (str): The table name to store chat store.
            schema_name (str): The schema name to store the chat store table.
                Default: "public".
            overwrite_existing (bool): Whether to drop existing table.
                Default: False.
        Returns: None
        """
        await self._run_as_async(
            self._ainit_chat_store_table(
                table_name,
                schema_name,
                overwrite_existing,
            )
        )

    def init_chat_store_table(
        self,
        table_name: str,
        schema_name: str = "public",
        overwrite_existing: bool = False,
    ) -> None:
        """Create a table to save chat store.

        Args:
            table_name (str): The table name to store chat store.
            schema_name (str): The schema name to store the chat store table.
                Default: "public".
            overwrite_existing (bool): Whether to drop existing table.
                Default: False.

        Returns: None
        """
        self._run_as_sync(
            self._ainit_chat_store_table(
                table_name,
                schema_name,
                overwrite_existing,
            )
        )

    async def _aload_table_schema(
        self, table_name: str, schema_name: str = "public"
    ) -> Table:
        """
        Load table schema from an existing table in a PgSQL database, potentially from a specific database schema.

        Args:
            table_name: The name of the table to load the table schema from.
            schema_name: The name of the database schema where the table resides.
                Default: "public".

        Returns:
            (sqlalchemy.Table): The loaded table, including its table schema information.
        """
        metadata = MetaData()
        async with self._pool.connect() as conn:
            try:
                await conn.run_sync(
                    metadata.reflect, schema=schema_name, only=[table_name]
                )
            except InvalidRequestError as e:
                raise ValueError(
                    f"Table, '{schema_name}'.'{table_name}', does not exist: " + str(e)
                )

        table = Table(table_name, metadata, schema=schema_name)
        # Extract the schema information
        schema = []
        for column in table.columns:
            schema.append(
                {
                    "name": column.name,
                    "type": column.type.python_type,
                    "max_length": getattr(column.type, "length", None),
                    "nullable": not column.nullable,
                }
            )

        return metadata.tables[f"{schema_name}.{table_name}"]
