Cloud SQL for PostgreSQL for LlamaIndex

|preview| |pypi| |versions|

- `Product Documentation`_

The **Cloud SQL for PostgreSQL for LlamaIndex** package provides a first class experience for connecting to
Cloud SQL instances from the LlamaIndex ecosystem while providing the following benefits:

- **Simplified & Secure Connections**: easily and securely create shared connection pools to connect to Google Cloud databases utilizing IAM for authorization and database authentication without needing to manage SSL certificates, configure firewall rules, or enable authorized networks.
- **Improved metadata handling**: store metadata in columns instead of JSON, resulting in significant performance improvements.
- **Clear separation**: clearly separate table and extension creation, allowing for distinct permissions and streamlined workflows.

.. |preview| image:: https://img.shields.io/badge/support-preview-orange.svg
   :target: https://github.com/googleapis/google-cloud-python/blob/main/README.rst#stability-levels
.. |pypi| image:: https://img.shields.io/pypi/v/llama-index-cloud-sql-pg.svg
   :target: https://pypi.org/project/llama-index-cloud-sql-pg/
.. |versions| image:: https://img.shields.io/pypi/pyversions/llama-index-cloud-sql-pg.svg
   :target: https://pypi.org/project/llama-index-cloud-sql-pg/
.. _Product Documentation: https://cloud.google.com/sql/docs

Quick Start
-----------

In order to use this library, you first need to go through the following
steps:

1. `Select or create a Cloud Platform project.`_
2. `Enable billing for your project.`_
3. `Enable the Cloud SQL Admin API.`_
4. `Setup Authentication.`_

.. _Select or create a Cloud Platform project.: https://console.cloud.google.com/project
.. _Enable billing for your project.: https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project
.. _Enable the Cloud SQL Admin API.: https://console.cloud.google.com/flows/enableapi?apiid=sqladmin.googleapis.com
.. _Setup Authentication.: https://googleapis.dev/python/google-api-core/latest/auth.html

Installation
~~~~~~~~~~~~

Install this library in a `virtualenv`_ using pip. `virtualenv`_ is a tool to create isolated Python environments. The basic problem it addresses is
one of dependencies and versions, and indirectly permissions.

With `virtualenv`_, it's
possible to install this library without needing system install
permissions, and without clashing with the installed system
dependencies.

.. _`virtualenv`: https://virtualenv.pypa.io/en/latest/

Supported Python Versions
^^^^^^^^^^^^^^^^^^^^^^^^^

Python >= 3.9

Mac/Linux
^^^^^^^^^

.. code-block:: console

   pip install virtualenv
   virtualenv <your-env>
   source <your-env>/bin/activate
   <your-env>/bin/pip install llama-index-cloud-sql-pg

Windows
^^^^^^^

.. code-block:: console

    pip install virtualenv
    virtualenv <your-env>
    <your-env>\Scripts\activate
    <your-env>\Scripts\pip.exe install llama-index-cloud-sql-pg

Example Usage
-------------

Code samples and snippets live in the `samples/`_ folder.

.. _samples/: https://github.com/googleapis/llama-index-cloud-sql-pg-python/tree/main/samples

Vector Store Usage
~~~~~~~~~~~~~~~~~~~

Use a vector store to store embedded data and perform vector search.

.. code-block:: python

   import google.auth
   from llama_index.core import Settings
   from llama_index.embeddings.vertex import VertexTextEmbedding
   from llama_index_cloud_sql_pg import PostgresEngine, PostgresVectorStore


   credentials, project_id = google.auth.default()
   engine = await PostgresEngine.afrom_instance(
      "project-id", "region", "my-cluster", "my-instance", "my-database"
   )
   Settings.embed_model = VertexTextEmbedding(
      model_name="textembedding-gecko@003",
      project="project-id",
      credentials=credentials,
   )

   vector_store = await PostgresVectorStore.create(
      engine=engine, table_name="vector_store"
   )


Document Store Usage
~~~~~~~~~~~~~~~~~~~~~

Use a document store to make storage and maintenance of data easier.

.. code-block:: python

   from llama_index_cloud_sql_pg import PostgresEngine, PostgresDocumentStore


   engine = await PostgresEngine.afrom_instance(
      "project-id", "region", "my-cluster", "my-instance", "my-database"
   )
   doc_store = await PostgresDocumentStore.create(
      engine=engine, table_name="doc_store"
   )


Index Store Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use an index store to keep track of indexes built on documents.

.. code:: python

   from llama_index_cloud_sql_pg import PostgresIndexStore, PostgresEngine


   engine = await PostgresEngine.from_instance(
      "project-id", "region", "my-cluster", "my-instance", "my-database"
   )
   index_store = await PostgresIndexStore.create(
      engine=engine, table_name="index_store"
   )


Contributions
~~~~~~~~~~~~~

Contributions to this library are always welcome and highly encouraged.

See `CONTRIBUTING`_ for more information how to get started.

Please note that this project is released with a Contributor Code of Conduct. By participating in
this project you agree to abide by its terms. See `Code of Conduct`_ for more
information.

.. _`CONTRIBUTING`: https://github.com/googleapis/llama-index-cloud-sql-pg-python/tree/main/CONTRIBUTING.md
.. _`Code of Conduct`: https://github.com/googleapis/llama-index-cloud-sql-pg-python/tree/main/CODE_OF_CONDUCT.md

License
-------

Apache 2.0 - See
`LICENSE <https://github.com/googleapis/llama-index-cloud-sql-pg-python/tree/main/LICENSE>`_
for more information.

Disclaimer
----------

This is not an officially supported Google product.
