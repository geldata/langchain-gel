from typing import Generator
import subprocess

import pytest
from langchain_gel.vectorstore import GelVectorStore
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

try:
    subprocess.run(["gel", "project", "init", "--non-interactive"], check=True)
except subprocess.CalledProcessError as e:
    print(e)

class TestGelVectorStoreSync(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""

        store = GelVectorStore(self.get_embeddings())
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        store.delete()
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            store.delete()

    @property
    def has_sync(self) -> bool:
        """
        Configurable property to enable or disable sync tests.
        """
        return True

    @property
    def has_async(self) -> bool:
        """
        Configurable property to enable or disable async tests.
        """
        return False


class TestGelVectorStoreAsync(VectorStoreIntegrationTests):
    @pytest.fixture()
    async def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""

        store = GelVectorStore(self.get_embeddings())
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        await store.adelete()
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            await store.adelete()

    @property
    def has_sync(self) -> bool:
        """
        Configurable property to enable or disable sync tests.
        """
        return False

    @property
    def has_async(self) -> bool:
        """
        Configurable property to enable or disable async tests.
        """
        return True
