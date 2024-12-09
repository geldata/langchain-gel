import unittest
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_vectorstore import EdgeDBVectorStore, filter_to_edgeql

from dotenv import load_dotenv

load_dotenv()


DOCS = [
    Document(
        id="1",
        page_content="there are cats in the pond",
        metadata={"location": "pond", "topic": "animals"},
    ),
    Document(
        id="2",
        page_content="ducks are also found in the pond",
        metadata={"location": "pond", "topic": "animals"},
    ),
    Document(
        id="3",
        page_content="fresh apples are available at the market",
        metadata={"location": "market", "topic": "food"},
    ),
    Document(
        id="4",
        page_content="the market also sells fresh oranges",
        metadata={"location": "market", "topic": "food"},
    ),
    Document(
        id="5",
        page_content="the new art exhibit is fascinating",
        metadata={"location": "museum", "topic": "art"},
    ),
    Document(
        id="6",
        page_content="a sculpture exhibit is also at the museum",
        metadata={"location": "museum", "topic": "art"},
    ),
    Document(
        id="7",
        page_content="a new coffee shop opened on Main Street",
        metadata={"location": "Main Street", "topic": "food"},
    ),
    Document(
        id="8",
        page_content="the book club meets at the library",
        metadata={"location": "library", "topic": "reading"},
    ),
    Document(
        id="9",
        page_content="the library hosts a weekly story time for kids",
        metadata={"location": "library", "topic": "reading"},
    ),
    Document(
        id="10",
        page_content="a cooking class for beginners is offered at the community center",
        metadata={"location": "community center", "topic": "classes"},
    ),
]


class TestEdgeDBVectorStore(unittest.TestCase):

    def setUp(self):

        # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002", api_version="2023-07-01-preview"
        )

        self.vectorstore = EdgeDBVectorStore(
            embeddings=self.embeddings, collection_name="test", record_type="Record"
        )
        self.docs = DOCS

    def tearDown(self):
        self.vectorstore.delete()
        self.vectorstore.client.close()

    def test_filter_to_edgeql(self):
        test_cases = [
            ({"field": "value"}, '<str>json_get(.metadata, "field") = "value"'),
            ({"field": 1}, '<str>json_get(.metadata, "field") = 1'),
            (
                {"$eq": {"field": "value"}},
                '<str>json_get(.metadata, "field") = "value"',
            ),
            ({"$eq": {"field": 1}}, '<str>json_get(.metadata, "field") = 1'),
            (
                {"$ne": {"field": "value"}},
                '<str>json_get(.metadata, "field") != "value"',
            ),
            (
                {"$lt": {"field": "value"}},
                '<str>json_get(.metadata, "field") < "value"',
            ),
            (
                {"$lte": {"field": "value"}},
                '<str>json_get(.metadata, "field") <= "value"',
            ),
            (
                {"$gt": {"field": "value"}},
                '<str>json_get(.metadata, "field") > "value"',
            ),
            (
                {"$gte": {"field": "value"}},
                '<str>json_get(.metadata, "field") >= "value"',
            ),
            (
                {"$in": {"field": [1, 2, 3]}},
                '<str>json_get(.metadata, "field") in array_unpack([1, 2, 3])',
            ),
            (
                {"$nin": {"field": [1, 2, 3]}},
                '<str>json_get(.metadata, "field") not in array_unpack([1, 2, 3])',
            ),
            (
                {"$like": {"field": "pattern"}},
                '<str>json_get(.metadata, "field") like "pattern"',
            ),
            (
                {"$ilike": {"field": "pattern"}},
                '<str>json_get(.metadata, "field") ilike "pattern"',
            ),
            (
                {"$and": [{"field1": "value1"}, {"field2": "value2"}]},
                '(<str>json_get(.metadata, "field1") = "value1" and <str>json_get(.metadata, "field2") = "value2")',
            ),
            (
                {"$or": [{"field1": "value1"}, {"field2": "value2"}]},
                '(<str>json_get(.metadata, "field1") = "value1" or <str>json_get(.metadata, "field2") = "value2")',
            ),
            (
                {
                    "$and": [
                        {
                            "$or": [
                                {"$in": {"field1": [1, 2, 3]}},
                                {"$gt": {"field2": 100}},
                            ]
                        },
                        {"$like": {"field3": "%pattern%"}},
                    ]
                },
                '((<str>json_get(.metadata, "field1") in array_unpack([1, 2, 3]) or <str>json_get(.metadata, "field2") > 100) and <str>json_get(.metadata, "field3") like "%pattern%")',
            ),
        ]

        for filter_dict, expected in test_cases:
            self.assertEqual(filter_to_edgeql(filter_dict), expected)

    def test_add_documents(self):
        inserted_ids = self.vectorstore.add_documents(self.docs)

        self.assertEqual(len(inserted_ids), len(self.docs))

        for doc in self.docs:
            self.assertIn(doc.id, inserted_ids)

    def test_similarity_search(self):
        inserted_ids = self.vectorstore.add_documents(self.docs)
        results = self.vectorstore.similarity_search("cats", k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "1")

        results = self.vectorstore.similarity_search(
            "coffee",
            k=2,
            filter={
                "$and": [{"location": "market"}, {"$in": {"$topic": ["food", "art"]}}]
            },
        )
        self.assertNotIn("7", [r.id for r in results])

    def test_get_by_ids(self):
        inserted_ids = self.vectorstore.add_documents(self.docs)
        results = self.vectorstore.get_by_ids(["1", "2"])

        self.assertEqual(len(results), 2)

        self.assertEqual(results[0].id, "1")
        self.assertEqual(results[1].id, "2")

    def test_delete(self):
        inserted_ids = self.vectorstore.add_documents(self.docs)

        self.vectorstore.delete(ids=["1", "2"])
        results = self.vectorstore.get_by_ids(ids=["1", "2"])
        self.assertEqual(len(results), 0)
        results = self.vectorstore.get_by_ids(ids=["3", "4"])
        self.assertEqual(len(results), 2)

        self.vectorstore.delete()
        results = self.vectorstore.get_by_ids(ids=["3", "4"])
        self.assertEqual(len(results), 0)


class TestEdgeDBVectorStoreAsync(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002", api_version="2023-07-01-preview"
        )

        self.vectorstore = EdgeDBVectorStore(
            embeddings=self.embeddings,
            collection_name="test",
            record_type="Record",
            use_async=True,
        )
        self.docs = DOCS

    async def asyncTearDown(self) -> None:
        await self.vectorstore.adelete()
        await self.vectorstore.client.aclose()

    async def test_aadd_documents(self):
        inserted_ids = await self.vectorstore.aadd_documents(self.docs)

        self.assertEqual(len(inserted_ids), len(self.docs))

        for doc in self.docs:
            self.assertIn(doc.id, inserted_ids)

    async def test_asimilarity_search(self):
        inserted_ids = await self.vectorstore.aadd_documents(self.docs)
        results = await self.vectorstore.asimilarity_search("cats", k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "1")

    async def test_aget_by_ids(self):
        inserted_ids = await self.vectorstore.aadd_documents(self.docs)
        results = await self.vectorstore.aget_by_ids(["1", "2"])

        self.assertEqual(len(results), 2)

        self.assertEqual(results[0].id, "1")
        self.assertEqual(results[1].id, "2")

    async def test_adelete(self):
        inserted_ids = await self.vectorstore.aadd_documents(self.docs)

        await self.vectorstore.adelete(ids=["1", "2"])
        results = await self.vectorstore.aget_by_ids(ids=["1", "2"])
        self.assertEqual(len(results), 0)
        results = await self.vectorstore.aget_by_ids(ids=["3", "4"])
        self.assertEqual(len(results), 2)

        await self.vectorstore.adelete()
        results = await self.vectorstore.aget_by_ids(ids=["3", "4"])
        self.assertEqual(len(results), 0)
