import unittest
from langchain_core.documents import Document
from langchain_gel.vectorstore import filter_to_edgeql

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

    
