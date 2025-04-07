import pytest
from langchain_gel.vectorstore import filter_to_edgeql

# Test cases with the correct protocol
@pytest.mark.parametrize("filter_dict,expected", [
    # Simple equality
    ({"field": "value"}, '<str>json_get(.metadata, "field") = "value"'),
    ({"field": 1}, '<str>json_get(.metadata, "field") = 1'),
    
    # Field with operators
    ({"field": {"$eq": "value"}}, '<str>json_get(.metadata, "field") = "value"'),
    ({"field": {"$eq": 1}}, '<str>json_get(.metadata, "field") = 1'),
    ({"field": {"$ne": "value"}}, '<str>json_get(.metadata, "field") != "value"'),
    ({"field": {"$lt": "value"}}, '<str>json_get(.metadata, "field") < "value"'),
    ({"field": {"$lte": "value"}}, '<str>json_get(.metadata, "field") <= "value"'),
    ({"field": {"$gt": "value"}}, '<str>json_get(.metadata, "field") > "value"'),
    ({"field": {"$gte": "value"}}, '<str>json_get(.metadata, "field") >= "value"'),
    ({"field": {"$in": [1, 2, 3]}}, '<str>json_get(.metadata, "field") in array_unpack([1, 2, 3])'),
    ({"field": {"$nin": [1, 2, 3]}}, '<str>json_get(.metadata, "field") not in array_unpack([1, 2, 3])'),
    ({"field": {"$like": "pattern"}}, '<str>json_get(.metadata, "field") like "pattern"'),
    ({"field": {"$ilike": "pattern"}}, '<str>json_get(.metadata, "field") ilike "pattern"'),
    
    # Logical operators
    ({"$and": [{"field1": "value1"}, {"field2": "value2"}]}, 
     '(<str>json_get(.metadata, "field1") = "value1" and <str>json_get(.metadata, "field2") = "value2")'),
    ({"$or": [{"field1": "value1"}, {"field2": "value2"}]}, 
     '(<str>json_get(.metadata, "field1") = "value1" or <str>json_get(.metadata, "field2") = "value2")'),
    
    # Complex nested operators
    ({
        "$and": [
            {
                "$or": [
                    {"field1": {"$in": [1, 2, 3]}},
                    {"field2": {"$gt": 100}},
                ]
            },
            {"field3": {"$like": "%pattern%"}},
        ]
    },
    '((<str>json_get(.metadata, "field1") in array_unpack([1, 2, 3]) or <str>json_get(.metadata, "field2") > 100) and <str>json_get(.metadata, "field3") like "%pattern%")'),
    
    # Multiple field filters (implicit AND)
    ({"field1": "value1", "field2": "value2"}, 
     '(<str>json_get(.metadata, "field1") = "value1" and <str>json_get(.metadata, "field2") = "value2")'),
    
    # Multiple field filters with operators
    ({"field1": {"$in": [1, 2, 3]}, "field2": {"$gt": 100}}, 
     '(<str>json_get(.metadata, "field1") in array_unpack([1, 2, 3]) and <str>json_get(.metadata, "field2") > 100)'),
])
def test_filter_to_edgeql(filter_dict, expected):
    assert filter_to_edgeql(filter_dict) == expected

    
