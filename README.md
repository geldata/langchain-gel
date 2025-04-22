# langchain-gel

This package enables LangChain to interact with Gel as a vectorstore.
See LangChain's [documentation](https://python.langchain.com/docs/concepts/vectorstores/) to learn more about how to take advantage of that.

> Note: check out Gel's [AI extension](https://docs.geldata.com/reference/ai) to learn how to automate embedding management away while taking advantage of poweful schema and EdgeQL query language.

## Usage 

1. Install Gel's Python binding and this package

```bash
pip install gel langchain-gel
```

2. Initialize the project

Locally:

```bash
gel project init
```

In the [cloud](https://docs.geldata.com/cloud):

```bash
gel project init --server-instance <org-name>/<instance-name>

```

3. Add necessary components to the schema. Gel uses explicit schema and migrations, which gives you more control and preserves data integrity. `langchain-gel` expects the following schema: 

```gel
using extension pgvector;
                                    
module default {
    scalar type EmbeddingVector extending ext::pgvector::vector<1536>;

    type Record {
        required collection: str;
        text: str;
        embedding: EmbeddingVector;
        external_id: str {
            constraint exclusive;
        };
        metadata: json;

        index ext::pgvector::hnsw_cosine(m := 16, ef_construction := 128)
            on (.embedding)
    } 
}
```
Copy-paste this to `dbschema/default.gel` and run a migration:

```bash
gel migration create \
&& gel migrate
```

4. Use `GelVectorStore` as usual. It's a drop-in replacement for any other vectorstore in the LangChain ecosystem.

```python
from langchain_gel import GelVectorStore

vectorstore = GelVectorStore()
```

## Next steps

When you are ready to migrate to Gel's native vector handling, check out [Gel's documentation](docs.geldata.com) to find instructions.
