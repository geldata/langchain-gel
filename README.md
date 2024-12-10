# langchain-gel

Gel integration package for the LangChain framework.


## Getting started in the cloud



## Getting started locally

1. Install Gel CLI

```bash
brew install edgedb
```

2. Create a Gel project

```bash
gel project init
```

3. Install the Vectorstore extension

```bash
gel extension install vectorstore -I example
```

4. Enable the Vectorstore extension in the schema

```gel
using extension vectorstore
```

5. Install the Gel LangChain integration

```bash
pip3 install langchain-gel
```

6. Import and connect the `GelVectorstore`

```python
from langchain_gel import GelVectorStore

vectorstore = GelVectorStore.from_texts()
```

7. Run similarity search

```python
vectorstore.query()
```

8. Connect to a RAG application

```python
# example RAG chain featuring the vectorstore
```
