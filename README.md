# langchain-gel

Gel integration package for the LangChain framework.


## Getting started 

### 1. Install Gel CLI

```bash
brew install gel
```

### 2. Create a Gel project

For the cloud:

```bash
edgedb project init --server-instance <org-name>/<instance-name>
```

Locally:

```bash
gel project init
```

and install the Vectorstore extension

```bash
gel extension install vectorstore -I example
```

### 3. Enable the Vectorstore extension in the schema

```gel
using extension vectorstore
```

### 4. Install the Gel LangChain integration

```bash
pip3 install langchain-gel
```

### 5. Import and connect the `GelVectorstore`

```python
from langchain_gel import GelVectorStore

vectorstore = GelVectorStore.from_texts()
```

### 6. Run similarity search

```python
vectorstore.query()
```

### 7. Connect to a RAG application

```python
# example RAG chain featuring the vectorstore
```
