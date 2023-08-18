**Install Azure Cognitive Search SDK**


```python
!pip install --index-url=https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple/ azure-search-documents==11.4.0a20230509004
!pip install azure-identity
```

**Import required libraries**


```python
import os, json
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings("ignore")
```

**Configure OpenAI and vector store settings**


```python
openai.api_type = os.getenv('OPENAI_API_TYPE')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version = os.getenv('OPENAI_API_VERSION')
# #openai.api_key = os.getenv('OPENAI_API_KEY')
model: str = "text-embedding-ada-002"
vector_store_address = os.getenv('AZURE_SEARCH_ENDPOINT')
vector_store_password = os.getenv('AZURE_SEARCH_ADMIN_KEY')
index_name: str = "langchain-vector-demo"
```

**Create embeddings and vector store instances**


```python
embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=model, chunk_size=1)
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=vector_store_address,
    azure_search_key=vector_store_password,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
)
```


```python
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
```


```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("data/SparkOfGPT-4.pdf")
# pages = loader.load_and_split()
```

**Insert text and embeddings into vector store**


```python
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

vector_store.add_documents(documents=docs)
```

**Perform a vector similarity search**


```python
# Perform a similarity search
docs = vector_store.similarity_search(
    query="What are the areas of improvement for GPT-4?",
    k=3,
    search_type="similarity",
)
print(docs[0].page_content)
```

**Hybrid Search**


```python
# Perform a hybrid search
docs = vector_store.similarity_search(
    query="What are the areas of improvement for GPT-4 over GPT-3.5?",
    k=5,
    search_type="hybrid",
)
print(docs[0].page_content)
```


```python
len(docs)
```


```python
docs[0]
```

**Use llm to answers the query**


```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import CSVLoader
# from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
```


```python
llm = ChatOpenAI(engine='gpt-4', temperature = 0.0)
```


```python
qdocs = "".join([docs[i].page_content for i in range(len(docs))])
```


```python
response = llm.call_as_llm(f"{qdocs} Question: What are the areas \
of improvement for GPT-4 over GPT-3.5?") 
```


```python
display(Markdown(response))
```
