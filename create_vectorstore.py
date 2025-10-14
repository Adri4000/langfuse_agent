from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


docs1 = TextLoader("data/books/alice_in_wonderland.txt").load()
docs2 = TextLoader("data/books/moby_dick.txt").load()
docs3 = TextLoader("data/books/government_position.txt").load()

# Combine into one list
all_docs = docs1 + docs2 + docs3

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,   # max ~500 tokens per chunk (roughly 1000-2000 chars)
    chunk_overlap=20  # overlap between chunks, helps with context continuity
)
split_docs = text_splitter.split_documents(all_docs)


# Build vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(split_docs, embeddings, persist_directory="data/chroma_db")