# from langchain.document_loaders import PyPDFLoader
# from langchain.text_splitters import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain_ollama import OllamaLLM


# loader = PyPDFLoader("path/to/your/pdf.pdf")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# all_chunks = text_splitter.split_documents(documents)
# embeddings = OpenAIEmbeddings()
# vectorstore = Chroma.from_documents(all_chunks, embeddings)
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# List of URLs to load documents from
urls = [
    "<https://lilianweng.github.io/posts/2023-06-23-agent/>",
    "<https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/>",
    "<https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/>",
]
# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
# llm = OllamaLLM(model="llama3.2")
# response = llm.invoke("Hello, tell me a programmer joke")
# print(response)