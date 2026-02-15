import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Directory containing data files
data_folder = "data/"
all_doc_splits = []

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=150, chunk_overlap=20
)

print("Reading and splitting files...")

# Read each file in 'data/'
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(data_folder, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Create chunks
            chunks = text_splitter.split_text(content)

            for chunk in chunks:
                new_doc = Document(
                    page_content=chunk,
                    metadata={"source": filename,
                              "topic": filename.replace(".txt", "")}
                )

                all_doc_splits.append(new_doc)

print(f"Total chunks generated: {len(all_doc_splits)}")

# Create embeddings for documents and store them in a vector store
print("Embedding chunks...")
vectorstore = SKLearnVectorStore.from_texts(
    texts=all_doc_splits,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
)
print("Chunks embeded.")
retriever = vectorstore.as_retriever(k=5)

prompt = PromptTemplate.from_template(
    """
    You are an expert in the rules of Magic the Gathering.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    When giving an answer, provide the ultimate ruling first. 
    Then, list out the rules and their contents in a list fashion so the user knows where the information was referenced from. 
    Keep the ruling to the question compact and concise. 
    Make the list of rules easy to read.:
    Question: {question}
    Documents: {documents}
    Answer:
    """
)

# Initialize the LLM with Llama 3.1 model
llm = ChatOllama(
    model="llama3.1",
    temperature=0.5,
)

# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()

# Define the RAG application class


class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke(
            {"question": question, "documents": doc_texts})
        return answer


# Initialize the RAG application
print("Initializing RAG...")
rag_application = RAGApplication(retriever, rag_chain)
print("RAG initialized.\n")
running = True
while running:
    # Example usage
    question = input("Enter quesiton: ")
    if question == 'quit':
        running = False
    else:
        answer = rag_application.run(question)
        print("Answer:", answer)
    print(150*"=")
    print("\n")
