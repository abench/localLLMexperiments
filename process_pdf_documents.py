
from dotenv import load_dotenv
import os

from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever

from langchain.chains.query_constructor.base import AttributeInfo
# Load environment variables

load_dotenv()

# load pdf documents

#pdf_path = ".data/priklyucheniya_na_orbitah.pdf"
pdf_path = ".data/vladimirov-finkelstein_sovetsky_kosmichesky_blef_1973__ocr.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()
#splitter = CharacterTextSplitter(chunk_size=100,chunk_overlap=20,length_function=len)
#splitter = NLTKTextSplitter(chunk_size=100)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
pages = splitter.split_documents(documents)
#print(len(pages))
#print(pages)

# load html documents

embeddings = OpenAIEmbeddings()
#for page in pages:
#    print(page)
vectordb = Chroma.from_documents(pages, embedding=embeddings, persist_directory=".state/")
vectordb.persist()
metadata_field_info=[
    AttributeInfo(
        name="page",
        description="Number of document page",
        type="integer",
    ),
    AttributeInfo(
        name="source",
        description="Source file name",
        type="string",
    ),
]
document_content_description = "історія космічних досліджень"
llm = OpenAI(temperature=0)
retriever = SelfQueryRetriever.from_llm(llm,
                                        vectordb,
                                        document_content_description,
                                        metadata_field_info,
                                        verbose=True)

search_items = ["катастрофа", "вибух", "загибель", "аварія", "смерть", "нещасний випадок"]

for search_item in search_items:
    results = retriever.get_relevant_documents(search_item)
    for doc in results:
        print(doc.metadata["page"])
        print("-----")
        print(doc.page_content)




# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectordb.as_retriever(), memory=memory,
#                                                #reduce_k_below_max_tokens=True
#                                                )
#
# query = "згадується  нещасний випадок, катастрофа, вибух чи аварія? Якщо так, то вкажи дату, прізвища загиблих, подробиці. Перепахуй усі випадки. Відповідай українською."
# result = pdf_qa({"question": query})
# print("Answer:")
# print(result["answer"])
# print(result)

