from dotenv import load_dotenv
import os
from langchain.document_loaders import UnstructuredURLLoader
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from time import sleep

#
# Prepare environment
#

load_dotenv()

#
# Define list of articles to summarize
#

url_list_small = [
    # Ukrainian sources
    "https://uk.wikipedia.org/wiki/1%D0%9C",
    "https://uk.wikipedia.org/wiki/%D0%92%D0%B0%D0%B6%D0%BA%D0%B8%D0%B9_%D1%81%D1%83%D0%BF%D1%83%D1%82%D0%BD%D0%B8%D0%BA_01",
    "https://uk.wikipedia.org/wiki/%D0%86%D0%A1%D0%97_%D0%94-1_%E2%84%96_1"
    ]

url_list_full = [
    # Ukrainian sources
    "https://uk.wikipedia.org/wiki/1%D0%9C",
    "https://uk.wikipedia.org/wiki/%D0%92%D0%B0%D0%B6%D0%BA%D0%B8%D0%B9_%D1%81%D1%83%D0%BF%D1%83%D1%82%D0%BD%D0%B8%D0%BA_01",
    "https://uk.wikipedia.org/wiki/%D0%86%D0%A1%D0%97_%D0%94-1_%E2%84%96_1",
    "https://uk.wikipedia.org/wiki/%D0%9A%D0%BE%D1%80%D0%B0%D0%B1%D0%B5%D0%BB%D1%8C-%D1%81%D1%83%D0%BF%D1%83%D1%82%D0%BD%D0%B8%D0%BA-1%D0%9A_%E2%84%96_1",
    "https://uk.wikipedia.org/wiki/%D0%9A%D0%BE%D1%80%D0%B0%D0%B1%D0%B5%D0%BB%D1%8C-%D1%81%D1%83%D0%BF%D1%83%D1%82%D0%BD%D0%B8%D0%BA-1%D0%9A_%E2%84%96_4",
    "https://uk.wikipedia.org/wiki/%D0%9A%D0%BE%D1%81%D0%BC%D0%BE%D1%81-60",
    "https://uk.wikipedia.org/wiki/%D0%9A%D0%BE%D1%81%D0%BC%D0%BE%D1%81-359",
    "https://uk.wikipedia.org/wiki/%D0%9A%D0%BE%D1%81%D0%BC%D0%BE%D1%81-482",
    "https://uk.wikipedia.org/wiki/%D0%9B%D1%83%D0%BD%D0%B0-1A",
    "https://uk.wikipedia.org/wiki/%D0%9B%D1%83%D0%BD%D0%B0-1B",
    "https://uk.wikipedia.org/wiki/%D0%9B%D1%83%D0%BD%D0%B0-1C",
    "https://uk.wikipedia.org/wiki/%D0%9B%D1%83%D0%BD%D0%B0-2A",
    "https://uk.wikipedia.org/wiki/%D0%9B%D1%83%D0%BD%D0%B0-4A",
    "https://uk.wikipedia.org/wiki/%D0%9B%D1%83%D0%BD%D0%B0-4B",
    "https://uk.wikipedia.org/wiki/%D0%9C%D0%B0%D1%80%D1%81-1%D0%9C_%E2%84%96_1",
    "https://uk.wikipedia.org/wiki/%D0%9C%D0%B0%D1%80%D1%81-1%D0%9C_%E2%84%96_2",
    "https://uk.wikipedia.org/wiki/%D0%9F%D0%BE%D0%BB%D1%8E%D1%81_(%D0%BA%D0%BE%D1%81%D0%BC%D1%96%D1%87%D0%BD%D0%B8%D0%B9_%D0%B0%D0%BF%D0%B0%D1%80%D0%B0%D1%82)",
    "https://uk.wikipedia.org/wiki/%D0%A1%D0%BE%D1%8E%D0%B7-18%D0%B0",
    # English sources from page https://en.wikipedia.org/wiki/Category:Space_accidents_and_incidents_in_the_Soviet_Union
    "https://en.wikipedia.org/wiki/1980_Plesetsk_launch_pad_disaster",
    "https://en.wikipedia.org/wiki/Kosmos_96",
    "https://en.wikipedia.org/wiki/Kosmos_167",
    "https://en.wikipedia.org/wiki/Kosmos_482",
    "https://en.wikipedia.org/wiki/Kosmos_1164",
    "https://en.wikipedia.org/wiki/Mars_1M_No.1",
    "https://en.wikipedia.org/wiki/Mars_1M_No.2",
    "https://en.wikipedia.org/wiki/N1_(rocket)",
    "https://en.wikipedia.org/wiki/Nedelin_catastrophe",
    "https://en.wikipedia.org/wiki/Polyus_(spacecraft)",
    "https://en.wikipedia.org/wiki/Soyuz_1",
    "https://en.wikipedia.org/wiki/Soyuz_2A",
    "https://en.wikipedia.org/wiki/Soyuz_7K-ST_No.16L",
    "https://en.wikipedia.org/wiki/Soyuz_7K-T_No.39",
    "https://en.wikipedia.org/wiki/Soyuz_11",
    #English sources from page https://en.wikipedia.org/wiki/Category:Space_accidents_and_incidents_in_Russia
    "https://en.wikipedia.org/wiki/CryoSat-1",
    "https://en.wikipedia.org/wiki/Foton-M_No.1",
    "https://en.wikipedia.org/wiki/Kosmos_2470",
    "https://en.wikipedia.org/wiki/Meridian_2",
    "https://en.wikipedia.org/wiki/Meridian_5",
    # English sources from page https://en.wikipedia.org/wiki/Category:Space_accidents_and_incidents_in_Kazakhstan
    "https://en.wikipedia.org/wiki/Astra_1K",
    "https://en.wikipedia.org/wiki/BelKA",
    "https://en.wikipedia.org/wiki/Ekspress-AM4",
    "https://en.wikipedia.org/wiki/ION_(satellite)",
    "https://en.wikipedia.org/wiki/Progress_M-12M",
    "https://en.wikipedia.org/wiki/Progress_M-27M",
    "https://en.wikipedia.org/wiki/Progress_MS-04",
    "https://en.wikipedia.org/wiki/Rincon_1",
    "https://en.wikipedia.org/wiki/SACRED",
    "https://en.wikipedia.org/wiki/Soyuz_MS-10"
    ]

#
# Load html documents
#

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
loader = UnstructuredURLLoader(url_list_full)
docs = loader.load()
print("Downloaded {} pages.".format(len(docs)))

#
# Set language model (Not that for 3.5 model the ChatOpenAI object is needed. For v3 text models enough to use OpenAI object)
#

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

#
# Prepare custom prompt
#

prompt_template = """Write a concise summary. Output language is Ukrainian.
Please, cartridge return symbols in answer to split in on 
80 characters long rows.
Summary should contain the following information:
 - date of launch or incident
 - space ship name
 - astronauts name if it have pilots
 - description of incident with details
Text for analysis

{text}

Text of output language is Ukrainian"""

result_template = """
Write a concise summary. Output language is Ukrainian.
We have provided an existing summary up to a certain point: {existing_answer}\n"
Summary should contain the following information:
 - date of launch or incident
 - space ship name
 - COSPAR ID
 - astronauts name if it have pilots
 - Nice description of the incident with posssible details
Text for analysis

{text}

Give output in CSV file with following fields:
<date>,<space_ship_name>,<COSPAR ID>, <astronauts_name>,<summary>
Use an empty text string for missing fields
Use quotation marks for text fields and semicolon as separator
Text language is Ukrainian"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
RESULT_PROMPT = PromptTemplate(template=result_template, input_variables=["text","existing_answer"])

#
# Prepare summarization chain
#

#chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=PROMPT, combine_prompt=RESULT_PROMPT)
chain = load_summarize_chain(llm, chain_type="refine", question_prompt=PROMPT, refine_prompt=RESULT_PROMPT)


#
# Prepare output document
#

results_table = []
f = open("output_1.csv", "w")
f.write("\"Date\"; \"Space ship name\"; \"COSPAR ID\"; \"Astronauts name(s)\"; \"Summary\"; \"Source\"\n")

#
# Run summarization chain for each document
#

for i, doc in enumerate(docs):
    print(f'Document {i}:')
    print(f'Source: {doc.metadata["source"]}')
    # split to avoid long prompts
    documents = splitter.split_documents([doc])
    # run inference
    result = chain.run(documents)
    print(f'Summary length: {len(result.splitlines())}')
    print(f'Summary: \n {result}')
    for row in result.splitlines():
        record = row+ ";"+f'\"{doc.metadata["source"]}\"\n'
        f.write(record)
    # Wait 5 sec to avoid reaching request rate limit
    sleep(5)
f.close()



