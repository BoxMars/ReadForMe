from pypdf import PdfReader
from llama_index.readers import Document
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os
os.environ["OPENAI_API_KEY"] ='sk-xxx'
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_outputs = 2000
# set maximum chunk overlap
max_chunk_overlap = 20
# set chunk size limit
chunk_size_limit = 600

path='path/to/your/file'

reader = PdfReader(path)
documents=[]
for page in reader.pages:
    documents.append(Document(page.extract_text().replace('\n','')))
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

while True:
    query = input("What do you want to ask? ")
    response = index.query(query, response_mode="compact")
    print(response.response)
