from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

import json
import requests
from dotenv import load_dotenv,find_dotenv
from bs4 import BeautifulSoup

#### Langchian ####
#OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

## Prompt 
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

#Document Loader
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.merge import MergedDataLoader
#Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Index
import lancedb
from langchain_community.vectorstores import Chroma, LanceDB
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document


## Others
from langchain_core.runnables import RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#Helpers
from typing import Literal





class RAG_Pipeline:
    def __init__(self):

        
        self.embedding = OpenAIEmbeddings()

        self.master_memory = ''
        self.data_router = self.initialize_data_router()

        self.troubleshoot_chain = self.initialize_troubleshoot_chain()

        self.comptability_chain = self.initialaize_comptability_chain()
        
        self.product_lookup_chain = self.initialize_product_lookup_chain()

        self.default_chain = self.initialaize_default_chain()
        
    #ROUTER
    def initialize_data_router(self):
        # Data model
        class DataRouteQuery(BaseModel):
            """Route a user query to the most relevant datasource."""

            datasource: Literal["product_database_lookup", "troubleshooting_guidelines", "compatability_questions","general"] = Field(
                ...,
                description='''Given a user question choose which datasource would be most relevant for answering their questions
                            here product_database_lookup has information about part id, product id, how to install products, all installation guidelines etc. and
                            troubleshooting  guidelines helps in solving various issues about dishwashers and refrigerators and
                            compatability_questions answers weather a part is compatable with the model and 
                            for any general questions or greetings should go to general
                            Only answer questions for Dishwasher and Refrigerators all other things do not answer ''',
            )
        # LLM with function call
        router_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        structured_llm = router_llm.with_structured_output(DataRouteQuery)

        # Prompt
        self.router_system = """You are an expert at routing a user question to the appropriate data source and a kind chatbot for e-commerce site Part Select

        Based on the help user is asking for route it to the relevant data source."""

        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", system),
        #         ("human", "{question}"),
        #     ]
        # )

        prompt = PromptTemplate.from_template(

        ''' 
        System : {system}

        Human_user, {question}


        Also smartly make use of past conversations: if they exist {memory} to make decision 
        '''
                        
                                            
        )
        # Define router
        data_router = prompt | structured_llm

        return data_router

    #QA
    def initialize_troubleshoot_chain(self):
      
        #Document Loader
        loader = DirectoryLoader('./data', glob="**/*.txt", loader_cls=TextLoader)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=590,
            chunk_overlap=200)

        # Make splits
        splits = text_splitter.split_documents(docs)

        #The data is small so saving locally
        vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=self.embedding)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        template = """ You are an expert person who can repair dishwashers and refrigerators, gi
        Answer the question kindly,, step-by-step, briefly and based only on the following context:
        {context}
        Make sure to summarize slightly 
        Question: {question}
        """
        troubleshoot_prompt = ChatPromptTemplate.from_template(template)

        # troubleshoot_llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.4)
        troubleshoot_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.4)


        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | troubleshoot_prompt
            | troubleshoot_llm
            | StrOutputParser()
        )

        return rag_chain

    #QA and BM25
    def initialize_product_lookup_chain(self):
        
        def metadata_func_dishwasher(record: dict, metadata: dict) -> dict:

            metadata["product_id"] = record.get("PartSelect_Number")
            metadata["manufacture_part_id"] = record.get("Manufacturer_Part_Number")
            metadata['device'] = 'Dishwasher'
            metadata['name'] = record.get("Product_Name")

            return metadata
        def metadata_func_refrigerator(record: dict, metadata: dict) -> dict:

            metadata["product_id"] = record.get("PartSelect_Number")
            metadata["manufacture_part_id"] = record.get("Manufacturer_Part_Number")
            metadata['device'] = 'Refrigerator'
            metadata['name'] = record.get("Product_Name")

            return metadata
        

        dishwasher_loader = JSONLoader(
            file_path='data/dishwasher_all_product_data.json',
            jq_schema='.[]',
            metadata_func=metadata_func_dishwasher,
            text_content=False)
        # dishwasher_data = dishwasher_loader.load()

        refrigerator_loader = JSONLoader(
            file_path='data/refrigerator_all_product_data.json',
            jq_schema='.[]',
            metadata_func=metadata_func_refrigerator,
            text_content=False)
        # refrigerator_data = refrigerator_loader.load()
        merge_loader = MergedDataLoader(loaders=[dishwasher_loader, refrigerator_loader])
        all_data = merge_loader.load()
        # print(len(all_data))
        bm25_retriever = BM25Retriever.from_documents(all_data)
        bm25_retriever.k =  5   
        product_lookup_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        product_lookup_template = """Answer the question based only on the following context:
                        {context}
                        Summarize
                        Question: {question}
                        """
        product_lookup_prompt = ChatPromptTemplate.from_template(product_lookup_template)
        product_lookup_rag_chain = (
            {"context": bm25_retriever, "question": RunnablePassthrough()}
            | product_lookup_prompt
            | product_lookup_llm
            | StrOutputParser()
        )

        return product_lookup_rag_chain
        
    #Parsing/WebSearch
    def initialaize_comptability_chain(self):
        self.comptability_memory = ''
        compatibility_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        compatibility_template = """You are an expert an extracting product_id (usually start with 'PS' but not always) and model_number to check compatability between them
            BOTH CANNOT BE SAME
            Given a {query} ONLY extract 'product_id' and 'model_number' in  JSON format. 
            
            ONLY IF BOTH EXIST RETURN JSON AND SAY NOTHING

            If ONLY ONE value is present kindly ask user for the missing value, DO NOT ASSUME

            You can make use of {memory} for past conversation if exists. DO NOT ASK USER FOR CONTEXT

            """
        compatibility_prompt = ChatPromptTemplate.from_template(compatibility_template)
        compatibility_chain =  compatibility_prompt | compatibility_llm | StrOutputParser()
        return compatibility_chain

    #Default Chat
    def initialaize_default_chain(self):
        default_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
        default_template = """ You are a kind and chatbot for e-commerce website called Part Select that sells electronic parts.

        YOU CAN ONLY HELP REGARDING DISHWASHERS AND REFRIGERATORS. THE MAIN TASKS YOU CAN PERFORM ARE 
        1. HELP TROUBLESHOOT AND REPAIR ISSUES IN REFRIGERATORS AND DISHWASHERS 
        2. CHECK IF A PARTICULAR PART IS COMPATIBLE WITH DEVICE MODEL
        3. GIVE GENERAL INFORMATION AND GUIDE INSTALLATION ABOUT A PRODUCT

        IF USER ASKS YOU ANYTHING KINDLY INFORM THEM ABOUT WHAT YOU CAN DO OR GREET THEM BACK.

        USER_QUERY: {query}
        """
        default_prompt = ChatPromptTemplate.from_template(default_template)
        default_chain = default_prompt | default_llm | StrOutputParser()
        return default_chain

    #RUN
    def run(self, user_query):
        router_response = self.data_router.invoke({"question": user_query, "system":self.router_system, "memory":self.master_memory})
        if self.master_memory == '':
             self.master_memory = self.master_memory + 'Past Conversation: ' 
        self.master_memory = self.master_memory + user_query
        print('ROUTER: ', router_response.datasource.lower())

        if "product_database_lookup" in router_response.datasource.lower():
                #Empty Other Memories
                
                self.comptability_memory = ''
            
                ###########################
                
                #Invoke something 
                
                product_database_response = self.product_lookup_chain.invoke(user_query.replace('?',''))

                self.master_memory = self.master_memory + ' --> Routing to product_database_lookup ' + product_database_response 
                return product_database_response
        
        elif "troubleshooting_guidelines" in router_response.datasource.lower():
            #Empty Other Memories
            
            self.comptability_memory = ''
            self.master_memory = ''
            ###########################

            trouble_shoot_response = self.troubleshoot_chain.invoke(user_query)


            
            return trouble_shoot_response
        
        elif "compatability_questions" in router_response.datasource.lower():
            
            self.master_memory = self.master_memory + ' --> Routing to compatability_questions ' 
            dummy_size = int(len(self.master_memory)/2)
            # +'and this to some extent' +self.master_memory
            compatability_response = self.comptability_chain.invoke({"memory": self.comptability_memory + self.master_memory, "query":user_query })
            if self.comptability_memory == '':
                self.comptability_memory = self.comptability_memory + 'Past Conversation: ' + user_query + ' --> '
            else:
                self.comptability_memory = self.comptability_memory + ' --> ' + user_query
            
            
            if 'json' in compatability_response and '"product_id":' in compatability_response and '"model_number":' in compatability_response:
                print(compatability_response)
                start, end = compatability_response.find('{'), compatability_response.find('}') + 1
                json_substring = compatability_response[start:end]
                compatability_json = json.loads(json_substring)
            else:
                self.comptability_memory = self.comptability_memory + ' --> ' + compatability_response
                self.master_memory = self.master_memory + ' --> ' + compatability_response
                return compatability_response
            

            #### Now Scraping website #####
            model_number = compatability_json['model_number']
            product_id = compatability_json['product_id']
            base_url = 'https://www.partselect.com/Models/'

            model_url = base_url + str(model_number) + '/#Parts'
            part_url = base_url + str(model_number) + '/Parts/?SearchTerm=' + str(product_id)

            response = requests.get(part_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            if soup.find('div', class_ = 'mega-m__part') == None:
                output = 'Parts are not compatable. You can check compatability at '+model_url
            else:
                output = 'Parts are compatable. You can see the part here: '+part_url

            self.comptability_memory = self.comptability_memory + ' --> ' + output


            self.master_memory = self.master_memory + ' --> ' + output


            return output

        else:   #general - default
            default_response = self.default_chain.invoke(user_query)
            return default_response






@app.route('/api/get_ai_message', methods=['POST'])
def get_ai_message():

    data = request.get_json()
    user_query = data.get('userQuery')
    print(user_query)


    response_content = chatbot.run(user_query)
    
    

    # Process the user query here and generate a response
    response = {
        "role": "assistant",
        "content": response_content
        # "content": "This is a response from the Python backend. hahhahahaha"
    }
    
    return jsonify(response)

if __name__ == '__main__':

    #initilaize API KEYS
    load_dotenv(find_dotenv())


    # Initialize models
    print('Initializing RAG Pipeline and Databases')
    chatbot = RAG_Pipeline()
    print('Chatbot Ready!!')

    app.run(debug=True)
