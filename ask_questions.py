# pip install tiktoken
# pinecone 을 pip install 시 pinecone-client 으로 인스톨하고 requirement 에도 pinecone-client 로 넣어줘라

import os
import json
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
#import gpt_tokenizer


PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-west1-gcp-free') # You may need to switch with your env
INDEX_NAME = "med-info" # put in the name of your pinecone index here

add_string = "JSON 형식으로 알려주고 답변에 대한 키는 ```answer``` 로 하고 관련 예상질문을 2개 뽑아서 리스트를 ```expected``` 키에 넣어줘"

class AskQuestions:
    #openai_key = ""
    #pinecone_key = ""
    #embeddings

    def __init__(self, openai_key, pinecone_key):
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.conversation_count = 0

        # create embedding instance
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_key)

        # initialize pinecone
        pinecone.init(api_key=self.pinecone_key, environment=PINECONE_API_ENV)

        # 이미 문서를 임베딩해서 pinecone vector store 에 넣었다면 거기에서 끌어다 쓰면 된다
        self.vectorstore = Pinecone.from_existing_index(INDEX_NAME, self.embeddings)

        # k는 2가 적당한것 같다
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})

        # We now initialize the ConversationalRetrievalChain
        # max_tokens parameter 는 completion 의 길이인 듯 하다. 짧게 설정했더니 답이 제한만큼만 짤려서 날아온다.
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, openai_api_key=self.openai_key)
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm, 
            retriever=self.retriever,
            chain_type="stuff",
            return_source_documents=True    # 검색한 소스문서가 무엇인지 반환
        )


    # 대화를 처음 시작
    def ask_first(self, query):
        self.conversation_count = 0
        json_data = {}

        try:        
            # Initialize chat history list
            self.chat_history = []

            new_query = query + " " + add_string

            # Begin conversing
            response = self.chain({"question": new_query, "chat_history": self.chat_history})
            #print("\nQ: " + query)
            #print("\nA: " + response["answer"])
            # 어떤 chunk 에서 검색된 것인지 확인
            #print(response["source_documents"][:])
            # Add the answer to the chat history
            self.chat_history.append((query, response["answer"]))
            self.conversation_count = 1
                        
            # json.loads 를 써야 type 이 json 이 된다. encoding 이 되있으면 dumps 를 써야 하는것 같던데.. 확실하진 않다. 둘 중 하나 쓰면 된다.
            # answer = json.loads(json.dumps(response["answer"]))
            answer = json.loads(response["answer"])                
            json_data["answer"] = answer["answer"]            
            json_data["expected_1"] = answer["expected"][0]            
            json_data["expected_2"] = answer["expected"][1]

            
        except Exception as e:
            print("Exception!!:" + str(e))            
            
            json_data["answer"] = "Exception!!"
            json_data["expected_1"] = str(e)
            json_data["expected_2"] = str(e)

        #return result
        return json_data
    
    
    # 앞에 대화에 이어서 질문
    def ask(self, query):
        json_data = {}

        try:
            # chat history 를 최근 2개까지만 저장한다. 1개도 괜찮긴한데 안괜찮을때가 있어서..
            if len(self.chat_history) > 2:
                chat_0 = self.chat_history[1]
                chat_1 = self.chat_history[2]
                self.chat_history = []
                self.chat_history.append(chat_0)
                self.chat_history.append(chat_1)

            new_query = query + " " + add_string
             
            response = self.chain({"question": new_query, "chat_history": self.chat_history})
            #print("\nQ: " + query)
            #print("\nA: " + response["answer"])
            # 어떤 chunk 에서 검색된 것인지 확인
            #print(response["source_documents"][:])
            # Add the answer to the chat history
            self.chat_history.append((query, response["answer"]))
            self.conversation_count = self.conversation_count + 1

            answer = json.loads(response["answer"])                
            json_data["answer"] = answer["answer"]            
            json_data["expected_1"] = answer["expected"][0]            
            json_data["expected_2"] = answer["expected"][1]
            
            
        except Exception as e:
            print("Exception!!:" + str(e))

            json_data["answer"] = "Exception!!"
            json_data["expected_1"] = str(e)
            json_data["expected_2"] = str(e)


        return json_data


"""
# API 호출수
1회 질문의 경우: text-embedding-ada 은 1번, gpt-3.5-turbo 는 1번.
421 prompt + 69 completion = 490 tokens

2회 질문의 경우: text-embedding-ada 은 2번, gpt-3.5-turbo 는 3번.
1,631 prompt + 184 completion = 1,815 tokens

3회 질문의 경우: text-embedding-ada 은 3번, gpt-3.5-turbo 는 5번.
2,204 prompt + 384 completion = 2,588 tokens

* 2회 이상 질문시 앞의 대화 내용을 또 보내기 때문에 (횟수 * 2 + 1) 의 호출수가 발생하는 듯 한다.
그래서 대화를 이어서 질문시에는 차감되는 사용횟수를 +1 해야 할 것 같다(앱에서).

* chat_history 의 경우 중간 중간 리셋하지 않고 모든 대화를 저장해서 보낸다고 해도 호출수 공식은 변하지 않는다.
단, 그렇게 되면 그만큼 사용 tokens 수가 늘어난다.

* embedding 의 경우 본 코드에서 1번이 더 나오는데 그것은 저위에서 어떤 문서가 검색되었는지 확인할때 
embedding 을 해서이다. 확인 코드를 없애면 질문수 만큼 embedding 을 한다.

# 사용 token
embedding 은 계산대로, completion 은 대략 비슷하다. 
하지만 prompt 의 경우 계산보다 많이 사용한다. vectorstrore 에서 찾은 문자열과 쿼리 외에 뭔가 더 추가되는게 있는듯 하다.
거기다 Memory 기능으로 대화를 이어갈땐 앞의 대화내용을 보냄으로 대략 앞의 대화 token 많큼을 더 사용하는 듯 하다.
앱에서 사용에 대한 비용을 요구할때는 그냥 max token 을 기준으로 정하는 것이 좋을 것 같다.
"""
