# pip install tiktoken
# pinecone 을 pip install 시 pinecone-client 으로 인스톨하고 requirement 에도 pinecone-client 로 넣어줘라

import os
import json
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
#import gpt_tokenizer



PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-west1-gcp-free') # You may need to switch with your env
INDEX_NAME = "med-info-index" # put in the name of your pinecone index here



class AskQuestions:
    
    def __init__(self, openai_key, pinecone_key):
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.conversation_count = 0
        self.source_docs = []
        #self.chat_history = []

        # create embedding instance
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_key)

        # initialize pinecone
        pinecone.init(api_key=self.pinecone_key, environment=PINECONE_API_ENV)

        # 이미 문서를 임베딩해서 pinecone vector store 에 넣었다면 거기에서 끌어다 쓰면 된다
        self.vectorstore = Pinecone.from_existing_index(INDEX_NAME, self.embeddings)

        '''
        # Contextual Compression Retriever 의 EmbeddingsFilter
        1000 정도의 chunk 사이즈에서는 별로 걸러낼게 없는것인지 프롬프트로 보내는 토큰수가 그대로 이다. 
        하지만 chunk 사이즈가 커지거나 k 가 커지면 효과를 볼 것 같다.
        similarity_threshold 가 0.8 부터는 검색되는게 없다. 또 0.76 은 토큰수가 줄지않고 거의 그대로라 0.78 로 한다.
        현재 설정에서(chunk size=1000, k=3) 테스트해 본 결과 청크안의 내용을 줄이는게 아니라 임계치에 의해 점수가 낮은 chunk 를 빼버린다.
        retriever 가 뽑아낸 chunk 는 3개 인데 filtering 을 통해 2개의 chunk 로 줄어드는 것을 확인할 수 있었다.
        하지만 query 가 매우 짧은 경우 점수가 낮아 아예 검색이 안 될 수도 있다.
        '''
        self.embeddings_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.78) # or 0.76

        # We now initialize the ConversationalRetrievalChain
        # max_tokens parameter 는 completion 의 길이인 듯 하다. 짧게 설정했더니 답이 제한만큼만 짤려서 날아온다.
        # temperature 0 이 정확도는 물론이고 속도면에서 가장 빠르다고 한다.
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, openai_api_key=self.openai_key)

        # 리스트로 구현한 명시적 chat_history 를 사용하지 않고(명시적으로 쓸땐 이부분이 필요없다) memory 를 사용.
        # ConversationBufferWindowMemory 는 저장할 chat_history 의 갯수를 지정할 수 있다.
        # key 는 이대로 써주지 않으면 에러난다.
        self.memory = ConversationBufferWindowMemory(k=2, input_key='question', output_key='answer', memory_key='chat_history', return_messages=True)


    # 대화를 처음 시작
    def ask_first(self, query):
        self.conversation_count = 0        
        
        try: 
            # k는 2~3 적당한것 같다. 이 프로젝트에서는 search_type 이 mmr 보다 similarity 가 더 적합하다.
            base_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
            # test. search_type = "mmr" 
            #self.retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":3})
            # test. search_type="similarity_score_threshold"
            #self.retriever = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":3, "score_threshold":0.5})
            # Contextual Compression Retriever
            self.retriever = ContextualCompressionRetriever(base_compressor=self.embeddings_filter, base_retriever=base_retriever)

            # create chain
            '''
            # 기본형
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm, 
                retriever=self.retriever,
                chain_type="stuff",
                return_source_documents=True    # 검색한 소스문서가 무엇인지 반환
            )
            '''
            # chat_history 에 memory 를 사용.    
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,        
                retriever=self.retriever,
                chain_type="stuff",
                return_source_documents=True,    # 검색한 소스문서가 무엇인지 반환
                memory = self.memory
            )
            
            # Initialize chat history list
            #self.chat_history = []

            # 원본
            #new_query = f"""JSON 형식으로 세 개의 역따옴표로 구분된 텍스트에 대해 답변해라. reply 키에 답변을, expected 키에 2개의 관련 예상 질문을 넣어라. ```{query}```"""                        
            
            # prompt template 사용
            # 이러한 형식은 첫 질문에만 사용한다. 질문을 이어서 할때는 형식지정이 무시되는 버그가 있어서..
            template = """[{query}]에 대해 답변 해라. JSON 형식으로 응답하고 reply 키에 []안 질문에 답변을, expected 키에 {question_n}개의 예상 질문을 리스트로 넣어라."""
            prompt = PromptTemplate(
                input_variables=["query", "question_n"],
                template=template,
            )
            new_query = prompt.format(query=query, question_n=3)
        
            # Begin conversing
            # 명시적 chat_history 를 사용할 때
            #response = self.chain({"question": new_query, "chat_history": self.chat_history})
            
            # chat_history 는 memory 를 쓰고, search_distance filter 를 사용할 때
            '''
            search_distance. vector store 가 검색 거리별 필터링을 지원한다면 쓸 수 있다.
            pinecone 은 되는 것 같다. chroma 는 테스트 안 해봐서 모른다.
            '''
            vectordbkwargs = {"search_distance": 0.9}        
            response = self.chain({"question": new_query, "vectordbkwargs": vectordbkwargs}) 
    
            #print("\nQ: " + query)
            #print("\nA: " + response["answer"])
            # 어떤 chunk 에서 검색된 것인지 확인
            #print(response["source_documents"][:])
            # Add the answer to the chat history        
            #self.chat_history.append((query, response["answer"]))
            self.conversation_count = 1

            # gpt 가 보낸(response) 형식이 이미 json 이기 때문에 key 로 접근을 하려면 json.loads 를 통해 dictionary 로 변환해야 한다.
            # 다시 말해, json(str)->dict 는 json.loads, dict->json(str) 는 json.dumps
            # encoding 이 되있으면 dumps 를 써야 하는것 같던데.. 확실하진 않다. (answer = json.loads(json.dumps(response["answer"])) 이런식으로)  
            # 이미 type 이 str 인 상태에서 또 다시 json.dumps 를 뜨면 encoding 이 되어버린다. 그래서 하면 안됨.          
            #answer = json.loads(response["answer"])                
            #json_data["answer"] = answer["answer"]            
            #json_data["expected_1"] = answer["expected"][0]            
            #json_data["expected_2"] = answer["expected"][1]

            # response 는 dictionary 이다. gpt 에게 json 형식으로 달라고 했지만 response 에는 이거저거 다 들어있기 때문.
            # print(response)

            # 첫 대화에서 검색된 소스문서로 이어지는 대화의 검색범위를 한정한다.
            self.source_docs = []
            source_documents = response['source_documents']
            if (len(source_documents) > 0):
                for documentWithState in source_documents:
                    self.source_docs.append(documentWithState.metadata["source"])
                    #print(documentWithState.metadata["source"])

            #return response
            # type 을 확인해보면 'answer_1 type is <class 'str'>' 라고 찍힌다. 즉 json 이라는 의미이기 때문에 그대로 보낸다.
            #print('answer_1 type is ' + str(type(response['answer'])))
            return response['answer']
            

        except Exception as e:
            print("Exception!!:" + str(e))            
            
            json_data = {}
            json_data["reply"] = "Exception!!"
            json_data["expected"] = [str(e), "", ""]            

            # json 형식으로 보내기 위해 json.dumps 를 사용해 dictionary 를 json 으로 변환
            return json.dumps(json_data)                
    
    
    # 앞에 대화에 이어서 질문      
    def ask(self, query):    


        try:
            '''
            # meatadata 로 filtering 한다. vector store 종류에 따라 인터페이스가 다를 수 있다.
            아래 filter 는 pinecone 에서 support 하는 형식이다. Chroma 는 다를 수 있다.
            중요한 점은 이 필터를 아래 chain 에서도 넣을 수 있지만 테스트결과 chain 에서 넣었을 때는 제대로 필터링이 되지 않는다.
            그래서 필터에 조건이 그때 그때 달라진다면 retriever 생성 관련 부분을 상황에 따라 매번 다시 해주는 수 밖에 없다. 함수로 빼자.
            #filter = {'source': {"$eq": 'documents/Sciatica.txt'}}     # 하나만 넣을때는 이렇게도 쓸 수 있다.
            #filter = {'source': {"$in": ["documents/Sciatica.txt", "documents/Rheumatoid_arthritis.txt"]}}  # 복수일 때
            '''
            if (len(self.source_docs) > 0):                
                filter = {'source': {"$in": self.source_docs}}
                base_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3, "filter":filter})
                #print('source_doc:')
                #print(source_doc)
            else:
                base_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
            
            # Contextual Compression Retriever
            self.retriever = ContextualCompressionRetriever(base_compressor=self.embeddings_filter, base_retriever=base_retriever)

            # create chain.    
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,        
                retriever=self.retriever,
                chain_type="stuff",
                return_source_documents=True,    # 검색한 소스문서가 무엇인지 반환
                memory = self.memory             # chat_history 에 memory 를 사용
            )

            '''
            # chat_history 를 명시적으로 쓸 때 저장 갯수 제한. chat history 를 최근 2개까지만 저장한다. 1개도 괜찮긴한데 안괜찮을때가 있어서..
            if len(self.chat_history) > 2:
                chat_0 = self.chat_history[1]
                chat_1 = self.chat_history[2]
                self.chat_history = []
                self.chat_history.append(chat_0)
                self.chat_history.append(chat_1)            
            '''

            # 여기에서 PromptTemplate 을 사용하지 않는 이유는 대화를 이어서 할때는 요청한 형식을 무시해버린다.
            # OpenAI API 만을 사용할 때는 그렇지 않은 것을 확인한 것으로 볼때 LangChain 의 버그 인 듯.
            # Json 형식을 고집하고 싶다면 OutputParser 를 사용하면 될 거 같긴하다. 참고: https://revf.tistory.com/280
            new_query = query
             
            # chat_history 를 명시적으로 쓸 때
            #response = self.chain({"question": new_query, "chat_history": self.chat_history})
            '''
            # meatadata 로 filtering. 여기서 이렇게 넣어도 에러는 안나고 돌아는 가지만 제대로 필터링하지 못한다.
            # 상황설명을 위해 코드와 주석을 남겨둔다.
            filter = {'source': {"$in": ["documents/Sciatica.txt", "documents/Rheumatoid_arthritis.txt"]}}           
            vectordbkwargs = {"search_distance": 0.9, "filter": filter}            
            '''
            vectordbkwargs = {"search_distance": 0.9}        
            response = self.chain({"question": new_query, "vectordbkwargs": vectordbkwargs})       

            #print("\nQ: " + query)
            #print("\nA: " + response["answer"])
            # 어떤 chunk 에서 검색된 것인지 확인
            #print(response["source_documents"][:])
            # Add the answer to the chat history            
            #self.chat_history.append((query, response["answer"]))
            self.conversation_count = self.conversation_count + 1

            #answer = json.loads(response["answer"])                
            #json_data["answer"] = answer["answer"]            
            #json_data["expected_1"] = answer["expected"][0]            
            #json_data["expected_2"] = answer["expected"][1]

            # print(response)
            #return response
            #print('answer_2 type is ' + str(type(response['answer'])))
            return response['answer']
            
            
        except Exception as e:
            answer = "Exception!!:" + str(e)
            print(answer)
                
            return answer


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
