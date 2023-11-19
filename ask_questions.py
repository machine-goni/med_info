# pip install tiktoken
# pinecone 을 pip install 시 pinecone-client 으로 인스톨하고 requirement 에도 pinecone-client 로 넣어줘라

import os
import json
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import base64
#import gpt_tokenizer



PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'us-west1-gcp-free') # You may need to switch with your env
INDEX_NAME = "med-info-with-msd-asan-230815" # put in the name of your pinecone index here
sk = 'rkskekfkakqktkdmgpdmgpdmgpgp2955'



class AskQuestions:
    
    def __init__(self, openai_key, pinecone_key):
        self.openai_key = decrypt(sk, openai_key)
        self.pinecone_key = decrypt(sk, pinecone_key)                
        self.source_docs = []    
        self.gpts_choice_list = []        
        self.search_distance = 0.9 
        self.memory_slot = 1
        #self.chat_history = []
        #self.conversation_count = 0

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
        #self.embeddings_filter = EmbeddingsFilter(embeddings=self.embeddings, similarity_threshold=0.78) # or 0.76

        # We now initialize the ConversationalRetrievalChain
        # max_tokens parameter 는 completion 의 길이인 듯 하다. 짧게 설정했더니 답이 제한만큼만 짤려서 날아온다.
        # temperature 0 이 정확도는 물론이고 속도면에서 가장 빠르다고 한다.
        # 오픈AI 블로그(openai.com/blog) 에 따르면 23.12.11 부로 gpt-3.5-turbo-1106 이 gpt-3.5-turbo 로 자동업그레이드되고 
        # gpt-3.5-turbo-0613 및 gpt-3.5-turbo-16k-0613 는 명시적으로 모델명을 적어야하며, 24.06.13 까지만 사용가능하다고 한다.
        # 23.11.17 일 현재 gpt-3.5-turbo-1106 가 개똥같은 성능으로 물어보면 다 모른다고 하기때문에 0613 버전을 당분간 계속 사용하기 위해 명시해준다.
        # 24.06.13 전에 1106 버전을 사용할지, 아니면 해당기능을 뺄지 결정해서 수정해야 한다.
        # self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, openai_api_key=self.openai_key)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0.0, openai_api_key=self.openai_key)

        # 리스트로 구현한 명시적 chat_history 를 사용하지 않고(명시적으로 쓸땐 이부분이 필요없다) memory 를 사용.
        # ConversationBufferWindowMemory 는 저장할 chat_history 의 갯수를 지정할 수 있다.
        # key 는 이대로 써주지 않으면 에러난다.
        # 현재 코드로는 대화를 이어서 하지 않기 때문에 memory 가 필요없지만 ConversationalRetrievalChain 에서 설정해주지 않으면
        # 에러가 나기 때문에 설정은 해준다. 하지만 ask_first 에서 매번 초기화하기 때문에 동작은 하지 않는다.
        self.memory = ConversationBufferWindowMemory(k=self.memory_slot, input_key='question', output_key='answer', memory_key='chat_history', return_messages=True)


    # 대화를 처음 시작
    def ask_first(self, query):
        self.gpts_choice_list = []        
        #self.conversation_count = 0        
        
        try:             
            # embeddings_filter 가 score 로 관계없는 chunk 를 걸러내긴 하는데 엉뚱한 chunk 만 검색되었을때는 먹통이되기도한다.
            # embeddings_filter 가 먹통이되면 k개만큼의 chunk 를 gpt 한테 전부 보내서 max token 에러가 날 수도 있다. 조심해야한다.
            # 이 프로젝트에서는 search_type="mmr" 보다 "similarity" 가 더 적합한데, 
            # "similarity_score_threshold" 로 걸러주는 것이 훨씬 좋은것 같다
            nearest_k = 4
            score_threshold = 0.1
            #base_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":nearest_k})            
            base_retriever = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":nearest_k, "score_threshold": score_threshold})            
            # Contextual Compression Retriever
            #self.retriever = ContextualCompressionRetriever(base_compressor=self.embeddings_filter, base_retriever=base_retriever)
            # retriever 의 search type 을 similarity_score_threshold 해서 이미 거르고 있기때문에 embedding filter 는 뺀다.
            self.retriever = base_retriever

            # 여기서 다시 객체를 만드는 이유는 memory 에 내용이 남아 있으면 아래서 만드는 prompt template 이 무시되어 버린다.
            # 대화의 시작은 prompt template 가 먹어야 하고 이전의 대화내용은 필요없기 때문에 여기서 리셋해줘야 한다.
            # 그럼에도 불구하고 init 함수에서도 생성해 놓은 이유는 혹시 ask_first 로 시작하지 않고 ask 로 시작하게 되면 에러가 나기 때문에 일단 생성은 해 놓는다. 
            self.memory = ConversationBufferWindowMemory(k=self.memory_slot, input_key='question', output_key='answer', memory_key='chat_history', return_messages=True)
        
            # Initialize chat history list
            #self.chat_history = []

            # 원본
            #new_query = f"""JSON 형식으로 세 개의 역따옴표로 구분된 텍스트에 대해 답변해라. reply 키에 답변을, expected 키에 3개의 관련 예상 질문을 넣어라. ```{query}```"""                                    
            # prompt template 사용
            # 이러한 형식은 첫 질문에만 사용한다. 질문을 이어서 할때는 형식지정이 무시되어서..
            '''
            template = """[{query}]에 대해 답변 해라. JSON 형식으로 응답하고 reply 키에 []안 질문에 답변을, expected 키에 {question_n}개의 예상 질문을 리스트로 넣어라."""            
            prompt = PromptTemplate(
                input_variables=["query", "question_n"],
                template=template,
            )
            new_query = prompt.format(query=query, question_n=3)  
            '''
            # 원래는 위처럼 prompt 를 만들어서 self.chain({"question": new_query, "vectordbkwargs": vectordbkwargs}) 으로 썻다.
            # 하지만 이렇게 쓰면 retriever 가 검색을 할때 질문외 나머지 지정사항까지 모두 query 에 포함하여 검색해서 엉뚱한 검색결과가 나온다.
            # 이를 해결하기 위해 아래 처럼 PromptTemplate 를 만들고 chain 생성시 combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT} 처럼
            # 옵션을 넣어준다. 그러면 프롬프트 전체가 아니라 question 에 대한 문장만 검색하여 chunk 를 찾는다.
            # 참고로 아래 template 에서 {context} 가 없으면 "컨텍스트를 사용하여 질문에 답하라" 를 이해하지 못하고,
            # Question: {question} 의 {question} 은 변수명이 아니라 self.chain({"question" 의 question 이라는 키이름 이다.
            template = """다음 컨텍스트를 사용하여 질문에 답하라. 답을 모르면 모른다고 답하고 지어내지 마라. reply 키와 expected 키를 가진 JSON 으로 응답하고 reply 에 답변, expected 에 3개의 예상 질문을 list 로 넣어라. 최대 2개의 문장을 사용하고 답변은 가능한 간결하게 해라."
            {context}
            Question: {question}
            ["reply":, "expected":[]]"""
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template) 

            # create chain
            # chat_history 에 memory 를 사용.                            
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,        
                retriever=self.retriever,
                chain_type="stuff",
                return_source_documents=True,    # 검색한 소스문서가 무엇인지 반환
                memory = self.memory,
                combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}               
            )    
        
            # Begin conversing
            # 명시적 chat_history 를 사용할 때
            #response = self.chain({"question": query, "chat_history": self.chat_history})            
            # chat_history 는 memory 를 쓰고, search_distance filter 를 사용할 때
            '''
            search_distance. vector store 가 검색 거리별 필터링을 지원한다면 쓸 수 있다.
            pinecone 은 되는 것 같다. chroma 는 테스트 안 해봐서 모른다.
            '''
            vectordbkwargs = {"search_distance": self.search_distance}                    
            response = self.chain({"question": query, "vectordbkwargs": vectordbkwargs})
    
            #print("\nQ: " + query)
            #print("\nA: " + response["answer"])
            # 어떤 chunk 에서 검색된 것인지 확인
            #print(response["source_documents"][:])
            # Add the answer to the chat history        
            #self.chat_history.append((query, response["answer"]))
            #self.conversation_count = 1

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
                    #print(f"source:{documentWithState.metadata['source']}")
                    #print(f"section ID:{documentWithState.metadata['section_id']}")
            
            # type 을 확인해보면 'answer_1 type is <class 'str'>' 라고 찍힌다. 즉 json 이라는 의미이기 때문에 그대로 보낸다.            
            # 바로 응답을 리턴하지 않고 다시 검색 한다
            #return response['answer']

            # 분명 json 으로 답변하라고 했는데 갑자기 json 같은 쌩 스트링을 내놓을 때가 있다. 그럴땐 수동으로 처리해줘야 한다.
            reply_str = None
            expected_list = None
            if type(response['answer']) is str:
                #print("answer is str")                
                answer_str = response['answer']
                answer_str_2 = answer_str.replace('\n', '')
                answer_str = answer_str_2.replace('"', '')
                answer_str_2 = answer_str.replace("'", '')
                answer_str = answer_str_2.replace("{", '')
                answer_str_2 = answer_str.replace("}", '')
                answer_str = answer_str_2.replace("[", '')
                answer_str_2 = answer_str.replace("]", '')
                splitted = answer_str_2.split('expected: ')                
                reply_str = splitted[0].replace('reply: ', '')                                                                             
                expected_list = splitted[1].split(', ') 

            else:
                #print("answer is json")
                answer_dict = json.loads(response['answer'])
                reply_str = answer_dict['reply']
                expected_list = answer_dict['expected']
                
            #print(f"reply: {reply_str}")
            #print(f"expected: {expected_list}")

            # 원래는 최종 1개만 뽑았는데 n개를 뽑는 코드로 변경                       
            #self.gpts_choice_doc, self.gpts_choice_id, f_answer = self.search_doc_from_gpt_answer(reply_str)
            self.gpts_choice_list, gpts_choice_ids, gpt_answer_str = self.search_doc_from_gpt_answer(reply_str)

            new_choice_url_list = []
            if self.gpts_choice_list is not None:
                # url 에 콤마가 들어있는게 있다. 안드로이드에서 콤마 기준으로 분할할거라서 바꿔줘야 한다.            
                for individual in self.gpts_choice_list:
                    new_choice_url_list.append(individual.replace(',', '||'))

            json_data = {}
            json_data["reply"] = gpt_answer_str
            json_data["expected"] = expected_list
            json_data["related_search_url"] = new_choice_url_list
            json_data["related_search_id"] = gpts_choice_ids            

            return json.dumps(json_data)
            

        except Exception as e:
            print("Exception!!:" + str(e))            
            
            json_data = {}
            json_data["reply"] = "Exception!!"
            json_data["expected"] = [str(e), "", ""]            

            # json 형식으로 보내기 위해 json.dumps 를 사용해 dictionary 를 json 으로 변환
            return json.dumps(json_data)                
    
    
    """
    # ask() 메서드는 사용하지 않기로 했다. 대화를 이여서 하는게 정보검색에 별로 의미가 없고
    # 토큰수에 부담을 줘서 오히려 답변 퀄리티를 낮추기 때문이다.
    # 따라서 이 메서드는 ask_first 와 같이 수정하지 않은 상태이다. 만약 사용하려면 수정해야 한다.
    # 앞에 대화에 이어서 질문      
    def ask(self, query):    

        try:
            nearest_k = 3

            '''
            # meatadata 로 filtering 한다. vector store 종류에 따라 인터페이스가 다를 수 있다.
            아래 filter 는 pinecone 에서 support 하는 형식이다. Chroma 는 다를 수 있다.
            중요한 점은 이 필터를 아래 chain 에서도 넣을 수 있지만 테스트결과 chain 에서 넣었을 때는 제대로 필터링이 되지 않는다.
            그래서 필터에 조건이 그때 그때 달라진다면 retriever 생성 관련 부분을 상황에 따라 매번 다시 해주는 수 밖에 없다. 함수로 빼자.
            #filter = {'source': {"$eq": 'documents/Sciatica.txt'}}     # 하나만 넣을때는 이렇게도 쓸 수 있다.
            #filter = {'source': {"$in": ["documents/Sciatica.txt", "documents/Rheumatoid_arthritis.txt"]}}  # 복수일 때
            '''
            '''
            if (len(self.source_docs) > 0):                
                filter = {'source': {"$in": self.source_docs}}
                base_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3, "filter":filter})                
                print(f"source_doc: {self.source_docs}")   
            else:
                base_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
            '''
            if self.gpts_choice_doc is not None:                
                filter = {'source': {"$in": [self.gpts_choice_doc]}}
                base_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":nearest_k, "filter":filter})                             
                #print(f"gpts_choice_doc: {self.gpts_choice_doc}")                
            else:
                base_retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":nearest_k})
            
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
                         
            # chat_history 를 명시적으로 쓸 때
            #response = self.chain({"question": query, "chat_history": self.chat_history})
            '''
            # meatadata 로 filtering. 여기서 이렇게 넣어도 에러는 안나고 돌아는 가지만 제대로 필터링하지 못한다.
            # 상황설명을 위해 코드와 주석을 남겨둔다.
            filter = {'source': {"$in": ["documents/Sciatica.txt", "documents/Rheumatoid_arthritis.txt"]}}           
            vectordbkwargs = {"search_distance": 0.9, "filter": filter}            
            '''
            vectordbkwargs = {"search_distance": self.search_distance}        
            response = self.chain({"question": query, "vectordbkwargs": vectordbkwargs})       

            #print("\nQ: " + query)
            #print("\nA: " + response["answer"])
            # 어떤 chunk 에서 검색된 것인지 확인
            #print(response["source_documents"][:])
            # Add the answer to the chat history            
            #self.chat_history.append((query, response["answer"]))
            #self.conversation_count = self.conversation_count + 1

            #answer = json.loads(response["answer"])                
            #json_data["answer"] = answer["answer"]            
            #json_data["expected_1"] = answer["expected"][0]            
            #json_data["expected_2"] = answer["expected"][1]
        
            #return response['answer']
            #print(f"reply: {response['answer']}")
            self.gpts_choice_doc, self.gpts_choice_id, gpt_answer_str = self.search_doc_from_gpt_answer(response['answer'], False)
            return gpt_answer_str
            
            
        except Exception as e:
            answer = 'Exception!!:' + str(e)
            print(answer)
                
            return answer
    """


    def search_doc_from_gpt_answer(self, gpt_answer, is_first=True):
        ## 이 부분은 gpt 로 부터 요약된 응답으로 vector db 에서 다시 관련 부분을 찾아 페이지와 페이지 내 위치를 얻는 코드이다.
        # 이 코드를 넣은 이유는 앞에서 사용한 retriever 가 n개의 chunk 를 뽑아내기 때문에 gpt 가 어떤것을 참조한지 모르기 때문이다.
    
        # response answer 를 다시 vector db 에서 검색한다.                
        # 검색된 문서가 있다면 그 안에서만 검색해서 시간을 줄이자        

        '''
        # vector db 의 search_type 을 similarity_score_threshold 로 설정해서 사용한다.
        이유는 문서에서 답을 찾지 못 했을 때에도 GPT는 답을 하기때문에 이때 다르게 처리하기위해서 이다.
        여기서 주의할 점은 Vector DB 마다 score_threshold 가 다르다.(또한 Metric 방식에 따라서도 다르다)
        예를 들어 같은 문서에서 같은 내용을 검색한다고 했을 때 

        print(str(score) + " - " + page_num)

        # faiss 경우
        0.31319273 - 58
        0.34054485 - 61
        0.39556867 - 36
        0.40675405 - 60
        0.40916008 - 33

        # Pinecone 경우
        0.84340357 - 58.0
        0.82972753 - 61.0
        0.80216819 - 36.0
        0.79670471 - 60.0
        0.79520261 - 33.0

        # For Chroma, the scores are numbers above 1. In ascending order:
        1.21817
        1.225833
        1.227054

        이렇게 다르다.
        '''
        # 주의할 점은 score_threshold 를 너무 높게 잡으면 GPT 가 요약을 했을때 검색이 안된다.        
        """
        nearest_k = 1
        score_threshold = 0.1
        
        if (len(self.source_docs) > 0) and is_first == True:                
            filter = {'source': {"$in": self.source_docs}}
            retriever_2 = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":nearest_k, "filter":filter, "score_threshold": score_threshold})    
            #retriever_2 = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":nearest_k, "filter":filter})    
        elif (self.gpts_choice_doc is not None) and is_first == False:
            filter = {'source': {"$in": [self.gpts_choice_doc]}}                                    
            retriever_2 = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":nearest_k, "filter":filter, "score_threshold": score_threshold})
            #retriever_2 = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":nearest_k, "filter":filter})
        else:
            retriever_2 = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":nearest_k, "score_threshold": score_threshold})
            #retriever_2 = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":nearest_k})

        docs_2 = retriever_2.get_relevant_documents(gpt_answer)
        """    
        # 원래는 위의 코드였으나 프로세스를 약간 바꾼다. 처음 검색된 문서 위주로 검색을 하고, threshold 를 넘지 못하면 전체 문서에서 다시 검색을 한다.
        # 그리고 이어지는 대화 관련 코드는 뺀다.
        no_filter = False
        nearest_k = 3
        score_threshold = 0.25

        if (len(self.source_docs) > 0):                           
            filter = {'source': {"$in": self.source_docs}}
            retriever_2 = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":nearest_k, "filter":filter, "score_threshold": score_threshold})                
        else:
            retriever_2 = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":nearest_k, "score_threshold": score_threshold})
            no_filter = True

        #print(f"no_filter: {no_filter}")
        docs_2 = retriever_2.get_relevant_documents(gpt_answer)

        if (len(docs_2) < 1) and no_filter == False:
            # 문서 전체에서 재검색            
            retriever_2 = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":nearest_k, "score_threshold": score_threshold})
            docs_2 = retriever_2.get_relevant_documents(gpt_answer)
            #print("re-search")

        # 검색된 문서가 없다면 GPT 의 answer 만을 반환
        if (len(docs_2) < 1):            
            return None, [], gpt_answer
                
        #print(f"gpt_answer:{gpt_answer}")
        #print(f"score: {score_threshold}, All docs({len(docs_2)}):")  
        final_source_url_list = []
        final_source_id_list = []             
        for i, doc2 in enumerate(docs_2):            
            final_source_url_list.append(doc2.metadata['source'])
            final_source_id_list.append(doc2.metadata['section_id'])
            #print(f"{i}. final source: {doc2.metadata['source']}")
            #print(f"{i}. final id: {doc2.metadata['section_id']}")
            #print(f"{i}. final content:{doc2.page_content}")
        
        
        # page_content 는 확인을 위한것이지 user 에게 보여질 것이 아니다.    
        # user 는 해당 url 의 id 를 따라 링크만 전달되면 된다.
        # 하지만 문서를 찾지 못 할 수도 있다. 그럴땐 URL 이 아니라 그냥 answer 를 보여줘야 한다.
        #print(f"final content:{docs_2[0].page_content}")
        
        # link
        #if docs_2[0].metadata["section_id"] != 'None':
        #    print(docs_2[0].metadata["url"] + '#' + docs_2[0].metadata["section_id"])
        # 위의 link 는 확인을 위함이고 페이지를 띄울때 처음부터 저런식으로는 페이지의 해당 id 위치로 이동시킬 수 없다.
        # 이유는 페이지가 로드되면서 다시 제일위로 이동되기 때문에 페이지 로드가 완료되고 나면 id 의 위치로 이동시킬 수 있다.
        # 코틀린 웹뷰에서의 구현코드는 따로 정리된것이 있으니 해당 문서 참고.
        # 추가적으로 여기에서는 section id 만 사용하지만 태그에 id 가 있다면 어디등 id 의 위치로 이동시킬 수 있다.

        return final_source_url_list, final_source_id_list, gpt_answer
    


def decrypt(key, ciphertext):
    decoded_ciphertext = base64.b64decode(ciphertext)
    iv = decoded_ciphertext[:AES.block_size]
    cipher = AES.new(key.encode(), AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(decoded_ciphertext[AES.block_size:]), AES.block_size).decode('utf-8')
    
    return plaintext


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