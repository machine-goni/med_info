'''
fastapi 는 pip install 로 설치해줘야 하고
서버를 돌리기 위해 pip install "uvicorn[standard]" 도 실행
'''

# backend 를 FastAPI 로 구현

from fastapi import FastAPI
from pydantic import BaseModel
from receive_questions import RecvQuestions

'''
POST 메세지를 받을 클래스. FastAPI 에서는 이걸 model 이라고 부른다.
operation, x, y 는 메세지로 받을 param 이다.
'''
class User_key(BaseModel):
    openai_key : str
    pinecone_key : str


class User_input(BaseModel):
    question : str
    isFirst : bool


# FastAPI instance 를 만들고
app = FastAPI()

receiver = RecvQuestions()

'''
FastAPI instance 로 REST API 를 정의 한다.
@app.post("/question_to_medinfo") 안의 "/question_to_medinfo" 는 route
'''

@app.post("/init")
def operate(input:User_key):
    result = receiver.set_keys(input.openai_key, input.pinecone_key)
    return result


@app.post("/ask")
def operate(input:User_input):
    result = receiver.recv_question(input.question, input.isFirst)
    return result


# For running the FastAPI server we need to run the following command:
# uvicorn fast_api:app --reload
# fast_api 는 실행할 FastAPI 가 구현되어있는 python script
# 커맨드를 실행하면 접속할 수 있는 local url 이 나온다
# http://127.0.0.1:8000/docs 를 열면 Swagger UI 를 볼 수 있다.