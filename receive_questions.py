# backend 에서 돌아갈 파이썬 스크립트

from ask_questions import AskQuestions
import json


class RecvQuestions:
    def set_keys(self, openai_key, pinecone_key):
        self.ask_q = AskQuestions(openai_key, pinecone_key)

        return True


    def recv_question(self, question, isFirst):
        if isFirst == True:
            result = self.ask_q.ask_first(question)
        
        else:
            result = self.ask_q.ask(question)
        
        #print(type(result))
        #print(result)
        return result        