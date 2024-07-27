import numpy as np 
import pandas as pd
import tensorflow as tf 
from konlpy.tag import Okt
from keybert import KeyBERT
from mylib.test_text_subject import extract_topics #test_text_subject.py에서 extraact_topics import 
from mylib.check_similarity import Similarity

#중요도 논의 방법 
'''
1. 설문조사 : 설문조사 돌려서 학생들 인식대로
2. 국가적/국제적 지표 
3. 그냥 뇌피셜로
'''
data = None

class Jaenan_moonja() : 

    def __init__(self,text):
        '''
        text : 문자내용을 인자로 받음
        subject 변수 생성 -> 이후 find_subject method를 통해 subject 변수 업데이트 필요 
        '''
        self.text = text 
        subject = None
        self.subject = subject
        importance_map = {} #주제 : 중요도(정수형으로 저장) 
        self.imp = importance_map
    
    def find_subject(self) : 
        '''
        자연어 처리를 통한 주제 찾기 method 
        return은 없습니다. 
        
        ### 고민 ###
        self.subject를 바로 업데이트 가능한데 subject가 필요할까?
        -> 일단은 안전하게 변수 할당하고 가는걸로 합시다. 

        ###보완 예정 작업###
        1. 모든 text를 nlp를 이용할 필요 없다. -> is_in_data로 판단하면 됨 
        2. 일부 data에 포함되지 않는 경우 -> 이때만 자연어 처리를 이용해도 됨 
        '''
        if self.is_in_data() : #when is_in_data is True
            '이 경우에는 data의 구체적 구조가 필요하여 보류합니다.'
        else : 
            self.subject= extract_topics(self.text)
        
    def sebject_of_message(self) : 
        return self.subject #return subject

    def is_important(self) : 
        '''
        중요도 판단 메서드입니다. 
        중요도에 따른 재난을 분류할 기준은 데이터 분석 이후 정리하는 것을 목표로 합니다. 
        return 에는 다른 기능(필터링, tts등)과 연결될 수 있게 작성하는것을 목표로 합니다. 
        '''
        importance = self.imp[self.find_subject()]
        std_level = 5 #임의적 설정. 이후에 논의 후 변경
        if importance >= std_level : 
            return '심각' 
        else : 
            return '별거 아닌듯?' 
    def is_in_data(self) : 
        '''
        data : 재난문자 발송 기준 or 사전 모델링된 재난문자 데이터
        text가 기존 데이터에 있다면 그냥 원래 방법으로 진행하면 되는거고 data에 text가 포함될 수 없는 형식이라면 새로운 처리 과정을 거치는게 유리함
        '''
        return Similarity(self.text, data)


