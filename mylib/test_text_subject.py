# 라이브러리 임포트
from konlpy.tag import Okt
from keybert import KeyBERT

# 텍스트 전처리 및 주제 추출 함수
def extract_topics(text):
    okt = Okt()
    # 형태소 분석을 통해 명사만 추출
    nouns = okt.nouns(text)
    # 추출된 명사 리스트를 하나의 문자열로 결합
    noun_text = ' '.join(nouns)
    # KeyBERT를 사용해 키워드 추출
    kw_model = KeyBERT('paraphrase-MiniLM-L6-v2')
    keywords = kw_model.extract_keywords(noun_text, top_n=3)
    return [keyword[0] for keyword in keywords]

print(extract_topics('7.22.마카켐(신척산단5로130)의 유독물질 유출사고는 관계기관 등과 안전조치 등을 완료했음을 알려드립니다. [진천군]'))