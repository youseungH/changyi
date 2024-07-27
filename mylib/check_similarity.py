from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    # [CLS] 토큰의 임베딩을 사용
    cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return cls_embedding

sentence1 = "현재 폭염특보 발효중. 농사일, 건설현장작업, 야외활동 시 충분한 물 섭취와 그늘 휴식, 주변 노약자분들께 안부 확인등 안전관리에 유의하시기 바랍니다.[충청북도]"
sentence2 = "전주시에서 실종된 민도식씨(남,70세)를 찾습니다-160cm,55kg,회색줄무늬티,검정칠부등산바지,검정운동화,지팡이\nvo.la/TXWtJ/ ☎182N"

embedding1 = get_sentence_embedding(sentence1)
embedding2 = get_sentence_embedding(sentence2)

# 코사인 유사도 계산
similarity = cosine_similarity(embedding1, embedding2)

print(f"두 문장의 유사도: {similarity[0][0]}")

##일단 실험 결과 전혀 상관 없는 내용의 문장들은 8n%대의 일치율을 보였음 -> 내용 일치의 경우 93%이상
#이중 체크를 위해 주제 체크도 할까?

def Similarity(text, datas):
    while True : 
        i = 0 
        text_embeded = get_sentence_embedding(text)
        data_embeded = get_sentence_embedding(datas[i])

        if cosine_similarity(text_embeded,data_embeded) >= 0.93 : 
            #구체적 점검 필요
            return True
        if i == len(datas) : 
            #끝까지 못찾는다면 Flase return 
            return False 
        i += 1

        
