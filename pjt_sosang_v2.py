#%% Lib: 기본
import pymongo
import json
import sys
import pandas as pd
import numpy as np
import time
import datetime
import math
import openpyxl

import os
#import glob
import re
from ast import literal_eval

from konlpy.tag import Okt

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib


#%% DB 커넥션

conn = pymongo.MongoClient(host='218.232.111.79', port=27017, unicode_decode_error_handler='ignore')

db = conn['jejusc'] #데이터 베이스

collectionList = ["jejusc_review_202007","jejusc_document_202007", 
              "jejusc_review_202008","jejusc_document_202008",                  
              "jejusc_review_202009","jejusc_document_202009",
              "jejusc_review_202010","jejusc_document_202010",
              "jejusc_review_202011","jejusc_document_202011"]              


# 함수: DB 세팅 - DB별 세팅
db_number = 9 # 0부터 시작
doc_type = 2  # review=1, document=2

db_name = collectionList[db_number]  # 테스트

if doc_type == 1:
    max_words=30
elif doc_type == 2:
    max_words=300
else: 
    print("dic_type 에러입니다.")    

# 함수: DB 세팅 공통
collection = db[db_name]

# collection_list = db.collection_name()
# collection.inset = ["": ]

#전체조회
# results = collection.find() #find = select
# for result in results:
#     print(result)

#조건조회
# results = collection.find({"filtered": 1, "sentimented": 1, "copied": 1}) #find = select

results = collection.find({"filtered": 1}) #find = select
#results = collection.find({"filtered": 1}).limit(10) #find = select
# for result in results:
#     print(result)

# collection.update({},{},upsert=true,False)# {}키,{}값이 있을경우 진행,없을경우 insert
# collection.update({},{},usert=true, multi=True)# {}키,{}값이 모두 있을 경우 여려개 모두 update

# collection.remove({"":{:$tg}})

# DF 생성 (하단 코드와 병합)
content_df = pd.DataFrame(results)
#print(df)

#%% (설명) 사전종류

#1 UNINAN
'''
3,717 개

매우 긍정(2)
긍      정(1)	
중      립(0)	
부      정(-1)	
매우 부정(-2)	
'''
#2 군산대
# http://dilab.kunsan.ac.kr/knusl.html
'''
14,843 개

긍부정어 통계
긍부정어	단어개수
1-gram 긍부정어	6,223
2-gram 긍부정어	7,861
긍부정 어구	278
긍부정 문형	253
긍부정 축약어	174
긍부정 이모티콘	54
1-gram 긍부정어	6,451
2-gram 긍부정어	8,135
3-gram 긍부정어	226
4-gram 긍부정어	20
5-gram 긍부정어	5
6-gram 긍부정어	3
7-gram 긍부정어	2
8-gram 긍부정어	1
매우 긍정(2)	2,597
긍      정(1)	2,266
중      립(0)	154
부      정(-1)	5,029
매우 부정(-2)	4,797
'''
#3 감성사전


#%% 실행: 토근화 DF 생성

if doc_type == 1:
    token_df = pd.DataFrame(columns=['_id','point','p_words','n_words','token_words','tokens','train_tokens','d1_tokens','d2_tokens','d3_tokens','p1','n1','p2','n2','p3','n3','_p1','_n1','_p2','_n2','_p3','_n3','p_score','n_score','sentimented'])
    token_df[['_id','point']] = content_df[['_id','point']]
    token_df[['p_words','n_words','token_words','tokens','train_tokens','d1_tokens','d2_tokens','d3_tokens','d1','d2','d3']] = ''
    token_df[['p1','n1','p2','n2','p3','n3','_p1','_n1','_p2','_n2','_p3','_n3','p_score','n_score','sentimented']]=0
elif doc_type == 2:    
    token_df = pd.DataFrame(columns=['_id','p_words','n_words','token_words','tokens','train_tokens','d1_tokens','d2_tokens','d3_tokens','p1','n1','p2','n2','p3','n3','_p1','_n1','_p2','_n2','_p3','_n3','p_score','n_score','sentimented'])
    token_df['_id'] = content_df['_id']
    token_df[['p_words','n_words','token_words','tokens','train_tokens','d1_tokens','d2_tokens','d3_tokens','d1','d2','d3']] = ''
    token_df[['p1','n1','p2','n2','p3','n3','_p1','_n1','_p2','_n2','_p3','_n3','p_score','n_score','sentimented']]=0
else:
    print("doc_type 에러입니다.")

#%% 실행: 사전 카테고리명 지정
# 사전별 카테고리명 (전체단어 통계에는 카테고리명 반영)    
ca_d1 = "UNINAN"
ca_d2 = "KNU"
ca_d3 = "감성"
    
#%% 실행: 사전 단어 1
stime = time.time()   # 시작시간
print(ca_d1,"사전을 로딩 중입니다. 기다려주세요")
path = 'C:/Users/user/.spyder-py3/e2on/p_sosang/uninan.xlsx'   
d1_df = pd.read_excel(path,header=0)
d1_df.drop_duplicates(['단어'],keep='last',inplace=True)
d1_df.reset_index(drop=True,inplace=True)
d1_df.to_csv('./d1_df.csv')
etime = time.time()   # 종료시간
print("# 소요시간 : ", str(datetime.timedelta(seconds=etime-stime)))    

#%% 실행: 사전 단어 2 
stime = time.time()   # 시작시간
print(ca_d2,"사전을 로딩 중입니다. 기다려주세요")
d2_df = pd.DataFrame(columns=['단어','점수'])
p1 = re.compile("[가-힣]+")   
p2 = re.compile("([가-힣]+)[ ]+[가-힣]+[가-힣 ]*")                    
path = './KnuSentiLex-master/SentiWord_Dict.txt'
with open('./KnuSentiLex-master/data/SentiWord_info.json', encoding='utf-8-sig', mode='r') as f:
    data = json.load(f) # 리스트내 집합
    #print(len(data))
    for n, word in enumerate(data):
        if type(word['word']) == str:
            if p1.search(word['word']) == None: pass
            else:
                word['word'] = re.sub(p2,"\\1",word['word'])
                #if len(word['word']) == 1 and word['word'] in ['다','이','제','그','데','산','성','해','을','늘','안','더','온']:
                if len(word['word']) == 1:
                    pass
                else: 
                    d2_df.loc[n,'단어'] = word['word']
                    d2_df.loc[n,'점수'] = word['polarity']
                       
d2_df = d2_df.dropna(subset=["단어"])
d2_df.drop_duplicates(['단어'],keep='last',inplace=True)
d2_df.reset_index(drop=True,inplace=True)
d2_df.to_csv('./d2_df.csv')           
etime = time.time()   # 종료시간
print("# 소요시간 : ", str(datetime.timedelta(seconds=etime-stime)))    

#%% 실행: 사전 단어 3
stime = time.time()   # 시작시간
print(ca_d3,"사전을 로딩 중입니다. 기다려주세요")
dic = {}
# dic_list = []
num_files = 11
for n in range(1,num_files):
    dic_key = ('{0}'.format(n))
    dic_item = 'jejusc'+str(n) 
    dic[dic_key] = dic_item   # 파일명 사전형으로 묶음
    # dic_list.append(dic_item)
    
    path = './jejusc'+str(n)+'.xlsx'   
    d3_df = pd.read_excel(path,header=0)  # 엑셀파일 1개로 합치기
d3_df.drop_duplicates(['단어'],keep='last',inplace=True)    
d3_df.reset_index(drop=True,inplace=True)
d3_df.to_csv('./d3_df.csv')
etime = time.time()   # 종료시간
print("# 소요시간 : ", str(datetime.timedelta(seconds=etime-stime)))

#%% 함수: 형태소 분석

from nltk.tokenize import word_tokenize
import nltk
from konlpy.tag import Okt

# 형태소분석기
pos_tagger = Okt()

def tokenize(db_number, doc_type, max_words):  # max_words = 30 (리뷰)/ 300 (문서)   # 문서 500단어 결과값이 안나와서 300으로 줄임   
    db_name = collectionList[db_number]


# 라인별로 읽기   
    for n in range(len(content_df)):  
    #for n in range(10):        #테스트
        print(db_name,"형태소 분석 - 문서:",n) # DB명        
        content = content_df['content'][n]
        content = content.strip() #속도 안느릴 경우 .lower() 추가
        word_list = content.split(' ')
        if content == '':
            token_df['sentimented'][n] = 2 #리뷰가 없는 경우 sentimented에 표시
            continue

        num_words = len(word_list)
        if num_words > max_words:
            print("컨텐츠 사이즈가 너무 큽니다. 단어 {0}개로 줄입니다.".format(max_words))
            content = ' '.join(word_list[0:max_words])
            #word_list = content.split(' ') #30단어로 줄인 단어리스트
        #token_df['word_list'][n] = word_list  # 원하는 단어가 아님
        token_df['tokens'][n] = ['/'.join(t) for t in pos_tagger.pos(content, norm=True, stem=True)]
        temp_list = [re.sub('[\a-zA-Z]+', '', t) for t in token_df['tokens'][n]]
        #temp_list = [re.sub('[^가-힣]+','',item) for item in temp_list]    # 뭔가 이상함. 추후 체크 !!!!!!!!!!!!!!!
        token_df['token_words'][n] = list(filter(None,temp_list))
        if doc_type == 1:
            token_df['train_tokens'][n] = [(token_df['tokens'][n],token_df['point'][n]>=3 and 'pos' or 'neg')]    # 문서는 결과값 없으므로 미사용
        elif doc_type == 2:
            pass
        else: 
            print("잘못된 문서형태입니다.")
    #return

def token_exists(doc,n):
    tokens = token_df['tokens'][n] 
    return {word: (word in set(doc)) for word in tokens}

def train_xy(n):
    train_xy = [(token_exists(doc,n),c) for doc,c in token_df['train_tokens'][n]]
    print(train_xy)
    return train_xy

#실행: 형태소 분석
tokenize(db_number=db_number, doc_type=doc_type, max_words=max_words)  


#%% 함수: 사전별 검출단어 분석

def token_dic(dic_num):
    ################################## 사전단어 보유여부 체크 (tokens에서 검색)        
    for n in range(len(token_df)): 
    #for n in range(10):        #테스트
           
        print(dic_num,"dic 검출단어 분석:",n)  
        
        # 각 문서별 단어 통계
        word_dic = {}
        count_dic = {}
        tokens = []
        token_words = token_df.loc[n,'token_words'] #형태소분석된 tokens_words 기준 검색 예정
        
        if dic_num == 1:
            token_df['d1_tokens'][n] = ''  #사전별 단어 초기화 (기존에 단어가 있을 경우 오류발생 가능성 차단)
        elif dic_num == 2:
            token_df['d2_tokens'][n] = ''
        elif dic_num == 3:
            token_df['d3_tokens'][n] = ''
        else:
            print("함수호출 에러입니다.")
            break
        
        for i, word1 in enumerate(token_words):  #token_words 회전
            if dic_num == 1:
                #tokens = [m.group() for m in re.finditer(word1, content_df.loc[n,'content'])]  #content에서 검색
                for j, word2 in enumerate(d1_df['단어']): #사전 회전                    
                    if word1 == word2:
                        tokens.append(word1)
                token_df['d1_tokens'][n] = tokens
                
            elif dic_num == 2:
                for j, word2 in enumerate(d2_df['단어']): #사전 회전
                    if word1 == word2:
                        tokens.append(word1)
                token_df['d2_tokens'][n] = tokens
                
            elif dic_num == 3:                
                for j, word2 in enumerate(d3_df['단어']): #사전 회전                
                    if word1 == word2:
                        tokens.append(word1)
                token_df['d3_tokens'][n] = tokens
                
            else:
                print("함수호출 에러입니다.")
                break

# 실행: 사전별 검출단어 분석
token_dic(dic_num=1)
token_dic(dic_num=2)
token_dic(dic_num=3)

#%% 함수: 사전 점수

def point_dic(dic_num):
                          
    for n in range(len(token_df)):  
    #for n in range(10):        #테스트
        print("사전{0} 점수계산:".format(dic_num),n) 
        positive = 0
        negative = 0     
        score = 0
        
        if dic_num == 1:
            token_list = token_df['d1_tokens'][n]
            dictionary = d1_df['단어']
        elif dic_num == 2:
            token_list = token_df['d2_tokens'][n]     
            dictionary = d2_df['단어']
        elif dic_num == 3:
            token_list = token_df['d3_tokens'][n]
            dictionary = d3_df['단어']            
        else: 
            print("함수호출 에러입니다.")   
            break
        
        for i, word1 in enumerate(token_list):
            for j, word2 in enumerate(dictionary):  #사전단어 찾기
                if word1 == word2:
                    if dic_num == 1:
                        score = int(d1_df['점수'][j])
                    elif dic_num == 2:
                        score = int(d2_df['점수'][j])
                    else:
                        score = int(d3_df['점수'][j])  
                                                
                    if score > 0: 
                        print("긍정",score)
                        positive += score
                    elif score < 0:
                        print("부정",score)
                        negative += score
                    else:
                        pass
                    break
        if dic_num == 1:         
            token_df['p1'][n] = positive
            token_df['n1'][n] = negative
        elif dic_num == 2:         
            token_df['p2'][n] = positive
            token_df['n2'][n] = negative
        else:         
            token_df['p3'][n] = positive
            token_df['n3'][n] = negative
        
# 실행: 사전 점수계산
point_dic(dic_num=1)
point_dic(dic_num=2)
point_dic(dic_num=3)

#%% 함수: 최종 점수  # review: point 존재, document: point 미존재
# 참고: 리뷰점수 없는 경우는 포인트점수 반영

def evaluate():             
    if doc_type == 1:
        for n in range(len(token_df)):  
        #for n in range(10):        #테스트        
    
            if token_df.loc[n,'p1'] == 0:      #리뷰/문서 긍정단어점수가 없는 경우
                if token_df.loc[n,'n1'] == 0:  #리뷰/문서 부정단어점수가 없는 경우
                    token_df.loc[n,'_p1'] = round(token_df.loc[n,'point']/5, 6)      
                    token_df.loc[n,'_n1'] = round(1 - token_df.loc[n,'_p1'],6)
                else:                          #리뷰/문서 부정단어점수가 있는 경우
                    token_df.loc[n,'_p1'] = round((0.5 - (token_df.loc[n,'n1']/token_df['n1'].min())*0.1 ), 6) #최소값 0.1
                    token_df.loc[n,'_n1'] = round(1 - token_df.loc[n,'_p1'],6)
            else:                              #리뷰/문서 단어점수가 있는 경우
                if token_df.loc[n,'p1'] >= math.fabs(token_df.loc[n,'n1']):  #긍정단어점수가 부정단어점수보다 큰 경우
                    token_df.loc[n,'_p1'] = round(( token_df.loc[n,'point']/5 + (token_df.loc[n,'p1']/token_df['p1'].max())*0.1 ), 6) #최대값 1.0
                    if token_df.loc[n,'_p1'] > 1:
                        token_df.loc[n,'_p1'] = 1                    
                    token_df.loc[n,'_n1'] = round(1 - token_df.loc[n,'_p1'],6)
                else:                                                        #부정단어점수가 긍정단어점수보다 큰 경우
                    token_df.loc[n,'_p1'] = round(( token_df.loc[n,'point']/5 - (token_df.loc[n,'n1']/token_df['n1'].min())*0.1 ), 6) #최소값 0.1             
                    if token_df.loc[n,'_p1'] < 0.1:
                        token_df.loc[n,'_p1'] = 0.1                      
                    token_df.loc[n,'_n1'] = round(1 - token_df.loc[n,'_p1'],6)
            
            if token_df.loc[n,'p2'] == 0:
                if token_df.loc[n,'n2'] == 0:
                    token_df.loc[n,'_p2'] = round(token_df.loc[n,'point']/5, 6)          
                    token_df.loc[n,'_n2'] = round(1 - token_df.loc[n,'_p2'],6)
                else:
                    token_df.loc[n,'_p2'] = round((0.5 - (token_df.loc[n,'n2']/token_df['n2'].min())*0.1 ), 6) #최소값 0.1
                    token_df.loc[n,'_n2'] = round(1 - token_df.loc[n,'_p2'],6)
            else: 
                if token_df.loc[n,'p2'] >= math.fabs(token_df.loc[n,'n2']):
                    token_df.loc[n,'_p2'] = round(( token_df.loc[n,'point']/5 + (token_df.loc[n,'p2']/token_df['p2'].max())*0.1 ), 6) #최대값 1.0
                    if token_df.loc[n,'_p2'] > 1:
                        token_df.loc[n,'_p2'] = 1        
                    token_df.loc[n,'_n2'] = round(1 - token_df.loc[n,'_p2'],6)
                else:
                    token_df.loc[n,'_p2'] = round(( token_df.loc[n,'point']/5 - (token_df.loc[n,'n2']/token_df['n2'].min())*0.1 ), 6) #최소값 0.1                
                    if token_df.loc[n,'_p2'] < 0.1:
                        token_df.loc[n,'_p2'] = 0.1                      
                    token_df.loc[n,'_n2'] = round(1 - token_df.loc[n,'_p2'],6)
            
            if token_df.loc[n,'p3'] == 0:
                if token_df.loc[n,'n3'] == 0:
                    token_df.loc[n,'_p3'] = round(token_df.loc[n,'point']/5, 6)
                    token_df.loc[n,'_n3'] = round(1 - token_df.loc[n,'_p3'],6)
                else:
                    token_df.loc[n,'_p3'] = round((0.5 - (token_df.loc[n,'n3']/token_df['n3'].min())*0.1 ), 6) #최소값 0.1
                    token_df.loc[n,'_n3'] = round(1 - token_df.loc[n,'_p3'],6)
            else: 
                if token_df.loc[n,'p3'] >= math.fabs(token_df.loc[n,'n3']):
                    token_df.loc[n,'_p3'] = round(( token_df.loc[n,'point']/5 + (token_df.loc[n,'p3']/token_df['p3'].max())*0.1 ), 6) #최대값 1.0
                    if token_df.loc[n,'_p3'] > 1:
                        token_df.loc[n,'_p3'] = 1                
                    token_df.loc[n,'_n3'] = round(1 - token_df.loc[n,'_p3'],6)
                else:
                    token_df.loc[n,'_p3'] = round(( token_df.loc[n,'point']/5 - (token_df.loc[n,'n3']/token_df['n3'].min())*0.1 ), 6) #최소값 0.1                
                    if token_df.loc[n,'_p3'] < 0.1:
                        token_df.loc[n,'_p3'] = 0.1      
                    token_df.loc[n,'_n3'] = round(1 - token_df.loc[n,'_p3'],6)
                    
            token_df.loc[n,'p_score'] = round((token_df.loc[n,'_p1'] + token_df.loc[n,'_p2'] + token_df.loc[n,'_p3'])/3, 6)
            token_df.loc[n,'n_score'] = round((token_df.loc[n,'_n1'] + token_df.loc[n,'_n2'] + token_df.loc[n,'_n3'])/3, 6)

    elif doc_type == 2:
        for n in range(len(token_df)):  
        #for n in range(10):        #테스트        
            print("n:",n) 
            
            if token_df.loc[n,'p1'] == 0:
                if token_df.loc[n,'n1'] == 0:
                    token_df.loc[n,'_p1'] = 0.5            
                    token_df.loc[n,'_n1'] = 0.5
                else:
                    token_df.loc[n,'_p1'] = round((0.5 - (token_df.loc[n,'n1']/token_df['n1'].min())*0.4 ), 6) #최소값 0.1
                    token_df.loc[n,'_n1'] = round(1 - token_df.loc[n,'_p1'],6)
            else: 
                if token_df.loc[n,'p1'] >= math.fabs(token_df.loc[n,'n1']):
                    token_df.loc[n,'_p1'] = round((0.5 + (token_df.loc[n,'p1']/token_df['p1'].max())*0.5 ), 6) #최대값 1.0
                    if token_df.loc[n,'_p1'] > 1:
                        token_df.loc[n,'_p1'] = 1                
                    token_df.loc[n,'_n1'] = round(1 - token_df.loc[n,'_p1'],6)
                else:
                    token_df.loc[n,'_p1'] = round((0.5 - (token_df.loc[n,'n1']/token_df['n1'].min())*0.4 ), 6) #최소값 0.1          
                    #if token_df.loc[n,'_p1'] < 0.1:
                        #token_df.loc[n,'_p1'] = 0.1                      
                    token_df.loc[n,'_n1'] = round(1 - token_df.loc[n,'_p1'],6)
            
            if token_df.loc[n,'p2'] == 0:
                if token_df.loc[n,'n2'] == 0:
                    token_df.loc[n,'_p2'] = 0.5            
                    token_df.loc[n,'_n2'] = 0.5
                else:
                    token_df.loc[n,'_p2'] = round((0.5 - (token_df.loc[n,'n2']/token_df['n2'].min())*0.4 ), 6) #최소값 0.1
                    token_df.loc[n,'_n2'] = round(1 - token_df.loc[n,'_p2'],6)
            else: 
                if token_df.loc[n,'p2'] >= math.fabs(token_df.loc[n,'n2']):
                    token_df.loc[n,'_p2'] = round((0.5 + (token_df.loc[n,'p2']/token_df['p2'].max())*0.5 ), 6) #최대값 1.0
                    if token_df.loc[n,'_p2'] > 1:
                        token_df.loc[n,'_p2'] = 1                       
                    token_df.loc[n,'_n2'] = round(1 - token_df.loc[n,'_p2'],6)
                else:
                    token_df.loc[n,'_p2'] = round((0.5 - (token_df.loc[n,'n2']/token_df['n2'].min())*0.4 ), 6) #최소값 0.1                
                    #if token_df.loc[n,'_p2'] < 0.1:
                        #token_df.loc[n,'_p2'] = 0.1                      
                    token_df.loc[n,'_n2'] = round(1 - token_df.loc[n,'_p2'],6)
            
            if token_df.loc[n,'p3'] == 0:
                if token_df.loc[n,'n3'] == 0:
                    token_df.loc[n,'_p3'] = 0.5            
                    token_df.loc[n,'_n3'] = 0.5
                else:
                    token_df.loc[n,'_p3'] = round((0.5 - (token_df.loc[n,'n3']/token_df['n3'].min())*0.4 ), 6) #최소값 0.1
                    token_df.loc[n,'_n3'] = round(1 - token_df.loc[n,'_p3'],6)
            else: 
                if token_df.loc[n,'p3'] >= math.fabs(token_df.loc[n,'n3']):
                    token_df.loc[n,'_p3'] = round((0.5 + (token_df.loc[n,'p3']/token_df['p3'].max())*0.5 ), 6) #최대값 1.0 
                    if token_df.loc[n,'_p3'] > 1:
                        token_df.loc[n,'_p3'] = 1                
                    token_df.loc[n,'_n3'] = round(1 - token_df.loc[n,'_p3'],6)
                else:
                    token_df.loc[n,'_p3'] = round((0.5 - (token_df.loc[n,'n3']/token_df['n3'].min())*0.4 ), 6) #최소값 0.1                
                    #if token_df.loc[n,'_p3'] < 0.1:
                        #token_df.loc[n,'_p3'] = 0.1                      
                    token_df.loc[n,'_n3'] = round(1 - token_df.loc[n,'_p3'],6)
                    
            token_df.loc[n,'p_score'] = round((token_df.loc[n,'_p1'] + token_df.loc[n,'_p2'] + token_df.loc[n,'_p3'])/3, 6)
            token_df.loc[n,'n_score'] = round((token_df.loc[n,'_n1'] + token_df.loc[n,'_n2'] + token_df.loc[n,'_n3'])/3, 6)
        
# 실행: 문서평가 -point 없음
evaluate()
'''
token_df.loc[100,'p1'] #테스트
token_df.loc[100,'p2'] #테스트
token_df.loc[100,'p3'] #테스트
token_df.loc[100,'_p1'] #테스트
token_df.loc[100,'_p2'] #테스트
token_df.loc[100,'_p3'] #테스트
token_df.loc[100,'_n1'] #테스트
token_df.loc[100,'_n2'] #테스트
token_df.loc[100,'_n3'] #테스트
token_df.loc[100,'p_score'] #테스트
token_df.loc[100,'d1_tokens'] #테스트
token_df.loc[100,'d2_tokens'] #테스트
token_df.loc[100,'d3_tokens'] #테스트
'''
#%% 함수: 긍정부정 단어
#del token_df['point']
#token_df.rename(columns = {'lines':'p_words'},inplace=True)
#token_df.rename(columns = {'class':'n_words'},inplace=True)  

def word_reference():
    temp_df = pd.Series()
    token_df['temp'] = token_df['d1_tokens'] + token_df['d2_tokens'] + token_df['d3_tokens']
                      
    for n in range(len(token_df)):  
    #for n in range(10):        #테스트     
        print("긍정부정:",n) 
        p_words = []
        n_words = []
        token_list = list(set(token_df['temp'][n]))
        
        for i, word1 in enumerate(token_list):
            for j, word2 in enumerate(d1_df['단어']):  #사전단어 찾기
                if word1 == word2:
                    score = int(d1_df['점수'][j])
                    if score > 0: 
                        p_words.append(d1_df['단어'][j])
                    elif score < 0:
                        n_words.append(d1_df['단어'][j])
                    break
            for j, word2 in enumerate(d2_df['단어']):  #사전단어 찾기
                if word1 == word2:
                    score = int(d2_df['점수'][j])
                    if score > 0: 
                        p_words.append(d2_df['단어'][j])
                    elif score < 0:
                        n_words.append(d2_df['단어'][j])
                    break
            for j, word2 in enumerate(d3_df['단어']):  #사전단어 찾기
                if word1 == word2:
                    score = int(d3_df['점수'][j])
                    if score > 0: 
                        p_words.append(d3_df['단어'][j])
                    elif score < 0:
                        n_words.append(d3_df['단어'][j])
                    break
                        
        p_words = list(set(p_words))
        n_words = list(set(n_words))
        bad_words = ['이다','하다','세','되다']
        for i, item in enumerate(p_words):
            if item in bad_words:
                del p_words[i]
            else: pass
        
        for i, item in enumerate(n_words):
            if item in bad_words:
                del n_words[i]
                
        token_df['p_words'][n] = p_words
        token_df['n_words'][n] = n_words
    del  token_df['temp']
    
# 실행: 긍정부정 단어
word_reference()

#%% 함수: 데이터저장 체크
def check():
    for n in range(len(token_df)):  
    #for n in range(10):        #테스트     
       
        if token_df['p_score'][n] != 0:
            token_df['sentimented'][n] = 1
        elif type(token_df['p_score'][n]) == 0:
            token_df['sentimented'][n] = 2  #점가 없는 경우 2
        else:
            token_df['sentimented'][n] = 3                        
    
# 실행: 데이터저장 체크

check()

#%% 몽고DB 저장
'''
client = MongoClient('mongodb://%s:%s@아이피' % ('아이디','비번'), 27017)
db = client.DB명
collection = db.콜렉션명
'''

'''
db.컬렉션명.update({_id: ObjectId('24자리키')}, {$set: {sentimented: 1, .......}});
'''
from pymongo import MongoClient
from bson.objectid import ObjectId

def save():
    for n in range(len(token_df)):  
    #for n in range(10):        #테스트   

        collection.update({"_id": ObjectId(token_df['_id'][n])},
                          {"$set":
                          {
                            "sentiment" : {                          
                                "positive" : {
                                    "score" : token_df['p_score'][n],      
                                    "words" : token_df['p_words'][n]
                                }, 
                            
                                "negative" : {                           
                                    "score" : token_df['n_score'][n], 
                                    "words" : token_df['n_words'][n]                                              
                                }                           
                            },           
            "sentimented": int(token_df['sentimented'][n])
            }})
                    
save()

#%% (임시) 백업
#리뷰
content_df.to_csv('./content_rev7월_df.csv')
content_df.loc[:120,].to_excel('./content_rev7월_df_120개.xlsx') #테스트 자료
token_df.to_csv('./token_rev7월_df.csv')
token_df.loc[:120,].to_excel('./token_rev7월_df_120개.xlsx') #테스트 자료

content_df.to_csv('./content_rev8월_df.csv')
token_df.to_csv('./token_rev8월_df.csv')

content_df.to_csv('./content_rev9월_df.csv')
token_df.to_csv('./token_rev9월_df.csv')

#문서
content_df.to_csv('./content_doc7월_df.csv')
content_df.loc[:130,].to_excel('./content_doc7월_df_120개.xlsx') #테스트 자료
token_df.to_csv('./token_doc7월_df.csv')
token_df.loc[:130,].to_excel('./token_doc7월_df_120개.xlsx') #테스트 자료

content_df.to_csv('./content_doc8월_df.csv')
token_df.to_csv('./token_doc8월_df.csv')

content_df.to_csv('./content_doc9월_df.csv')
token_df.to_csv('./token_doc9월_df.csv')

content_df.to_csv('./content_doc10월_df.csv')
token_df.to_csv('./token_doc10월_df.csv')

content_df.to_csv('./content_doc10월_df.csv')
token_df.to_csv('./token_doc10월_df.csv')

#%% (임시) 로딩
#리뷰
content_df = pd.read_csv('./content_rev7월_df.csv')
token_df = pd.read_csv('./token_rev7월_df.csv')

content_df = pd.read_csv('./content_rev8월_df.csv')
token_df = pd.read_csv('./token_rev8월_df.csv')

content_df = pd.read_csv('./content_rev9월_df.csv')
token_df = pd.read_csv('./token_rev9월_df.csv')

#문서
content_df = pd.read_csv('./content_doc7월_df.csv')
token_df = pd.read_csv('./token_doc7월_df.csv')

content_df = pd.read_csv('./content_doc8월_df.csv')
token_df = pd.read_csv('./token_doc8월_df.csv')

content_df = pd.read_csv('./content_doc9월_df.csv')
token_df = pd.read_csv('./token_doc9월_df.csv')


d1_df = pd.read_csv('./d1_df.csv')
d2_df = pd.read_csv('./d2_df.csv')
d3_df = pd.read_csv('./d3_df.csv')