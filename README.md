# JEJU_EMOTION
---


1. 목적
    - 제주 소셜 데이터 감성분석
2. 사용처
    - 제주소상공인
3. 데이터
    - 제주 소상공인 소셜 데이터
    - 감성사전
4. 운영
    - LANGUAGE
        - python 3.X
    - DB
        - mongoDB
    - 주요 library
        - pandas  
    - database.ini
        - 매달 최신화 시켜줘야함 한달씩 +1  
        - review 와 document 두 개가 있음 각기 다른 방식으로 감성분석됌 
    - 주의사항
        - 순서를 맞춰 진행해야함
        - review는 별점이 반영된 감성분석
        - document는 오로지 감성만 분석
        - 현 감성분석은 하나의 점수 몰리는 경향이 있음 실제 감성분석으로는 이용하기  어려움
5. 실행
    - database.ini 를 현행화 후 
    - pjt_sosang_v4_fast.py 를 실행시킨다. (review, document 각각 1번씩)
        
        
<!-- 주석 필요시 따로 사용
- [ ]  체크 X
- [x]  체크

- 점
> 인용
>> 재인용
```jsx
const name = "송"
```
```json
{"name": "송", "age": 28}
```
-->
