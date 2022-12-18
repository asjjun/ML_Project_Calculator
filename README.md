# MachineLearning

<p align="center">
<br>
<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
<img src="https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=NumPy&logoColor=4ba6c9"/>
<img src="https://img.shields.io/badge/TensorFlow-efefef?style=flat-square&logo=TensorFlow&logoColor=FF6F00"/>
<img src="https://img.shields.io/badge/scikit learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/><br>
<img src="https://img.shields.io/badge/PyCharm-000000?style=flat-square&logo=PyCharm&logoColor=white"/>
<img src="https://img.shields.io/badge/Anaconda-44A833?style=flat-square&logo=Anaconda&logoColor=white"/>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
      <ul>
        <li><a href="#환경-설정">환경 설정</a></li>
      </ul>
    </li>
    <li>
      <a href="#본문">본문</a>
      <ul>
        <li><a href="#천체-분류">천체 분류</a></li>
        <li><a href="#사칙연산-구현">사칙연산 구현</a></li>
        <li><a href="#x-ray-이미지-분류">X-Ray 이미지 분류</a></li>
      </ul>
    </li>
    <li><a href="https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE10664524">연구 논문</a></li>
  </ol>
</details>

## Introduction
 > 머신러닝을 공부하며 만든 과제 및 학회 논문

## 환경 설정
     - Pycharm 사용, Anaconda 가상환경 사용
     - Python Interpreter 경로 Conda환경으로 설정
     - 각 프로젝트에 사용된 라이브러리 설치
     
## 본문

<h3>천체 분류</h3>

- 주제 설명\
\- SDSS(Sloan Digital Sky Survey)에서 측광관측 장비를 통해 100000여개 천체를 촬영한 것을 정리해둔 데이터셋 사용 (kaggle)\
\- 천체의 기울기와 분광특성, 적색편이 값 등을 통해 별, 은하, 퀘이사로 분류
  * 출처: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

- 입력데이터\
\- 적경 각도(알파), 기울기 각도(델타), 측광 시스템의 자외선 필터(u), 측광 시스템의 녹색 필터(g), 측광 시스템의 빨간색 필터(r),
    측광 시스템의 근적외선 필터(i), 측광 시스템의 적외선 필터(z), 적색편이. 총 8개의 특성

 - 출력데이터\
\- 객체등급(class, 별 or 은하 or 퀘이사)

 - 입력데이터의 정규화\
\- 각 특성 값의 범위가 다르기때문에 한 특성의 영향을 크게 받는 것을 방지하기위해 StandardScaler()를 사용해 정규화

- Logistic Regression\
\- 3개의 클래스(별, 항성, 퀘이사)에 따른 다항식의 계수
   
   <img width="500px" height = "110px"  src="https://user-images.githubusercontent.com/29851772/208314864-b8706185-e3eb-4c92-ae17-b40707a0dbde.png">

- Decision Tree\
\- tree depth를 조정해가며 정확도 측정 결과 depth= 12일 때가 가장 적합. 가장 높은 정확도를 보임
   
   <img width="400px" height = "75px"  src="https://user-images.githubusercontent.com/29851772/208315065-1217061f-c40d-48b8-9608-90bee7426f06.png">

- K-Neighbor\
\- 이웃 개수인 k 값이 커질수록 정확도가 작아지므로 k=3일때 가장 높은 정확도를 보임\
\- 이 데이터셋의 분류에는 4가지 방법 중 가장 정확도가 떨어짐

   <img width="180px" height = "220px"  src="https://user-images.githubusercontent.com/29851772/208315285-b2b7ec64-4c5b-4fc5-8127-216bc9988a7f.png">
   
- MLP\
\- 다음과 같이 모델 구성
\- input layer의 노드 수는 입력데이터 특성의 개수인 8\
\- output layer의 노드 수는 출력데이터 class의 개수인 3\
\- softmax로 마지막엔 각 class의 확률 값이 나오고 가장 높은 확률이 예측값\
\- optimizer는 'adam', loss는 다중분류에 쓰이는 'sparse_categorical_crossentropy' 사용, accuracy로 정확도 확인

   <img width="650px" height = "120px"  src="https://user-images.githubusercontent.com/29851772/208315314-6f28d087-ad37-4aeb-9168-93a9b40913c2.png">
   
 - 결과
 
   ![image](https://user-images.githubusercontent.com/29851772/208314740-cf5f0e37-cf6a-45e2-8b5a-1e53748c523e.png)



<h3>사칙연산 구현</h3>
  
  머신러닝으로 사칙연산을 구현해도 오차없이 정확한 값을 구할 수는 없다.\
  하지만 MLP에서 데이터가 대략 어떻게 진행되며 Node를 거쳐가는지,\
  따라서 모델을 어떻게 구성하고 데이터를 전처리 해야 하는지 공부 할 수 있었다.

- 변수 설명

  <img width="450px" height = "150px"  src="https://user-images.githubusercontent.com/29851772/208316200-a2cbad21-7fe7-4ecc-8b4c-7f827c0dbaa6.png">

  - symbol 값을 +, -, *, / 로 바꿔가며 각각 다른 연산 가능     
  - num_numbers의 값으로 연산할 숫자의 개수 설정 
  - largest의 값으로 입력 숫자의 범위를 설정하는데 범위가 커질수록 학습데이터를 많이 설정해줘야 제대로 된 학습 가능









<h3>X-Ray 이미지 분류</h3>

















