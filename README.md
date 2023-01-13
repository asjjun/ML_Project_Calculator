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
        <li><a href="#항성-분류">항성 분류</a></li>
        <li><a href="#사칙연산-구현">사칙연산 구현</a></li>
        <li><a href="#x-ray-이미지-분류">X-Ray 이미지 분류</a></li>
      </ul>
    </li>
  </ol>
</details>

## Introduction
 > 머신러닝을 공부하며 만든 과제

## 환경 설정
     - Pycharm 사용, Anaconda 가상환경 사용            
     
     - Python Interpreter 경로 Conda환경으로 설정           
     
     - numpy, tensorflow, keras, matplotlib, scikit-learn, pandas, opencv 등 각 프로젝트에 사용된 라이브러리 설치


     * 참고: 항성분류 프로젝트와 X-Ray CNN 모두 파일 다운 받고 엑셀파일, npy파일, h5파일을 py파일과 같은경로에 두고 그대로 사용    
            X-Ray CNN은 원본 이미지 파일이 용량이 커서 numpy 배열로 변환한 npy 파일 첨부
            원본 이미지 파일은 링크에서 다운로드 > https://drive.google.com/file/d/17MIx12tqVK5BH63hiEIcHiNvQbE2huz-/view?usp=sharing
            





     
## 본문

<h3>항성 분류</h3>

- 주제 설명\
\- SDSS(Sloan Digital Sky Survey)에서 측광관측 장비를 통해 100000여개 항성을 촬영한 것을 정리해둔 데이터셋 사용 (kaggle)\
\- 항성의 기울기와 분광특성, 적색편이 값 등을 통해 별, 은하, 퀘이사로 분류
  * 출처: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

- 입력데이터\
\- 적경 각도(알파), 기울기 각도(델타), 측광 시스템의 자외선 필터(u), 측광 시스템의 녹색 필터(g), 측광 시스템의 빨간색 필터(r),
    측광 시스템의 근적외선 필터(i), 측광 시스템의 적외선 필터(z), 적색편이. 총 8개의 특성

 - 출력데이터\
\- 객체등급(class, 별 or 은하 or 퀘이사)

 - 입력데이터의 정규화\
\- 각 특성 값의 범위가 다르기때문에 한 특성의 영향을 크게 받는 것을 방지하기위해 StandardScaler()를 사용해 정규화

- Logistic Regression\
\- 3개의 클래스(별, 은하, 퀘이사)에 따른 다항식의 계수
   
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


<br>
<h3>사칙연산 구현</h3>
  
  머신러닝으로 사칙연산을 구현해도 오차없이 정확한 값을 구할 수는 없다.\
  하지만 MLP에서 데이터가 대략 어떻게 진행되며 Node를 거쳐가는지,\
  따라서 모델을 어떻게 구성하고 데이터를 전처리 해야 하는지 공부 할 수 있었다.

- 변수 설명

  <img width="450px" height = "150px"  src="https://user-images.githubusercontent.com/29851772/208316200-a2cbad21-7fe7-4ecc-8b4c-7f827c0dbaa6.png">

  - symbol 값을 +, -, *, / 로 바꿔가며 각각 다른 연산 가능     
  - num_numbers의 값으로 연산할 숫자의 개수 설정 
  - largest의 값으로 입력 숫자의 범위를 설정하는데 범위가 커질수록 학습데이터를 많이 설정해줘야 제대로 된 학습 가능\
    (예시 사진은 적당히 100으로 설정)

- 입출력 데이터 생성

  <img width="490px" height = "450px"  src="https://user-images.githubusercontent.com/29851772/208497213-3ed000fd-0fa4-472e-9c70-50116c8914d4.png">
  <img width="325px" height = "450px"  src="https://user-images.githubusercontent.com/29851772/208496107-5b42f9c1-db26-4621-8021-f74b4b50aaed.png">
  
  - 왼쪽 사진은 입출력 데이터가 생성되는 코드인데 0 ~ 최대값 사이의 랜덤 실수를\
    num_numbers의 값 만큼 생성 후 symbol 값에 따라 연산 한다.
  - 그 결과 오른쪽 사진과 같은 형태의 입출력 데이터가 생성된다. (예시 사진은 덧셈 데이터 생성 결과)

 - 데이터 전처리 및 정규화
   - 덧셈\
   덧셈은 따로 전처리는 필요 없고 각 입출력 데이터의 값의 범위를 0~1로 만들어 정규화 해주기 위해 입력 최댓값(100)의 2배로 나눈다.\
   입력 최댓값(100)의 2배로 나눈다는 뜻은 예시의 경우에서 200으로 나눴다는 뜻.\
   사실 정규화는 모델에 입력될 데이터에 해주면 된다.\
   하지만 출력데이터도 같이 나눠주는것과 입력데이터만 나눠주는것에 모델 성능 차이가 있어서 함께 나눠주었다.
   - 뺄셈\
   뺄셈은 덧셈과 비슷하지만 출력결과에 음수가 나올 수 있다.\
   입출력 데이터의 값에 입력 최대값(100)을 더해주어 출력 값이 음수가 되는 것을 방지하고 입력 최댓값의 2배로 나눠 정규화.\
   예를들어 50-80=-30의 연산이라면 100을 더해 입력 값[150,180], 출력 값[70] 이 되고 이걸 200으로 나눔.
   - 곱셈\
   곱셈은 입력 값을 모두 곱한 값을 출력에 그대로 넣는 식으로 학습하면 제대로 학습이 안된다.\
   따라서 로그의 성질을 이용하여 처리.\
   ln xy = ln x + ln y이므로 xy의 계산을 하고싶다면 입력 값인 x와 y에 각각 log를 씌우고 더해준다음\
   log를 벗겨내면 xy의 값을 알 수 있다. log를 벗겨내는 방법은 e의 xy제곱을 계산\
   ln100은 약 4.6 이므로 10으로 나눠 정규화. 최대값을 크게 설정하면 그에 맞게 정규화하여 0~1 사이로 만들어줘야한다.
   - 나눗셈\
  곱셈과 같은 방법으로 자연로그를 이용하여 처리. ( ln⁡ 𝑥/𝑦  =  ln⁡ x − ln⁡ y)\
  다만 마이너스 연산이 나오기 때문에 출력데이터에 음수가 나올 수 있어 자연로그를 씌운 후 5(4.6의 올림값)를 더해 값이 음수가 되는 것을 방지\
  최대값이 100이라서 5이지만 최대값이 변경되면 그에 맞게 처리. 그 이후 정규화를 위해 10으로 나눈다.
  

 - 모델 구성
 
      <img width="600px" height = "100px"  src="https://user-images.githubusercontent.com/29851772/208502658-ca15d6af-2ebf-40c8-87fc-b22b34e2e228.png">

 - 결과
  
      <img width="800px" height = "370px"  src="https://user-images.githubusercontent.com/29851772/208503793-ac3a1938-96b4-4005-bb31-72c25bfb1a1d.png">
      <img width="800px" height = "370px"  src="https://user-images.githubusercontent.com/29851772/208504167-6ebb490a-511c-4806-9d04-779c9e3e2671.png">
   
   - 위 사진은 테스트 데이터를 넣었을때 원래 값과 모델 예측 값이고 아래 사진은 학습할때의 loss 값이다. 
   - 덧셈, 뺄셈 보다는 곱셈, 나눗셈에서 조금 더 오차가 있긴 하지만 전체적으로 잘 맞추는 것을 확인 할 수 있다.     
   - MAE는 평균 절대 오차로 원래값-예측값에 절대값을 씌우고 평균을 구한 값이다. 평균적으로 이 정도의 오차가 발생했다고 보면 된다. 

<br>
<h3>X-Ray 이미지 분류</h3>

   <img width="820px" height = "200px"  src="https://user-images.githubusercontent.com/29851772/208507768-0d361da9-03ce-45a7-899f-8de1d2209bdf.png">

- 주제 설명      
\- Kaggle에서 위와 같은 골절 및 비골절 X-ray 이미지 자료 수집하여 8392개의 부위 별 데이터로 분류 후 사용            
\- X-ray 사진의 골절 여부를 간단히 머신러닝으로 판별 가능         
  * 출처: https://www.kaggle.com/datasets/vuppalaadithyasairam/bone-fracture-detection-using-xrays

- 입출력 데이터  

     <img width="800px" height = "350px"  src="https://user-images.githubusercontent.com/29851772/208508626-ec26b704-3eca-4f9b-9981-f3b75ce2c65d.png">
   
   - 처음에 입력되는 데이터는 x-ray 이미지를 로드 후 용량을 줄이기 위해 64x64 픽셀로 resize, 그리고 numpy 배열로 변환 된다.         
     그리고 픽셀 값을 255로 나눠 0~1사이의 값으로 정규화 시킨 다음 입력 데이터로 사용.         
   - 전체 구조를 보면 data는 먼저 관절을 분류해주는 모델에 입력된다.             
     그럼 모델은 x-ray 이미지가 손목 아래쪽 팔 부분인지, 손날 부분인지, 손 부분인지를 판별하여 출력한다.         
     분류되어 출력된 이미지 데이터는 자신이 해당하는 각 부위별로 학습된 모델에 입력된다.                      
     2차 모델은 이 x-ray 이미지가 골절 사진인지, 골절되지 않은 사진인지를 판단하여 출력데이터로 나오게된다.

 - 결과
 
    <img width="650px" height = "420px"  src="https://user-images.githubusercontent.com/29851772/208510224-8291a89b-724e-4db1-aa9a-679bcd89fa47.png">

    <img width="600px" height = "450px"  src="https://user-images.githubusercontent.com/29851772/208511003-3273aad7-8b3b-4c01-ae91-4eceab24453f.png">
    
   - 왼쪽이 부위 판별 모델의 테스트 값이고 3가지 부위로 분류되기 때문에 0 또는 1 또는 2의 label로 분류       
     총 테스트데이터 2098개 중 8개를 틀렸다.       
   - 오른쪽은 골절 판별 모델의 테스트 값이고 골절 혹은 비골절로 분류 되기 때문에 0 또는 1의 label로 분류     
   - 1번 모델에서 8개가 잘못 넘어왔는데 2번 모델을 통과하고 총 틀린 개수는 6개다. 8개보다 더 많아져야 할 오차 개수가 오히려 줄어서 의아할 수 있다.
     하지만 2번 모델은 이진분류이기 때문에 골절인지 비골절인지 운이 좋게 맞췄다면 잘못 넘어간 데이터도 맞은걸로 체크되고 제대로 분류되어 넘어간 데이터도 틀릴 수 있다.
     따라서 앞의 오류 개수와는 상관없이 최종적으로 골절 여부의 오차가 6개라는 뜻이다.

- 최종 정리
  - 출처 링크에 모든 사진데이터를 그대로 입력으로 넣고 분류과정 없이 학습한 결과가 있는데 80% 정도로 수치가 낮았다.                
  - 하지만 단순한 input->output 형태가 아닌 모델을 통해 유형별로 분류 한 output 데이터를 새로운 input 데이터로 사용해                       
    각 유형에 맞게 학습한 모델을 사용해 골절 여부를 판별하게 설계함으로써 정확도를 높일 수 있었다.
















