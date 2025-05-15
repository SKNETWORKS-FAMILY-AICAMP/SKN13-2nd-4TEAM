# [SKN13-2nd-4TEAM] 서울특별시 폐업사업자 분류 예측 

## 팀 소개

* 

## 프로젝트 소개

* 기간: 2025.05.15. - 2025.05.16.

* 내용: 서울특별시 상권, 업종 별 폐업 분류 예측 및 솔루션 제공 App 개발

* 기대 효과: 폐업에 영향을 주는 요인 분석, 창업 지원 및 자영업 지원 사업에 인사이트 제공

## 개요

![image](https://github.com/user-attachments/assets/877f75f7-a7a9-44e6-88b4-0472f9994382)
<p align="center"><b>[그림 1]</b> 국세통계포털: 폐업자 현황</p>


### 전국 폐업사업자 급증

* 2023년 폐업사업자: 약 **98만 6천 명**.

* 글로벌 금융위기 당시(2008년, 2009년: 각각 84만 명), 코로나 시기보다 심각

* 경기도: 2024년 11월부터 **폐업 > 개업**. 올해 1분기 폐업률은 근 6년 중 최고치(2.85%)

### 원인은?

* **사업부진**

* 업계 전반의 **비용 구조 악화**

* **새로운 소비 행태** 등장(e.g. YONO, You Only Need One)


### 결국 **💸돈💸!!**

: 특정 상권, 특정 업종의 데이터들 중 **매출액** 기반으로 폐업을 예측해볼 수 있지 않을까?
 
<h3 align="center"><b>폐업 분류 예측 모델 및 솔루션 제공 App 개발!!</b></h3>

## 연구 방법

1. 탐색적 데이터 분석
2. 데이터 전처리
3. 분류 모델 학습
4. 성능 평가
5. 결론 도출 및 App 개발

## 기술 스택

### 프로그래밍 언어

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

### IDE

![VS Code](https://img.shields.io/badge/VSCode-007ACC?style=flat&logo=visualstudiocode&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

### 데이터 정제 및 전처리 라이브러리

![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

### 데이터 시각화 라이브러리

![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-377BA8?style=flat&logo=seaborn&logoColor=white)

### ML/DL 라이브러리

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC6C00?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-027B77?style=flat)
![CatBoost](https://img.shields.io/badge/CatBoost-FFA500?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)

### 성능 평가 라이브러리

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

### App

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

### 형상 관리 및 협업

![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)
![Discord](https://img.shields.io/badge/Discord-5865F2?style=flat&logo=discord&logoColor=white)

## 데이터

### 데이터 소개

#### 1. `data/expected_sales/*.csv`

[서울특별시: 서울시 상권분석서비스(추정매출-상권)](https://data.seoul.go.kr/dataList/OA-15572/S/1/datasetView.do#)

: 서울시 상권 내 점포들의 평균 매출 추정치

* Total Shape: (512626, 55)

* Columns

  * `기준_년분기_코드(int64)`: 다섯 자리 정수. 앞의 네 자리는 연도, 뒤의 한 자리는 분기.

    e.g. 20234: 2023년 4분기

  * `상권_구분_코드(object)` & `상권_구분_코드_명(object)`: 상권 대분류

    e.g. 'D' & '발달상권'

  * `상권_코드(int64)` & `상권_코드_명(object)`: 상권 소분류

    e.g. 3110695 & '개봉1동주민센터'

  * `서비스_업종_코드(object)` & `서비스_업종_코드_명(object)`: 업종 분류

    e.g. 'CS100004' & '양식음식점'

  * `당월_매출_금액(int64)`

  * `주중_매출_금액(int64)` & `주중_매출_금액_건수(int64)`
 
  * `주말_매출_금액(int64)` & `주말_매출_금액_건수(int64)`

  * `N요일_매출_금액(int64)` & `N요일_매출_금액_건수(int64)`

  * `시간대_XX~XX_매출_금액(int64)` & `시간대_XX~XX_매출_금액_건수(int64)`

    > 00~06시, 06~11시, 11~14시, 14~17시, 17~21시, 21~24시

  * `남성_매출_금액(int64)` & `남성_매출_금액_건수(int64)`

  * `여성_매출_금액(int64)` & `여성_매출_금액_건수(int64)`

  * `연령대_XX_매출_금액(int64)` & `연령대_XX_매출_금액_건수(int64)`

    > 10, 20, 30, 40, 50, 60_이상

  * 모든 금액의 단위는 `원`입니다.

#### 2. `data/the_number_of_shops/*.csv`

[서울특별시: 서울시 상권분석서비스(점포-상권)](https://data.seoul.go.kr/dataList/OA-15577/S/1/datasetView.do#)

: 서울시 상권 내 점포 정보(개폐업 점포 수, 프랜차이즈 점포 수 등)

* Total Shape: (1831925, 14)

* Columns

  * `기준_년분기_코드(int64)`: 다섯 자리 정수. 앞의 네 자리는 연도, 뒤의 한 자리는 분기.

    e.g. 20234: 2023년 4분기

  * `상권_구분_코드(object)` & `상권_구분_코드_명(object)`: 상권 대분류

    e.g. 'D' & '발달상권'

  * `상권_코드(int64)` & `상권_코드_명(object)`: 상권 소분류

    e.g. 3110695 & '개봉1동주민센터'

  * `서비스_업종_코드(object)` & `서비스_업종_코드_명(object)`: 업종 분류

    e.g. 'CS100004' & '양식음식점'

  * `점포_수(int64)`

  * `유사_업종_점포_수(int64)`: `점포_수` + `프랜차이즈_점포_수`

  * `개업_율(int64)`: `개업_점포_수` / `유사_업종_점포_수`

  * `개업_점포_수(int64)`

  * `폐업_률(int64)`: `폐업_점포_수` / `유사_업종_점포_수`

  * `폐업_점포_수(int64)`

  * `프랜차이즈_점포_수(int64)`

#### 3. data/datasets/*.csv

* `X_train`, `X_test`, `y_train`, `y_test`

  * 탐색적 데이터 분석 이후의 학습&평가 데이터

  * Shape: (318295, 33), (82864, 33), (318295,), (82864,)

### 탐색적 데이터 분석

#### 1. 테이블 병합(Join)

```python
df1 = pd.concat(pd.read_csv(i) for i in expected_sales)       # 추정 매출
df2 = pd.concat(pd.read_csv(i) for i in the_number_of_shops)  # 점포

df = df1.merge(
    df2,
    how = 'inner',
    on = ['기준_년분기_코드', '상권_구분_코드_명', '상권_코드_명', '서비스_업종_코드_명'] # Key
)
```

#### 2. 데이터 정제

* 파생변수 설정

  * `당월매출증감`: 당년 당월 매출액 $-$ 전년 당월 매출액

  * `유사업종점포수증감`: 당년 유사업종점포수 $-$ 전년 유사업종점포수

* 중복되거나 불필요한 행&열 제거

  * e.g. 매출액과 매출건수 중 매출액을 채택, 상권코드와 상권코드명 중 상권코드명을 채택

* 이상치 및 결측치 처리

  * 폐업률이 1보다 큰 경우는 이상치 -> 제거

  * 병합 과정에서 생기는 결측치 -> 제거

* 상관분석

  * 요일별 매출, 시간대별 매출, 연령대별 매출끼리 매우 높은 양의 상관관계를 가짐

  * `유사업종점포수증감`과 `당월매출금액증감`은 Target과 약한 음의 상관관계 존재

* 클래스 분포

  * 폐업률 기준 이진 분류 시 비대칭적 데이터 → 클래스 불균형 존재 (정상:폐업 ≒ 4:1)

### 데이터 전처리

#### 1. Encoding & Scaling

* `상권코드명`, `상권구분코드명`, `서비스업종코드명`은 **Label Encoding**

* `Target(폐업률)`은 평균(0.03)을 기준으로 Positive(1)과 Negative(0)로 Labeling

* **NO Scaling**(Tree-based Models)

### 모델 학습

#### Baseline Models



## References

[1] 경총 보고
[2] https://biz.heraldcorp.com/article/10481988?ref=naver
