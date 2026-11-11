# 데이터 분석 프로그램 — 1·2차 과제 보존판

동아대학교 조선해양공학과 서준원 · 데이터처리개론

선형·비선형 회귀분석(1차 과제)과 인공신경망(2차 과제)을 기존 Streamlit 화면으로 사용하는 별도 버전이다. `assignments-1-2` 브랜치에서 관리한다.

| 버전 | 배포 기준 | 용도 |
|---|---|---|
| 1·2차 과제 보존판 | `assignments-1-2` / `app.py` | 기존 파일 업로드·회귀 탭·신경망 학습 화면 |
| 마지막 과제 Data Lab | `main` / `app.py` | 대화로 분석하는 오픈소스 LLM 앱 |

[현재 Data Lab](https://data-analyzer-farcyhew6tpc7bdizm6rvw.streamlit.app/)은 계속 운영한다. 이 보존판은 [회귀·인공신경망 전용 사이트](https://junwon-data-analyzer-classic.streamlit.app/)에서 별도로 실행된다. 두 앱은 서로 다른 브랜치로 배포하므로 한쪽을 수정해도 다른 쪽의 코드가 바뀌지 않는다.

**바로 실행:** [https://junwon-data-analyzer-classic.streamlit.app/](https://junwon-data-analyzer-classic.streamlit.app/)

**2차 과제 바로 실행:** [인공신경망 모드](https://junwon-data-analyzer-classic.streamlit.app/?mode=ann)

## 주요 기능

- CSV·Excel·텍스트 파일 업로드, 예제 데이터 선택, 입력 데이터 필터
- 선형회귀와 비선형회귀(다항·지수·거듭제곱·로그), 회귀식·잔차·모델 비교
- 여러 입력변수 선택, 신경망 구조·학습률·epoch·분리 방식 선택
- 학습·검증·테스트 평가, 학습 곡선, 구조 비교, 새 입력값 예측
- 결과 CSV·JSON 다운로드

## 설치와 실행

Python 3.14.10에서 의존성 설치와 실행을 검사했다.

```bash
git clone --branch assignments-1-2 https://github.com/junwon1901-lab/data-analyzer.git data-analyzer-classic
cd data-analyzer-classic
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

왼쪽 **분석 모드**에서 회귀분석 또는 인공신경망을 선택한다. 회귀분석은 예제 데이터와 X/Y축을 선택한 뒤 네 개 탭으로 결과를 확인한다. 인공신경망은 기본 전체 변수 예제를 사용해 입력·목표·분리 방식·구조를 확인한 다음 학습한다. 새 입력을 바꾸고 예측하면 이전 입력과 비교할 수 있다.

## 별도 Streamlit 배포

새 앱 만들기에서 다음을 지정한다. 기존 Data Lab 앱의 배포 설정은 변경하지 않는다.

| 항목 | 값 |
|---|---|
| Repository | `junwon1901-lab/data-analyzer` |
| Branch | `assignments-1-2` |
| Main file path | `app.py` |
| Python | `3.14` |

## 계산 원리·해석

선형·다항 회귀 등은 잔차제곱합을 줄이는 계수를 구한다. 신경망은 입력을 표준화한 뒤 가중합과 비선형 활성화를 거쳐 연속적인 목표값을 예측한다. 라이브러리의 MLPRegressor와 Adam을 사용하며 역전파 전체를 직접 구현한 것은 아니다.

기본 신경망 데이터는 선형번호로 분리해 같은 선박의 행이 여러 세트에 섞이지 않게 한다. 표준화 기준은 학습 데이터에서만 구한다. 회귀 탭의 적합도는 맞춘 데이터 자체의 지표이므로 신경망 테스트 점수와 직접 비교하지 않는다.

기본 데이터의 목표는 정규화 저항 Rt/Δ × 10³이며 힘의 단위 N이 아니다. R²는 정답률이 아니며, 음수 저항과 입력 범위 밖의 예측은 물리적 의미와 불확실성을 확인해야 한다.

## 데이터 출처와 기록

- Caprace & Faller Fahrnholz (2022), Sailboat Hull Resistance Dataset and Predictive Model, v4, CC BY 4.0: https://doi.org/10.17632/gw23dgzn6h.4
- `data/example_hull_resistance.csv`: 회귀용 한글 정리본
- `data/example_hull_resistance_full.csv`: 전체 변수 정리본(1,018행, 21열, 70개 선형)
- [기존 신경망 설명](docs/ann_assignment.md), [기존 실험 기록](docs/ann_experiment)
- [보존판 복원·검증 기록](docs/RESTORATION.md)

분리를 위해 AI를 활용해 배포용 파일과 문서를 정리했다. `app.py`는 기존 `legacy_app.py` 화면을 복원하고 과제별 링크에 따른 초기 모드 선택을 추가했다. 회귀·신경망 계산 모듈은 변경하지 않았다. 실제 배포 성공 여부와 개발 환경 검증은 구분해 기록한다.

## 설명할 수 있어야 하는 질문

1. 회귀식과 신경망은 각각 무엇을 입력받고 예측하는가?
2. 같은 선박의 행을 왜 묶어서 분리하는가?
3. 학습·검증·테스트의 역할은 어떻게 다른가?
4. R²가 높아도 새 예측을 그대로 믿을 수 없는 이유는 무엇인가?
