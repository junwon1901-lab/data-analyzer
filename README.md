# 📊 데이터 분석 프로그램 (Data Analyzer)

어떤 CSV·Excel 파일이든 불러와서 X/Y축을 고르면 **선형회귀와 비선형회귀** 결과를 보여주는 Streamlit 프로그램입니다.
앞으로 인공신경망, 대화형 AI 기능을 추가하며 확장할 예정입니다.

## 기능
- **파일 불러오기**: CSV, Excel(.xlsx/.xls), 텍스트(.txt/.data)
  - 구분자(쉼표·세미콜론·탭), 한글 인코딩, 천 단위 쉼표, 소수점 쉼표 자동 인식
  - Excel 시트 선택, 열 이름 행 위치 지정 (표 위에 제목이 있는 파일 대응)
- **X/Y축 선택**: 숫자로 된 열 자동 탐지
- **데이터 거르기**: 글자 열(분류)로 원하는 그룹만 선택, X값 범위 지정
- **선형회귀**: y = a·x + b
- **비선형회귀**: 다항식(2~6차), 지수, 거듭제곱, 로그
- **결과**: 회귀식, R², RMSE, MAE, 잔차 그래프, 새 X값 예측
- **모델 비교**: 전체 모델 R² 순위표, CSV 다운로드

## 실행 방법
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 폴더 구조
```
app.py              메인 화면 (프로그램 이름은 APP_NAME 변수에서 변경)
core/data_loader.py 파일 불러오기, 숫자 변환
core/regression.py  회귀 모델과 평가지표
models/             (다음 과제) 인공신경망 모델 추가 예정
data/               예제 데이터
```

## 예제 데이터
- `data/example_hull_resistance.csv`: [Sailboat Hull Resistance Dataset](https://data.mendeley.com/datasets/gw23dgzn6h)
  (Caprace & Faller Fahrnholz, Mendeley Data)에서 시리즈·선형번호·프루드수·전저항 4개 열을 뽑아 한글로 정리 (1018행, 70척)
- `data/example_hull_resistance_full.csv`: 같은 데이터의 21개 열 전체 (인공신경망 과제용)

## 개발 계획
- [x] 파일 불러오기, 축 선택, 선형·비선형 회귀
- [ ] 인공신경망 회귀
- [ ] 대화형 AI 분석 도우미
