# 기존 GitHub 저장소에 적용하기

대상: https://github.com/junwon1901-lab/data-analyzer

기준 코드: 커밋 adfafff0bac23ffd95da4d7baf0e91395d59f423.
이번 작업은 로컬 구현·검증까지 완료했으며, GitHub 원격 저장소·Streamlit 앱·Notion·LMS는 수정하지 않았습니다.

2026-10-03 보강판입니다. 10월 2일 녹취의 입력·출력 형태, 은닉층 설계·테스트, 가중치·편향 학습 설명을 반영했습니다.
이 ZIP은 이전 신경망 업데이트의 전체 파일을 포함하므로 이전 ZIP을 먼저 적용하지 않아도 됩니다.

## 1. 변경 파일

교체: app.py, requirements.txt, README.md

추가: models/neural_network.py, ui/__init__.py, ui/ann_page.py,
tests/test_neural_network.py, tests/test_app.py, scripts/verify_ann.py, scripts/compare_ann.py,
requirements-dev.txt, requirements-tested.txt, .gitignore, docs/ann_assignment.md, docs/APPLY_UPDATE.md.
실제 보강 실험 기록: docs/ann_experiment/ 내 CSV 5개와 experiment_report.json.

기존 core/*.py, models/__init__.py, data/*.csv는 그대로 유지합니다.
업데이트 ZIP에는 기존 데이터와 기존 core 모듈이 없으므로 ZIP만 단독 실행하지 말고,
기존 저장소에 위 경로를 유지해 적용해야 합니다. 파일을 models 또는 ui 폴더 밖에 올리면 실행되지 않습니다.

## 2. 안전하게 반영하는 방법

1. GitHub 저장소에서 작업용 새 브랜치(예: feature/ann-assignment)를 만듭니다.
2. 업데이트 ZIP을 풀고, 저장소 루트 기준으로 파일 경로를 유지해 반영합니다.
3. 기존 core와 data 파일이 남아 있고 requirements.txt도 함께 갱신됐는지 확인합니다.
4. 새 환경에서 아래 명령으로 테스트합니다.

~~~bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/compare_ann.py
streamlit run app.py
~~~

5. 기존 회귀 모드와 새 인공신경망 모드를 모두 확인한 뒤 변경을 main에 병합합니다.
   이번 기준 커밋 이후 원격 코드가 바뀌었다면 기존 수정사항과 비교하고 병합해야 합니다.

이 설명서는 명령 안내일 뿐, 브랜치 생성·업로드·병합을 대신 수행했다는 뜻은 아닙니다.
태블릿에서 폴더 업로드가 어렵다면 컴퓨터에서 GitHub 웹 업로드 또는 Git 클라이언트를 이용하세요.

## 3. Streamlit 적용 확인

기존 앱 주소: https://data-analyzer-farcyhew6tpc7bdizm6rvw.streamlit.app/

기존 앱이 이 저장소의 main 브랜치와 app.py에 연결돼 있다면 해당 변경의 배포 상태를 확인합니다.
실제 연결 브랜치·배포 성공 여부는 이번 작업에서 확인하지 않았습니다.
배포 화면에서 의존성 설치와 실행 로그를 확인하고, 필요하면 재시작 후 테스트하세요.

반드시 확인할 동작:

- 1차 과제 회귀 모드에서 원래 예제와 그래프가 열린다.
- 2차 과제 모드에서 전체 변수 예제를 선택하고 학습할 수 있다.
- X/y 형태에서 관측 수와 입력 노드 수를 구분할 수 있다.
- ‘구조 비교 실험 실행’으로 네 후보를 비교하고 ‘비교에서 선택한 구조 적용’으로 선택할 수 있다.
- 학습률·데이터가 바뀌면 비교 결과와 이전 학습 결과가 숨겨진다.
- 실제 학습된 가중치·편향과 첫 은닉 노드 계산이 표시된다.
- 테스트 R²·RMSE·MAE, 학습 곡선, 선형 기준 모델이 표시된다.
- 새 입력 예측과 CSV·JSON 다운로드가 동작한다.

## 4. Notion 제출

1. 1차 과제 설명을 지우지 말고, 2차 과제용 하위 페이지나 별도 섹션을 만듭니다.
2. docs/ann_assignment.md 내용을 옮기거나 Markdown 파일로 가져옵니다.
3. 배포된 앱에서 직접 실행한 결과와 화면 캡처를 추가합니다.
   구조 비교 표, X/y 형태, 노드 계산, 학습 곡선·테스트 결과를 캡처하면 수업의 강조점을 설명하기 좋습니다.
4. 설명서 맨 위의 ‘온라인 반영 전’ 안내를 실제 상태에 맞게 수정합니다.
5. 로그인 없이 실행 앱·GitHub·Notion이 열리는지 확인하고 LMS에 Notion URL을 제출합니다.

새 과제 공지 사진에는 마감일시가 설정돼 있지 않았습니다. 별도 수업 안내 여부는 확인해야 합니다.
이전 대화에서 확인한 공지 상태이며 이번에 LMS를 다시 조회한 것은 아닙니다.
10월 2일 녹취에서도 2차 과제의 정확한 마감은 새로 확인되지 않았습니다. 42:29의 밤 12시는 1차 과제 설명 맥락이고,
9월 30일 55:00의 ‘한 2주 정도’를 특정 날짜의 확정 마감으로 바꾸지 않습니다.
프로그램 구현 완료와 과제 제출 완료는 별개입니다.
