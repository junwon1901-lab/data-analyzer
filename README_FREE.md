# 마지막 과제: 무료 Streamlit 배포용

기존 회귀·신경망 프로그램에 **LLM Chat**을 추가했다. 기본 구성은 **Qwen2.5-1.5B-Instruct Q4_K_M + llama-cpp-python + Streamlit**이다. 언어모델을 같은 서버 CPU에서 직접 실행하며, 유료 API 키나 내 PC의 Ollama 연결이 필요 없다.

**현재 상태:** 개발 환경 실행 검사 결과는 `docs/FREE_VERIFICATION.md`에 있다. GitHub 업로드·실제 Streamlit 공개 배포·LMS 제출은 아직 확인하지 않았다. 무료 서버에서의 실행 성공을 보장하는 문서가 아니다. 사용자 이름을 회상하는 질문에 오답이 남아 있으므로 해당 기능을 성공했다고 쓰지 않는다.

## 1. 새 ZIP 압축 풀기

1. PC에 ZIP을 내려받아 새 위치에 압축을 푼다.
2. `data-analyzer-llm` 폴더를 연다.
3. `app.py`, `requirements.txt`, `llm_lab`, `pages`, `data`가 보이면 맞는 위치다.

이번 ZIP은 기존 프로그램과 합쳐 놓은 완성 프로젝트다. 이전 `SETUP_PROJECT.py`와 `overlay` 방식은 사용하지 않는다. 모델 가중치는 ZIP에 넣지 않았다. 처음 실행하면 서버가 약 1.12GB를 다운로드한다.

## 2. 기존 GitHub에 올리기

저장소: https://github.com/junwon1901-lab/data-analyzer

1. 로그인하고 `main` 브랜치를 연다.
2. **Add file → Upload files**를 누른다.
3. `data-analyzer-llm` 폴더 자체가 아니라 **그 안의 파일·폴더**를 올린다. 같은 경로의 기존 파일은 새 내용으로 갱신된다.
4. 변경 설명에 `Add CPU LLM chat for final assignment`를 입력하고 **Commit changes**를 누른다.
5. 저장소 최상위에 `requirements.txt`, `packages.txt`, `llm_chat.py`가 있고, `llm_lab/cloud_client.py`, `pages/3_LLM_Chat.py`가 있는지 확인한다.

기존 공개 코드 commit `ec95689cad9ec0ad976ed247a450416d8940a2eb`을 기준으로 만들었다. 그 뒤 본인이 따로 수정한 코드가 있다면 먼저 비교해 보존한다. 기존 데이터와 분석 화면은 삭제하지 않는다.

`.models`, `.llm_sessions`, `.venv`, `__pycache__`, `.gguf`, 비밀번호·API 키는 올리지 않는다. 제공 ZIP에는 이런 실행 파일을 넣지 않았다. 숨김 `.gitignore`를 올리기 어려우면 `GITIGNORE_LLM.txt` 내용을 GitHub의 `.gitignore`에 추가한다.

## 3. 기존 Streamlit 앱 확인

기존 주소: https://data-analyzer-farcyhew6tpc7bdizm6rvw.streamlit.app/

1. GitHub 변경 후 Streamlit이 다시 빌드할 때까지 기다린다. **Manage app**에서 설치 로그를 확인한다.
2. 연결된 저장소는 `junwon1901-lab/data-analyzer`, 브랜치는 `main`, 시작 파일은 `app.py`인지 확인한다.
3. 검증 환경은 **Python 3.12, Linux x86_64 CPU**다. 새 앱을 만들 경우 Advanced settings에서 Python 3.12를 선택한다. 기존 앱은 먼저 현재 버전과 설치 로그를 확인하고, 곧바로 삭제하지 않는다.
4. 왼쪽 **LLM Chat**을 누른다.
5. **모델 준비 / 상태 확인**을 누른다. 첫 다운로드는 약 1.12GB이며 수 분 걸릴 수 있다.
6. **모델 준비 완료**가 표시된 뒤 아래 문장을 실제로 입력한다.

성공적으로 배포되면 내 PC를 꺼도 서버가 모델을 실행한다. 무료 앱은 절전·재시작·자원 제한으로 첫 응답이 늦거나 실행이 중단될 수 있다. 코드 작성 완료와 공개 서버 배포 완료를 구분해야 한다.

## 4. 직접 시험할 문장

| 입력 | 확인할 내용 |
|---|---|
| 안녕! 내 이름은 준원이야. 짧게 인사해줘. | 실제 한국어 생성 답변 |
| 선체 저항 데이터를 불러와서 설명해줘. | 1,018행·21열·선형 70개 |
| 선형회귀 해줘. | 실제 결과표·산점도 |
| 비선형회귀 해줘. | 2차 다항회귀 결과 |
| 신경망을 학습해줘. | 신경망 결과·학습 곡선 |
| 세 모델을 같은 조건으로 비교해줘. | 동일 분할의 세 모델 결과 |
| 50 epoch와 500 epoch를 비교해줘. | 상한·실행·채택 횟수와 검증 RMSE |
| 오차가 큰 예측 10개 보여줘. | 개별 큰 오차 표 |
| 보고서 만들어줘. | 실제 결과 Markdown·JSON 다운로드 |
| 신경망이 뭔지 짧게 설명해줘. 학습은 실행하지 마. | 설명만 하고 추가 학습하지 않음 |
| 이전 분석 결과를 다시 보여줘. | 저장 결과 읽기 |

새 입력 예측:

```text
Fn=0.1, Rn=9870390, Cp=0.5674, Cb=0.3667, Cw=0.6908, L_B=3.6232, B_T=3.043, LCB_L=-0.023으로 저항을 예측해줘.
```

이 입력은 원자료를 활용한 동작 확인용이며 독립적인 정확도 검증 자료가 아니다.

**알려진 한계:** `내 이름이 뭐라고 했지?`에서 사용자와 도우미의 이름을 혼동하는 오답이 관찰됐다. 이전 분석 수치를 파일에서 복원하는 기능과 LLM의 자유로운 개인 이름 회상은 서로 다르다. 작은 모델의 답변을 그대로 사실로 믿지 않는다.

## 5. 증거 저장과 제출

1. 일반 대화, 회귀 결과, epoch 비교, 큰 오차 표를 직접 캡처한다.
2. 대화 JSON과 실제 분석 보고서를 다운로드한다. 무료 서버 디스크는 영구 저장소가 아니며 재배포·절전 복구 시 사라질 수 있다. JSON 재업로드 기능은 없다.
3. 로그아웃/시크릿 창과 다른 기기에서 실제 질문을 입력한다.
4. Notion에 `docs/llm_assignment.md`를 붙이고, 직접 확인한 시각·캡처·GitHub·작동하는 앱 링크를 추가한다.
5. 개인 `?session=...` 쿼리는 빼고 기본 앱 주소를 LMS에 제출한다. 제출 완료 표시까지 확인한다.

공지 기준 마감은 **2026.11.12 23:59 (한국시간)**, 제출 형태는 웹사이트 URL이다. 11/6 강의의 다음 금요일 검토 발언을 마감 연장으로 해석하지 않는다.

강의 근거: `2026. 11. 6. 데이터처리개론.txt` 39:13~40:52. 오픈소스 LLM 선택·다운로드·설치 후 자신의 웹사이트에 대화창을 붙이는 과제다. 특정 유료 서버나 DeepSeek은 필수로 지정하지 않았다. 원본 음성이 아닌 녹취 텍스트를 확인했다.

## 작동 원리

- **Qwen**: 사전학습된 언어모델. 이번 과제에서는 설치 후 추론하며 LLM 가중치를 새로 학습하지 않는다.
- **GGUF / Q4_K_M**: 실행용 모델 파일 형식 / 메모리를 줄이는 양자화 방식.
- **llama-cpp-python**: 모델을 서버 CPU에서 실행하는 도구.
- **Streamlit**: 대화·결과표·그래프·다운로드 화면.
- **Python 분석 함수**: 실제 회귀·신경망 계산. LLM이 만든 임의 코드를 실행하지 않는다.

동작 분류에는 현재 요청과 간단한 분석 상태만 사용한다. 일반 대화에는 최근 최대 6개 메시지(각 240자)를 별도로 넣는다. 문맥 4,096토큰, 출력 최대 256토큰으로 제한한다. 모델 하나를 공유하고 추론은 한 요청씩 실행한다. 다른 요청 처리 중에는 재시도 안내를 표시한다.

Hugging Face는 공개 가중치 다운로드에만 사용한다. 질문을 유료 Inference API로 전송하지 않는다. 모델 파일의 고정 revision과 SHA-256을 검사한다.

## 실행·오류 해결

Linux CPU에서 직접 재현:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

실제 모델 검사: `python scripts/verify_llm.py --output docs/my_verification`

화면 검사: `python scripts/verify_cloud_ui.py`

자동 검사: `pip install pytest` 후 `python -m pytest tests -q`.

| 증상 | 먼저 확인 |
|---|---|
| LLM Chat 메뉴 없음 | pages/3_LLM_Chat.py 위치와 연결 브랜치 |
| llama_cpp 없음 | requirements.txt 반영 및 설치 로그 |
| 모델 다운로드 실패 | 네트워크 오류 확인 후 준비 버튼 재시도 |
| 메모리 제한 | Manage app 로그 확인. 유료 업그레이드를 먼저 누르지 않음 |
| 답변이 늦음 | 첫 다운로드인지 추론인지 구분하고 연속 요청하지 않음 |
| 잘못된 답변·명령 | 실행 기록 action 확인, 짧고 구체적으로 재질문 |

이번 배포 파일은 Linux CPU용이다. Windows/Mac에서 같은 wheel이 동작한다고 주장하지 않는다. iPad에서는 배포된 웹 화면을 사용할 수 있지만 위 명령은 iPad용 설치 명령이 아니다.

## 내가 설명할 다섯 질문

1. Qwen, GGUF, llama-cpp-python, Streamlit은 각각 무엇을 하는가?
2. 일반 LLM 답변과 Python이 계산한 수치를 왜 구분하는가?
3. 대화·결과 저장은 왜 모델 재학습이나 영구 기억과 다른가?
4. epoch 상한 500·실행 157·채택 117은 어떻게 다른가?
5. 높은 R²와 실제값의 절반을 예측하는 큰 오차가 함께 나타날 수 있는 이유는?

## 출처

- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF (Apache 2.0)
- https://github.com/abetlen/llama-cpp-python
- https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://doi.org/10.17632/gw23dgzn6h.4 (자료 V4, CC BY 4.0)
- 강의: 2026. 11. 6. 데이터처리개론.txt 28:48~35:45, 39:13~40:52.

공식 무료 서버의 자원 수치는 변경 가능하다. 로컬 측정은 실제 계정의 RAM 할당·속도·동시 접속을 보장하지 않는다.
