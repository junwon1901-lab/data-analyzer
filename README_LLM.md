# 무료 CPU 버전

현재 설치·배포 방법은 README_FREE.md를 따른다. 예전 Ollama·Docker 배포 방법은 기본 실행 경로가 아니다.

llm_lab/cloud_client.py: 모델 다운로드와 CPU 추론
llm_lab/client.py: 명령 스키마와 검증
llm_lab/agent.py: 허용된 분석 함수 호출
llm_lab/analysis.py: 실제 수치 계산
llm_lab/session.py: 대화·결과 JSON 저장
llm_lab/ui.py: Streamlit 화면
