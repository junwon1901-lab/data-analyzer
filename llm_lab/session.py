"""대화·계산 결과를 JSON으로 저장. 가중치 학습이나 벡터 검색 기반 RAG가 아니다."""
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone(timedelta(hours=9))).isoformat(timespec='seconds')


def new_session():
    return {'schema_version': 1, 'id': uuid.uuid4().hex, 'created_at': now(),
            'messages': [], 'runs': [], 'events': []}


def session_dir():
    return Path(os.environ.get('LLM_SESSION_DIR', str(ROOT / '.llm_sessions')))


def valid_id(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{32}', value) is not None


def save(session):
    if not valid_id(session.get('id')):
        raise ValueError('잘못된 대화 식별자')
    directory = session_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (session['id']+'.json')
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=directory, delete=False) as f:
        json.dump(session, f, ensure_ascii=False, allow_nan=False)
        temporary = f.name
    os.replace(temporary, target)


def load(session_id):
    if not valid_id(session_id):
        raise ValueError('잘못된 대화 식별자')
    path = session_dir() / (session_id+'.json')
    if not path.exists():
        session = new_session()
        session['id'] = session_id
        return session
    session = json.loads(path.read_text(encoding='utf-8'))
    if session.get('schema_version') != 1 or session.get('id') != session_id:
        raise ValueError('대화 기록 형식이 맞지 않습니다.')
    return session


def context(session):
    runs = session['runs']
    return {'saved_runs': len(runs), 'last_run': summary(runs[-1]) if runs else None,
            'earlier_actions': [e.get('action') for e in session['events'][-20:]]}


def summary(run):
    return {'action': run['action'], 'selected': run['selected'], 'time': run.get('time'),
            'metrics': [m for r in run['models'] for m in r['metrics'] if m['split']!='학습']}


def report(session):
    lines = ['# LLM 선체 저항 분석 기록', '', f'생성 시각: {now()} (한국시간)', '',
             '이 문서는 저장된 실제 계산 결과로 자동 작성되었다. LLM의 일반 대화가 과학적 검증을 대신하지 않는다.', '',
             '## 구성', 'Qwen2.5 + llama-cpp-python (CPU) → 구조화된 명령 → 기존 Python 회귀·신경망 함수 → 그래프·결과 → JSON 기록', '',
             '## 분석 결과']
    if not session['runs']:
        lines += ['분석 실행 기록이 없다. 성능을 계산했다고 주장할 수 없다.']
    for i, run in enumerate(session['runs'],1):
        lines += ['', f"### 실행 {i}: {run.get('time','')} / {run['action']}",
                  f"선택 기준: {run['selection_basis']}; 선택 모델: {run['selected']}"]
        for r in run['models']:
            lines += ['', f"데이터: {r['data']['source']} · {r['data']['rows']}행 · 입력 {', '.join(r['data']['features'])} · 목표 {r['data']['target']}"]
            if r.get('equation'):
                lines += ['', f"적합식: `{r['equation']}`"]
            lines += ['', '| 모델 | 데이터 | R² | RMSE | MAE | 행 수 |', '|---|---|---:|---:|---:|---:|']
            for m in r['metrics']:
                lines.append(f"| {m['model']} | {m['split']} | {m['R2']:.6f} | {m['RMSE']:.6f} | {m['MAE']:.6f} | {m['rows']} |")
            lines += ['', f"설정 ({r['name']}): `{json.dumps(r['config'], ensure_ascii=False)}`"]
        if run.get('epoch_candidates'):
            lines += ['', '#### 11/6 피드백: epoch 상한 비교',
                      '| 상한 | 실행 | 채택 | 검증 RMSE |', '|---:|---:|---:|---:|']
            for row in run['epoch_candidates']:
                lines.append(f"| {row['max_epochs']} | {row['executed_epochs']} | {row['best_epoch']} | {row['validation_RMSE']:.6f} |")
        from llm_lab.analysis import worst_errors
        lines += ['', f"#### 절대오차가 큰 {run.get('evaluation_label', '테스트')} 예측 10개 (선택 모델)",
                  '| 원자료 행 | 선형번호 | 실제 | 예측 | 절대오차 | 예측/실제 |', '|---:|---|---:|---:|---:|---:|']
        for point in worst_errors(run):
            ratio = point['predicted_actual_ratio']
            ratio_text = f'{ratio:.4f}' if ratio is not None else '정의 안 됨'
            lines.append(f"| {point['row']} | {point.get('hull','미기록')} | {point['actual']:.4f} | {point['predicted']:.4f} | {point['absolute_error']:.4f} | {ratio_text} |")
        lines += ['', run['note']]
    lines += ['', '## 실행 로그', '', '| 시각 | 모델 | 명령 | LLM 응답 시간(초) |', '|---|---|---|---:|']
    for e in session['events']:
        lines.append(f"| {e.get('time','')} | {e.get('model','')} | {e.get('action','')} | {e.get('seconds','')} |")
    lines += ['', '## 해석과 한계',
              '- 기본 데이터는 선형번호로 분리하고, 사용자 입력 데이터는 독립 표본을 가정해 무작위 분리한다.',
              '- 지표는 목표값의 원래 척도다. 기본 데이터는 정규화 저항, 사용자 데이터는 입력한 목표 열의 단위다.',
              '- R²는 맞힌 비율이 아니다. 전체 지표와 함께 큰 개별 오차를 확인해야 한다.',
              '- 2·3차 다항식은 입력에 비선형이지만 계수에 선형인 회귀이며, Ridge 규제값은 1이다.',
              '- LLM은 계산기를 호출하는 요청 해석기다. 부정확한 설명이나 잘못된 동작 선택이 가능하다.',
              '- 동작 선택은 현재 요청으로 수행하고 일반 대화는 최근 최대 6개 메시지(각 240자)를 다시 입력한다. 모든 과거 내용을 무제한으로 기억하지 않는다.',
              '- JSON에 기록을 저장할 뿐 LLM 가중치는 재학습하지 않는다.',
              '- 서버 실행 기록이며 GitHub·Notion 게시 또는 LMS 제출 증명은 아니다.', '',
              '## 출처',
              '- 데이터: Jean-David Caprace, Stefan Faller Fahrnholz (2022), Sailboat Hull Resistance Dataset and Predictive Model, Version 4, CC BY 4.0. https://doi.org/10.17632/gw23dgzn6h.4',
              '- 원자료의 한글 열 이름 정리본: https://github.com/junwon1901-lab/data-analyzer',
              '- 모델: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF',
              '- 실행 도구: https://github.com/abetlen/llama-cpp-python', '']
    return '\n'.join(lines)

