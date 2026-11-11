"""Chat data commands are parsed without asking an LLM to rewrite numbers."""
import csv
import io
import re
import numpy as np
from llm_lab import analysis


def parse_dataset(text):
    rows = list(csv.reader(io.StringIO(text.strip()), delimiter='\t' if '\t' in text.splitlines()[0] else ','))
    if not rows or not 2 <= len(rows[0]) <= 9:
        raise ValueError('첫 행에 입력 열 1~8개와 마지막 목표 열을 적으세요. CSV 또는 탭 구분 표를 사용하세요.')
    columns = [c.strip() for c in rows[0]]
    if len(set(columns)) != len(columns) or any(not re.fullmatch(r'[\w가-힣]+', c) for c in columns):
        raise ValueError('열 이름은 중복 없이 문자·숫자·밑줄로 적으세요.')
    if not 30 <= len(rows)-1 <= 1000:
        raise ValueError('학습용 실제 입력·정답 데이터 30~1000행이 필요합니다.')
    try:
        values = [[float(x) for x in row] for row in rows[1:]]
        if any(len(row) != len(columns) for row in values) or not np.isfinite(np.array(values)).all():
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError('각 행의 열 개수가 같아야 하며 모든 값은 빈칸 없는 유한한 숫자여야 합니다.') from None
    dataset = {'columns': columns, 'rows': values}
    analysis.custom_data(dataset)
    if len({r[-1] for r in values}) < 2:
        raise ValueError('목표값이 모두 같아 회귀 평가를 할 수 없습니다.')
    return dataset


def command(text, session):
    text = text.strip().rstrip('.!')
    if text.startswith('데이터 입력\n'):
        dataset = parse_dataset(text.split('\n', 1)[1])
        session['dataset'] = dataset
        session['runs'] = []
        return {'action': 'chat', 'reply': f"사용자 데이터 {len(dataset['rows'])}행을 등록했습니다. 입력: {', '.join(dataset['columns'][:-1])}, 목표: {dataset['columns'][-1]}. 이제 선형회귀 해줘 / 비선형회귀 해줘 / 신경망 학습해줘를 입력하세요. 데이터 변경으로 이전 분석 기록은 초기화했습니다.", 'degree': 2, 'values': {}}
    if text == '기본 데이터 사용':
        session.pop('dataset', None)
        session['runs'] = []
        return {'action': 'chat', 'reply': '기본 선체 저항 데이터로 전환했습니다. 이전 분석 기록은 초기화했습니다.', 'degree': 2, 'values': {}}
    if text.startswith('예측:'):
        values = {}
        try:
            for item in text.split(':', 1)[1].split(','):
                key, value = item.strip().split('=')
                key = key.strip()
                if key in values:
                    raise ValueError()
                values[key] = float(value)
            if not all(np.isfinite(v) for v in values.values()):
                raise ValueError()
        except ValueError:
            raise ValueError('예측: x=5, z=2 형식으로 중복 없이 유한한 숫자를 적으세요.') from None
        return {'action': 'predict', 'reply': '', 'degree': 2, 'values': values}
    actions = {'선형회귀 해줘': 'linear', '비선형회귀 해줘': 'polynomial', '신경망 학습해줘': 'neural', '인공신경망 학습해줘': 'neural', '모델 3가지 비교해줘': 'compare'}
    if text in actions:
        return {'action': actions[text], 'reply': '', 'degree': 2, 'values': {}}
    return None
