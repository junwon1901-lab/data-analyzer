"""Chat data commands are parsed without asking an LLM to rewrite numbers."""
import csv
import io
import re
import numpy as np
from llm_lab import analysis

NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


def plan(action='chat', reply='', degree=2, values=None):
    return {'action': action, 'reply': reply, 'degree': degree, 'values': values or {}}


def parse_xy(text):
    """Extract literal numeric lists only. Never ask an LLM to transcribe data."""
    match = re.search(r'\bx\s*[:：]\s*(.*?)\s*\by\s*[:：]\s*((?:[+\-\d.eE,，;；\s])+)', text, re.I | re.S)
    if not match:
        return None
    lists = []
    for group in match.groups():
        tokens = [s for s in re.split(r'[,，;；\s]+', group.strip()) if s]
        if any(not re.fullmatch(NUMBER, token) for token in tokens):
            raise ValueError('x: 숫자,숫자 y: 숫자,숫자 형식으로 입력하세요.')
        lists.append([float(token) for token in tokens])
    x, y = lists
    if len(x) != len(y):
        raise ValueError(f'x와 y 개수가 다릅니다: x {len(x)}개, y {len(y)}개. 같은 개수로 입력하세요.')
    if not 6 <= len(x) <= 1000:
        raise ValueError('짧은 x·y 목록은 6~1000쌍을 입력하세요. 표 입력은 30행 이상입니다.')
    if not np.isfinite([x, y]).all() or len(set(x)) < 4 or len(set(y)) < 2:
        raise ValueError('유한한 숫자와 서로 다른 x 4개 이상, 서로 다른 y 2개 이상이 필요합니다.')
    dataset = {'columns': ['x', 'y'], 'rows': [list(pair) for pair in zip(x, y)]}
    if len(x) < 30:
        dataset['small_sample'] = True
    else:
        analysis.custom_data(dataset)
    return dataset


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
    normalized = re.sub(r'\s+', '', text).lower()
    degree = 3 if re.search(r'3\s*차|삼차', text) else 2
    dataset = parse_xy(text)
    if dataset is not None:
        session['dataset'] = dataset
        session['runs'] = []
        if re.search(r'비교|가장|최적|잘\s*맞|best|compare', text, re.I):
            return plan('compare', degree=degree)
        if re.search(r'신경망|neural', text, re.I):
            return plan('neural', degree=degree)
        if re.search(r'비선형|다항|차\s*(?:회귀|다항)|polynomial', text, re.I):
            return plan('polynomial', degree=degree)
        if re.search(r'선형|linear', text, re.I):
            return plan('linear')
        return plan(reply=f'x·y {len(dataset["rows"])}쌍을 등록했습니다. 선형회귀 해줘 / 비선형회귀 해줘 / 신경망 학습해줘 / 모델 3가지 비교해줘로 이어가세요.')
    if re.search(r'\bx\s*[:：]|\by\s*[:：]', text, re.I):
        raise ValueError('x: 1,2,3,4,5,6 y: 2,4,6,8,10,12처럼 두 목록을 함께 입력하세요.')
    if text.startswith('데이터 입력\n'):
        dataset = parse_dataset(text.split('\n', 1)[1])
        session['dataset'] = dataset
        session['runs'] = []
        return {'action': 'chat', 'reply': f"사용자 데이터 {len(dataset['rows'])}행을 등록했습니다. 입력: {', '.join(dataset['columns'][:-1])}, 목표: {dataset['columns'][-1]}. 이제 선형회귀 해줘 / 비선형회귀 해줘 / 신경망 학습해줘를 입력하세요. 데이터 변경으로 이전 분석 기록은 초기화했습니다.", 'degree': 2, 'values': {}}
    if text == '기본 데이터 사용':
        session.pop('dataset', None)
        session['runs'] = []
        return {'action': 'chat', 'reply': '기본 선체 저항 데이터로 전환했습니다. 이전 분석 기록은 초기화했습니다.', 'degree': 2, 'values': {}}
    if normalized in ('도움말', '사용법', '가이드', '채팅가이드', '뭘할수있어?', '무엇을할수있어?'):
        return plan(reply='**채팅으로 이렇게 요청하세요.**\n\n'
                    '1. 선형회귀 해줘 / 비선형회귀 해줘 / 신경망 학습해줘\n'
                    '2. 모델 3가지 비교해줘\n'
                    '3. 내 숫자 분석: x: 1,2,3,4,5,6 y: 2,4,6,8,10,12 선형회귀 해줘\n'
                    '4. 후속 예측: x가 10일 때 예측해줘\n'
                    '5. 데이터 입력 다음 줄부터 CSV·엑셀 표 붙여 넣기 (30행 이상)\n'
                    '6. 이전 결과 보여줘 / 학습 곡선 보여줘 / 보고서 만들어줘\n'
                    '7. 기본 데이터 사용 / 새 대화 시작\n\n'
                    '**기본 선체 데이터의 새 값 예측**\n\n'
                    '`예측: Fn=0.3, Rn=10000000, Cp=0.6, Cb=0.4, Cw=0.7, L_B=3.5, B_T=2.5, LCB_L=0`\n\n'
                    '원하는 모델을 먼저 학습한 뒤 숫자를 수정하세요. 8개 값을 모두 입력해야 합니다.\n\n'
                    '일반 개념 질문에는 설치한 공개 LLM이 답합니다. 결과의 출처·단위는 현재 데이터를 기준으로 확인하세요.')
    aliases = {'이전결과보여줘': 'last', '결과보여줘': 'last', '학습곡선보여줘': 'last',
               '신경망구조보여줘': 'last', '보고서만들어줘': 'report', '보고서작성해줘': 'report',
               '오차가큰예측10개보여줘': 'errors', '데이터설명해줘': 'describe'}
    if normalized in aliases:
        return plan(aliases[normalized])
    current = session.get('dataset')
    if current and len(current['columns']) == 2:
        feature = current['columns'][0]
        match = re.search(rf'\b{re.escape(feature)}\s*(?:가|이|는|은|=|:|is)?\s*({NUMBER})\s*(?:일|이면|일때|일 때|넣|에서|when)?', text, re.I)
        if match and re.search(r'예측|predict|넣|이면', text, re.I):
            value = float(match.group(1))
            if not np.isfinite(value):
                raise ValueError('유한한 숫자를 입력하세요.')
            return plan('predict', values={feature: value})
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
    if re.fullmatch(r'(?:모델3가지|세모델|세가지모델|3개모델|모델세가지)(?:를)?비교해줘', normalized) or normalized in ('가장잘맞는모델찾아줘', '모델비교해줘'):
        return plan('compare', degree=degree)
    if re.fullmatch(r'(?:비선형회귀|다항회귀|[23]차다항식|[23]차다항회귀|[23]차회귀)(?:으?로)?(?:분석|실행)?해줘', normalized):
        return plan('polynomial', degree=degree)
    if re.fullmatch(r'(?:선형회귀|linearregression)(?:로)?(?:분석|실행)?해줘', normalized):
        return plan('linear')
    if re.fullmatch(r'(?:인공)?신경망(?:으로)?(?:학습|분석|실행)?해줘', normalized):
        return plan('neural')
    return None
