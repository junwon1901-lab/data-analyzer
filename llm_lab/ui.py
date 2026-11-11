"""A conversation workspace: commands and real Python results share one thread."""
from html import escape
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from llm_lab.client import LLMError
from llm_lab.agent import handle
from llm_lab import analysis, session as memory

COLORS = ['#5281d6', '#6aab9b', '#ad8acb']
REPORT_URL = 'https://app.notion.com/p/413d259f3ec78096923dcb8e5cc88b92'


def chart_style(fig, **kwargs):
    fig.update_layout(template='plotly_white', height=360, font={'family': 'Noto Sans KR, sans-serif', 'color': '#76859a', 'size': 11},
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#fafbfd',
                      margin={'l': 25, 'r': 15, 't': 45, 'b': 45},
                      legend={'orientation': 'h', 'y': -0.2}, **kwargs)
    fig.update_xaxes(gridcolor='#edf1f7', zerolinecolor='#e3eaf4')
    fig.update_yaxes(gridcolor='#edf1f7', zerolinecolor='#e3eaf4')
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


def network_diagram(model):
    cfg = model['config']
    layers = [len(model['data']['features']), *cfg.get('hidden_layers', (64,32,16)), 1]
    items = []
    for i, size in enumerate(layers):
        if i:
            items.append('<span class="arrow">→</span>')
        label = '입력' if i == 0 else '출력' if i == len(layers)-1 else f'은닉층 {i}'
        items.append(f'<div class="lab-layer {"output" if i == len(layers)-1 else ""}"><strong>{size}</strong>{label}</div>')
    st.markdown('<div class="lab-network">' + ''.join(items) + '</div>', unsafe_allow_html=True)
    st.caption(f"활성화 함수: {cfg.get('activation', 'relu')} · 입력을 가중합한 뒤 비선형 활성화를 거쳐 출력합니다.")


def draw_result(run, *, errors_only=False):
    custom = run.get('custom_data', False)
    evaluation = run.get('evaluation_label', '테스트')
    selected = next(r for r in run['models'] if r['kind'] == run['selected'])
    if errors_only:
        st.markdown('#### 오차가 큰 예측')
        table = pd.DataFrame(analysis.worst_errors(run)).drop(columns=['hull','Fn'], errors='ignore').rename(columns={
            'row':'원자료 행','actual':'실제값','predicted':'예측값','residual':'잔차','absolute_error':'절대오차','predicted_actual_ratio':'예측/실제'})
        st.dataframe(table, hide_index=True, width='stretch')
        return
    st.markdown(f"### {selected['name']}" + (' · 비교 결과' if len(run['models']) > 1 else ' · 분석 결과'))
    st.caption(f"{selected['data']['source'] if custom else '공개 선체 저항 데이터'} · {selected['data']['rows']:,}행 · {evaluation}")
    if selected.get('equation'):
        st.markdown(f'<div class="lab-equation">{escape(selected["equation"])}</div>', unsafe_allow_html=True)
    metric = next(m for m in selected['metrics'] if m['split'] == evaluation)
    for column, title, key in zip(st.columns(3), ['R² · 결정계수', 'RMSE · 제곱평균제곱근오차', 'MAE · 평균절대오차'], ['R2','RMSE','MAE']):
        column.metric(title, f'{metric[key]:.4f}')
    fig = go.Figure()
    if selected.get('curve'):
        fig.add_trace(go.Scatter(x=[p['x'] for p in selected['input_points']], y=[p['y'] for p in selected['input_points']], mode='markers', name='입력한 점', marker={'size':8,'color':'#263c5b'}))
        for color, model in zip(COLORS, run['models']):
            fig.add_trace(go.Scatter(x=[p['x'] for p in model['curve']], y=[p['y'] for p in model['curve']], mode='lines', name=model['name'],line={'color':color,'width':2}))
        chart_style(fig, title='입력 데이터와 적합 곡선', xaxis_title=selected['data']['features'][0], yaxis_title=selected['data']['target'])
        st.caption('곡선은 전체 입력 데이터에 적합했습니다. 위 지표는 각 행을 학습에서 제외한 교차검증 예측으로 계산합니다.')
    else:
        values = []
        for color, model in zip(COLORS, run['models']):
            points = model['test_points']
            y, p = [x['actual'] for x in points], [x['predicted'] for x in points]
            values.extend(y+p)
            fig.add_trace(go.Scatter(x=y,y=p,mode='markers',name=model['name'],marker={'size':6,'opacity':.65,'color':color}))
        lo,hi = min(values),max(values)
        fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='예측 = 실제',line={'dash':'dash','color':'#a9b7cb'}))
        chart_style(fig,title='실제값과 예측값',xaxis_title='실제값',yaxis_title='예측값')
    if not custom:
        st.caption('목표값: Rt/Δ × 10³ · 힘의 단위 N이 아닙니다. 같은 선박의 행은 학습·검증·테스트에 섞이지 않습니다.')
    rows = [m for r in run['models'] for m in r['metrics'] if m['split'] == evaluation]
    st.dataframe(pd.DataFrame(rows).rename(columns={'model':'모델','R2':'R²','rows':'평가 행 수','negative_predictions':'음수 예측 수'})[['모델','R²','RMSE','MAE','평가 행 수','음수 예측 수']], hide_index=True, width='stretch')
    neural = next((r for r in run['models'] if r['kind'] == 'neural'), None)
    if neural:
        st.markdown('#### 신경망은 이렇게 연결됩니다')
        network_diagram(neural)
        if neural['history']:
            history = pd.DataFrame(neural['history'])
            loss = go.Figure()
            for color, col in zip(COLORS, [c for c in history.columns if c != 'epoch']):
                loss.add_trace(go.Scatter(x=history['epoch'],y=history[col],name=col,mode='lines',line={'color':color,'width':2}))
            chart_style(loss,title='실제 학습 손실 곡선',xaxis_title='학습 횟수 (epoch)',yaxis_title='표준화한 목표 척도의 손실')
            if neural['best_epoch'] is not None:
                st.caption(f"검증 오차가 가장 작았던 {neural['best_epoch']} epoch의 가중치를 채택했습니다.")
    if run.get('epoch_candidates'):
        st.markdown('#### 50 / 500 epoch 상한 비교')
        rows = [{k:v for k,v in r.items() if k!='history'} for r in run['epoch_candidates']]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption(run['selection_basis'])
    st.caption(run['note'])
    st.caption('이어가기: x가 10일 때 예측해줘 · 다른 모델로 학습해줘 · 오차가 큰 예측 10개 보여줘 · 보고서 만들어줘' if custom and len(selected['data']['features']) == 1 else '이어가기: 오차가 큰 예측 10개 보여줘 · 보고서 만들어줘 · 이전 결과 보여줘')


def rail(current):
    recent = [m['content'].splitlines()[0][:24] for m in current['messages'] if m['role'] == 'user'][-4:]
    history = '<br>'.join(escape(m) for m in reversed(recent)) or '첫 번째 분석을 시작해 보세요.'
    st.markdown(f'''<aside class="lab-rail"><div class="lab-wordmark"><span class="lab-symbol">∿</span> Data Lab</div>
+    <div class="rail-sub">JUNWON'S WORKSPACE</div>
+    <div class="rail-label">CHAT GUIDE</div><div class="rail-guide">
+    <div class="rail-command"><strong>직선으로 관계 찾기</strong><code>선형회귀 해줘</code></div>
+    <div class="rail-command"><strong>곡선으로 관계 찾기</strong><code>비선형회귀 해줘</code></div>
+    <div class="rail-command"><strong>입력과 출력 학습하기</strong><code>신경망 학습해줘</code></div>
+    <div class="rail-command"><strong>같은 조건에서 비교하기</strong><code>모델 3가지 비교해줘</code></div>
+    <div class="rail-command"><strong>결과 정리하기</strong><code>보고서 만들어줘</code></div>
+    <div class="rail-command"><strong>처음부터 시작하기</strong><code>새 대화 시작</code></div></div>
+    <div class="rail-label">RECENT REQUESTS</div><div class="rail-history">{history}</div>
+    <div class="rail-bottom"><a href="https://github.com/junwon1901-lab/data-analyzer" target="_blank">GitHub ↗</a><a href="{REPORT_URL}" target="_blank">과제 보고서 ↗</a><br>동아대학교 · 데이터처리개론<br>서준원 · 오픈소스 LLM 실습</div></aside>'''.replace('\n+', '\n'), unsafe_allow_html=True)


def composer_guide(current):
    data = current.get('dataset')
    with st.container(key='composer_guide'):
        st.markdown('**채팅 가이드 · 항상 표시** 　`도움말`을 입력하면 전체 사용법을 볼 수 있습니다.')
        if data:
            values = ', '.join(f'{k}=1' for k in data['columns'][:-1])
            st.markdown(f"현재 **내 데이터 {len(data['rows'])}행** · 목표 `{data['columns'][-1]}` · 예측 형식 `{values}`")
            st.markdown(f'학습 후 `예측: {values}` · `기본 데이터 사용`으로 전환')
        else:
            st.markdown('`선형회귀 해줘` · `비선형회귀 해줘` · `신경망 학습해줘` · `모델 3가지 비교해줘`')
            st.markdown('내 숫자도 분석: `x: 1,2,3,4,5,6 y: 2,4,6,8,10,12 선형회귀 해줘`')
        st.markdown('표 붙여 넣기: 첫 줄 `데이터 입력`, 다음 줄부터 열 이름과 숫자 30행 이상 · 마지막 열은 정답 · 줄바꿈 Shift+Enter')


def render():
    from llm_lab.cloud_client import LlamaCppClient
    st.set_page_config(page_title='Data Lab · 선체 저항과 데이터 분석',page_icon='〰',layout='wide')
    st.markdown('<style>' + Path(__file__).with_name('ui.css').read_text(encoding='utf-8') + '</style>', unsafe_allow_html=True)
    if 'hull_llm_session' not in st.session_state:
        st.session_state.hull_llm_session = memory.new_session()
    current = st.session_state.hull_llm_session
    rail(current)
    st.markdown('<div class="lab-top"><span class="brand">데이터 분석 도우미</span><span class="lab-badge"><i></i>Python 분석 · 오픈소스 LLM</span></div>', unsafe_allow_html=True)
    if not current['messages']:
        st.markdown('''<section class="lab-hero"><div class="lab-eyebrow">YOUR DATA, A CLEARER PICTURE</div><h1><em>안녕하세요,</em><br>무엇을 분석해 볼까요?</h1><p>숫자를 건네고, 궁금한 것을 물어보세요.<br>선형회귀부터 인공신경망까지 대화로 이어집니다.</p></section>''', unsafe_allow_html=True)
        st.markdown('''<div class="lab-intro-grid"><div class="lab-intro-card"><div class="number">01 / REGRESSION</div><strong>관계와 곡선 찾기</strong><p>“선형회귀 해줘”<br>“3차 다항식으로 해줘”</p></div><div class="lab-intro-card"><div class="number">02 / NEURAL NETWORK</div><strong>데이터로 학습하기</strong><p>“신경망 학습해줘”<br>학습 곡선과 구조도 확인</p></div><div class="lab-intro-card"><div class="number">03 / YOUR DATA</div><strong>내 숫자로 예측하기</strong><p>x·y 목록이나 표를 입력하고<br>“x가 10일 때 예측해줘”</p></div></div>''', unsafe_allow_html=True)
        st.markdown('<div class="lab-data-strip"><b>처음이라면</b>　“모델 3가지 비교해줘”로 시작해 보세요.<br>기본 데이터 1,018행 · 70개 선형 · 입력 8개 · 목표: 정규화 저항</div>', unsafe_allow_html=True)
    runs = {r.get('id'): r for r in current['runs'] if r.get('id')}
    for message in current['messages'][-20:]:
        with st.chat_message(message['role'], avatar='〰' if message['role'] == 'assistant' else None):
            content = message['content']
            run = runs.get(message.get('run_id'))
            if run and message.get('action') in ('linear','polynomial','neural','compare','epoch_compare'):
                content = '데이터를 분석했습니다. 계산한 결과를 함께 살펴볼게요.'
            if message['role'] == 'user' and len(content) > 1400:
                content = content[:1000] + '\n\n… (입력한 데이터는 등록되어 있습니다.)'
            st.markdown(content)
            if message.get('report'):
                st.markdown(message['report'])
            elif run:
                draw_result(run, errors_only=message.get('action') == 'errors')
            if message.get('prediction'):
                p = message['prediction']
                st.metric(f"{p['model']} · {p['target']} 예측값", f"{p['value']:.6f}")
    composer_guide(current)
    prompt = st.chat_input('데이터를 입력하거나, 궁금한 것을 물어보세요',max_chars=100000)
    if prompt:
        if prompt.strip().rstrip('.!') in ('새 대화 시작','새 대화','대화 초기화'):
            st.session_state.hull_llm_session = memory.new_session()
            st.session_state.pop('hull_last_action',None)
            st.rerun()
        try:
            with st.spinner('요청을 처리하고 있습니다. 일반 대화의 첫 요청은 모델 준비에 시간이 걸릴 수 있습니다.'):
                output = handle(prompt,current,LlamaCppClient())
            st.session_state['hull_last_action'] = output['action']
            st.rerun()
        except (LLMError,ValueError,OSError) as exc:
            st.error(str(exc))
