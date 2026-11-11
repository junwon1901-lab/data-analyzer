"""대화 중심 Streamlit UI. 생성 답변과 실제 계산 결과를 구분해서 보여준다."""
import json
import os
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from llm_lab.client import OllamaClient, LLMError
from llm_lab.agent import handle
from llm_lab import analysis, session as memory


def draw_result(run):
    custom = run.get('custom_data', False)
    st.subheader('분석 결과')
    st.caption(f"{run['time']} · {run['selection_basis']}")
    rows = [m for r in run['models'] for m in r['metrics'] if m['split']=='테스트']
    table = pd.DataFrame(rows).rename(columns={'model':'모델','split':'구분','rows':'행 수',
        'negative_predictions':'음수 예측 수','R2':'R²'})
    st.dataframe(table[['모델','R²','RMSE','MAE','행 수','음수 예측 수']], hide_index=True,
                 use_container_width=True)
    fig = go.Figure()
    values = []
    for r in run['models']:
        points = r['test_points']
        y, p = [x['actual'] for x in points], [x['predicted'] for x in points]
        values.extend(y+p)
        fig.add_trace(go.Scatter(x=y,y=p,mode='markers',name=r['name'],
                      marker={'size':7,'opacity':0.7}))
    lo,hi = min(values),max(values)
    fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='예측 = 실제',
                             line={'dash':'dash','color':'#94a3b8'}))
    fig.update_layout(height=480,template='plotly_white',title='사용자 데이터 테스트 예측' if custom else '보지 않은 선박에 대한 예측',
        xaxis_title='실제값' if custom else '실제 정규화 저항',yaxis_title='예측값' if custom else '예측 정규화 저항',
        margin={'l':25,'r':25,'t':60,'b':100},legend={'orientation':'h','y':-0.22,'x':0})
    st.plotly_chart(fig,use_container_width=True)
    st.caption('사용자 데이터의 무작위 테스트 분할입니다. 단위는 입력 원자료 기준입니다.' if custom else '선형번호로 분리한 테스트 데이터입니다. 저항값은 N이 아닌 Rt/Δ × 10³입니다.')
    st.subheader('오차가 큰 개별 예측 10개')
    st.caption('선택 모델 · 절대오차 순. R²는 정답률이 아니며 아래 큰 오차를 가리지 못합니다. 비율 0.5는 실제값의 절반을 예측했다는 뜻입니다.')
    errors = pd.DataFrame(analysis.worst_errors(run)).rename(columns={
        'row':'원자료 행', 'hull':'선형번호', 'actual':'실제값', 'predicted':'예측값',
        'residual':'잔차(실제-예측)', 'absolute_error':'절대오차', 'predicted_actual_ratio':'예측/실제'})
    st.dataframe(errors, hide_index=True, use_container_width=True)
    if run.get('epoch_candidates'):
        st.subheader('50 / 500 epoch 상한 비교')
        rows = [{k:v for k,v in r.items() if k!='history'} for r in run['epoch_candidates']]
        st.dataframe(pd.DataFrame(rows).rename(columns={'max_epochs':'상한', 'executed_epochs':'실행',
                     'best_epoch':'채택', 'train_RMSE':'학습 RMSE', 'validation_RMSE':'검증 RMSE',
                     'minimum_validation_MSE':'최소 검증 MSE', 'last_validation_MSE':'마지막 검증 MSE'}),
                     hide_index=True, use_container_width=True)
        st.caption('두 후보 모두 동일한 시드·분할·구조·조기 종료를 사용합니다. 검증 RMSE로 선택한 모델만 위 테스트 표에 표시됩니다.')
    with st.expander('검증 성능 · 설정 · 학습 곡선'):
        all_metrics = pd.DataFrame([m for r in run['models'] for m in r['metrics']])
        st.dataframe(all_metrics,hide_index=True,use_container_width=True)
        for r in run['models']:
            st.write(r['name'],r['config'])
            if r['history']:
                st.caption(f"채택 epoch: {r['best_epoch']} · 곡선은 변환·표준화한 목표값의 MSE")
                st.line_chart(pd.DataFrame(r['history']).set_index('epoch'))
        st.info(run['note'])
        if run.get('epoch_candidates'):
            for candidate in run['epoch_candidates']:
                st.caption(f"상한 {candidate['max_epochs']} 전체 학습 곡선")
                st.line_chart(pd.DataFrame(candidate['history']).set_index('epoch'))


def render():
    from llm_lab.cloud_client import LlamaCppClient
    st.set_page_config(page_title='선체 저항 챗봇',page_icon='⚓',layout='wide')
    st.title('선체 저항 분석 챗봇')
    st.caption('선형회귀 · 비선형회귀 · 인공신경망 | 원하는 분석을 채팅으로 요청하세요.')
    if 'hull_llm_session' not in st.session_state:
        st.session_state.hull_llm_session = memory.new_session()
    current = st.session_state.hull_llm_session
    client = LlamaCppClient()
    with st.container(key='always_guide'):
        st.markdown('**채팅 가이드 — 항상 표시**')
        st.caption('학습: 선형회귀 해줘 · 비선형회귀 해줘 · 신경망 학습해줘 · 모델 3가지 비교해줘 | 기록: 보고서 만들어줘 | 초기화: 새 대화 시작')
        dataset = current.get('dataset')
        if dataset:
            st.caption(f"현재: 사용자 데이터 {len(dataset['rows'])}행 | 입력: {', '.join(dataset['columns'][:-1])} | 목표: {dataset['columns'][-1]}")
            st.code('예측: ' + ', '.join(f'{k}=1' for k in dataset['columns'][:-1]), language=None)
        else:
            st.caption('현재: 기본 선체 데이터 | 학습 후 아래 값들을 원하는 숫자로 바꾸어 예측하세요.')
            st.code('예측: Fn=0.3, Rn=10000000, Cp=0.6, Cb=0.4, Cw=0.7, L_B=3.5, B_T=2.5, LCB_L=0', language=None)
        st.caption('내 데이터 학습: 첫 줄에 데이터 입력, 다음 줄에 x,y 등 열 이름, 그 아래 숫자 표를 붙여 넣으세요. 마지막 열=정답, 앞 열=입력. CSV/엑셀 탭 구분, 30~1000행·입력 1~8개. 기본 데이터 사용으로 복귀. 여러 줄 입력은 Shift+Enter.')
    st.markdown('<style>.st-key-always_guide {position:sticky;top:0px;background:var(--background-color, white);border:1px solid #8885;border-radius:12px;padding:10px;z-index:99;max-height:32vh;overflow:auto;}</style>', unsafe_allow_html=True)
    if not current['messages']:
        with st.chat_message('assistant'):
            st.write('안녕하세요. 선형회귀 해줘, 비선형회귀 해줘, 신경망 학습해줘처럼 입력하세요. 모델 3가지 비교해줘, 이전 결과 보여줘, 보고서 만들어줘도 가능합니다. 처음 질문할 때 모델을 자동으로 준비합니다. 새 대화 시작이라고 입력하면 대화를 초기화합니다.')
    for message in current['messages'][-16:]:
        with st.chat_message(message['role']):
            st.write(message['content'])
    if current['runs']:
        with st.chat_message('assistant'):
            draw_result(current['runs'][-1])
    if st.session_state.get('hull_last_action') == 'report':
        with st.chat_message('assistant'):
            st.markdown(memory.report(current))
    prompt = st.chat_input('분석 요청이나 질문을 입력하고 Enter를 누르세요',max_chars=100000)
    if prompt:
        if prompt.strip().rstrip('.!') in ('새 대화 시작','새 대화','대화 초기화'):
            st.session_state.hull_llm_session = memory.new_session()
            st.session_state.pop('hull_last_action',None)
            st.rerun()
        try:
            with st.spinner('모델 준비 및 요청 처리 중… 첫 요청은 더 오래 걸릴 수 있습니다.'):
                output=handle(prompt,current,client)
            st.session_state['hull_last_action']=output['action']
            st.rerun()
        except (LLMError,ValueError,OSError) as exc:
            st.error(str(exc))
    st.caption('분석 수치와 그래프는 Python 계산 결과입니다. 일반 대화에는 오답이 있을 수 있습니다. 서버 재시작 시 대화 기록이 사라질 수 있습니다.')
