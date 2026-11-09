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
    fig.update_layout(height=480,template='plotly_white',title='보지 않은 선박에 대한 예측',
        xaxis_title='실제 정규화 저항',yaxis_title='예측 정규화 저항',
        margin={'l':25,'r':25,'t':60,'b':25},legend={'orientation':'h','y':1.12})
    st.plotly_chart(fig,use_container_width=True)
    st.caption('선형번호로 분리한 테스트 데이터입니다. 저항값은 N이 아닌 Rt/Δ × 10³입니다.')
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
                st.caption(f"채택 epoch: {r['best_epoch']} · 곡선은 표준화한 log(목표값)의 MSE")
                st.line_chart(pd.DataFrame(r['history']).set_index('epoch'))
        st.info(run['note'])
        if run.get('epoch_candidates'):
            for candidate in run['epoch_candidates']:
                st.caption(f"상한 {candidate['max_epochs']} 전체 학습 곡선")
                st.line_chart(pd.DataFrame(candidate['history']).set_index('epoch'))


def render():
    st.set_page_config(page_title='Hull Chat | 데이터처리개론',page_icon='⚓',layout='wide')
    st.markdown('''<style>
    .stApp {background:#f7f9fc;} h1,h2,h3 {color:#142c46;}
    [data-testid="stChatMessage"] {border:1px solid #e3e9f0;border-radius:16px;background:white;}
    [data-testid="stMetric"] {background:white;border:1px solid #e3e9f0;border-radius:12px;padding:16px;}
    .block-container {padding-top:2rem;max-width:1450px;}
    </style>''',unsafe_allow_html=True)
    st.caption('데이터처리개론 · 3차 과제 · OPEN LLM')
    st.title('말로 하는 선체 저항 분석')
    st.write('데이터를 살펴보고, 회귀와 신경망을 비교하고, 이전 분석을 이어가세요.')

    if 'hull_llm_session' not in st.session_state:
        sid = st.query_params.get('session','')
        try:
            current = memory.load(sid) if memory.valid_id(sid) else memory.new_session()
        except (ValueError,OSError,json.JSONDecodeError):
            st.warning('저장된 대화를 읽지 못해 새 대화를 시작합니다.')
            current = memory.new_session()
        st.session_state.hull_llm_session = current
        st.query_params['session'] = current['id']
    current = st.session_state.hull_llm_session

    with st.sidebar:
        from llm_lab.cloud_client import LlamaCppClient
        st.header('무료 CPU 대화 모델')
        client = LlamaCppClient()
        connected = True
        health = client.status()
        st.write('Qwen2.5 · 1.5B · Q4_K_M')
        if health['loaded']:
            st.success('모델 준비 완료 · 서버 CPU에서 실행')
        else:
            st.info('첫 요청에서 모델을 불러옵니다.')
            st.caption('최초 다운로드 약 1.12GB. 준비에 수 분 걸릴 수 있습니다.')
        if st.button('모델 준비 / 상태 확인', use_container_width=True):
            try:
                with st.spinner('모델 다운로드·확인·불러오기 중…'):
                    client.prepare()
                st.rerun()
            except LLMError as exc:
                st.error(str(exc))
        st.caption('유료 API와 내 PC의 Ollama 연결은 필요하지 않습니다. 이름 회상 등 일반 대화에는 오답이 있을 수 있습니다.')
        st.divider()
        st.subheader('대화 기록')
        st.caption('이 페이지 주소를 즐겨찾기에 저장하면 같은 대화를 다시 열 수 있습니다. '
                   '서버 임시 디스크에 저장됩니다. 재배포·절전 복구에서 사라질 수 있으니 다운로드로 보관하세요.')
        st.caption(f"저장된 분석 {len(current['runs'])}회")
        if st.button('새 대화 시작',use_container_width=True):
            fresh = memory.new_session()
            st.session_state.hull_llm_session = fresh
            st.query_params['session'] = fresh['id']
            st.rerun()
        st.download_button('대화·결과 JSON 다운로드',
            json.dumps(current,ensure_ascii=False,indent=2),file_name='llm_session.json',mime='application/json',
            use_container_width=True)
        st.caption('LLM 가중치를 학습시키는 기억 방식이 아니라 저장된 대화·분석 요약을 다시 읽는 방식입니다.')

    a,b,c = st.columns(3)
    a.metric('공개 데이터','1,018행')
    b.metric('선형 단위 분리','70개')
    c.metric('분석 입력 → 출력','8 → 1')
    if not connected:
        st.info('분석 화면은 준비되었습니다. 실제 대화는 Ollama와 선택한 모델이 연결되면 시작할 수 있습니다. '
                '이 화면을 LLM 실행 성공 증거로 사용하지 마세요.')

    quick = ['데이터를 설명해줘','선형회귀 해줘','비선형회귀 해줘','신경망 학습해줘','세 모델 비교해줘','이전 결과 보여줘']
    chosen = None
    for col,label in zip(st.columns(3),quick[:3]):
        if col.button(label,use_container_width=True,disabled=not connected): chosen=label
    for col,label in zip(st.columns(3),quick[3:]):
        if col.button(label,use_container_width=True,disabled=not connected): chosen=label
    for col,label in zip(st.columns(2),['50 epoch와 500 epoch를 비교해줘','오차가 큰 예측 10개 보여줘']):
        if col.button(label,use_container_width=True,disabled=not connected): chosen=label

    left,right=st.columns([1,1.4],gap='large')
    with left:
        st.subheader('분석 대화')
        for message in current['messages'][-16:]:
            with st.chat_message(message['role']): st.write(message['content'])
        if not current['messages']:
            with st.chat_message('assistant'):
                st.write('안녕하세요. “세 모델 비교해줘”라고 입력하면 같은 데이터로 회귀·신경망 분석을 실행할 수 있어요.')
        prompt=st.chat_input('예: 세 모델 비교해줘 / 이전 결과 보여줘',disabled=not connected,max_chars=1200)
        if prompt or chosen:
            try:
                with st.spinner('모델 준비 또는 응답 생성 중입니다. 첫 요청은 다운로드로 더 오래 걸립니다…'):
                    output=handle(prompt or chosen,current,client)
                st.session_state['hull_last_action']=output['action']
                st.rerun()
            except (LLMError,ValueError,OSError) as exc:
                st.error(str(exc))
        st.caption('자유로운 설명은 LLM 응답입니다. 분석 수치·그래프는 Python의 계산 결과입니다.')
    with right:
        if current['runs']:
            draw_result(current['runs'][-1])
            st.download_button('실제 분석 보고서 다운로드',memory.report(current),
                               file_name='hull_llm_report.md',mime='text/markdown',use_container_width=True)
        else:
            st.subheader('분석할 데이터')
            try:
                df,_=analysis.load_data()
                fig=go.Figure(go.Scatter(x=df['프루드수_Fn'],y=df['전저항_Rt/Δ×10³'],mode='markers',
                    marker={'color':'#1b7181','opacity':0.55,'size':6},name='관측값'))
                fig.update_layout(height=430,template='plotly_white',title='프루드수와 관측 저항',
                                  xaxis_title='프루드수 Fn',yaxis_title='Rt/Δ × 10³')
                st.plotly_chart(fig,use_container_width=True)
                st.caption('원자료 분포입니다. 아직 모델을 학습하거나 평가하지 않았습니다.')
                with st.expander('원자료 10행 보기'): st.dataframe(df.head(10),hide_index=True)
            except ValueError as exc: st.error(str(exc))
    with st.expander('내 코드 설명 연습 · 11/6 강의 반영'):
        st.markdown('''- **구조:** 입력 8 → 은닉 64, 32, 16 → 출력 1. 가중치·편향은 3,201개입니다.
- **표준화:** 학습 세트 평균·표준편차로 입력을 바꿉니다. 검증·테스트 통계는 학습에 쓰지 않습니다.
- **ReLU와 Adam:** ReLU는 은닉층에 비선형성을 주고, Adam은 역전파로 구한 기울기를 이용해 가중치를 갱신합니다.
- **MSE와 RMSE:** 곡선은 표준화한 log(목표)의 MSE, 표는 원래 저항 척도의 RMSE입니다. 서로 같은 수치로 비교하지 않습니다.
- **검증과 테스트:** 검증으로 epoch 상한을 선택하고 테스트로 선택 결과를 평가합니다. 테스트 결과를 보며 반복 조정하면 별도의 새 평가가 필요합니다.
- **50 epoch:** 교수님의 실험 제안입니다. 50이 항상 더 좋은 값은 아닙니다. 상한·실행·채택 epoch를 구분하세요.''')
        st.caption('근거: 2026. 11. 6. 데이터처리개론.txt 28:48~35:45. 이번 구현은 Keras가 아닌 sklearn MLPRegressor입니다.')
    with st.expander('실행 기록 확인'):
        st.caption('어떤 LLM 명령이 어떤 계산을 호출했는지 확인합니다. 임의 Python 코드를 실행하지 않습니다.')
        if current['events']:
            event=current['events'][-1]
            st.write({'모델':event['model'],'명령':event['action'],'응답 시간(초)':event['seconds']})
            st.caption('동작 선택 JSON 원문')
            st.code(event['raw_json'],language='json')
            if event.get('chat_raw_text'):
                st.caption('일반 대화 생성 원문')
                st.write(event['chat_raw_text'])
        else: st.write('아직 성공한 대화 요청이 없습니다.')

