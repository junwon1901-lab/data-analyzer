"""실제 CPU GGUF 호출로 검증하고 증거를 JSON/Markdown에 남긴다. 모의 응답 없음."""
import argparse
import json
import platform
import resource
import os
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from llm_lab.cloud_client import LlamaCppClient
from llm_lab.agent import handle
from llm_lab import session as memory


def main():
    parser=argparse.ArgumentParser()
        parser.add_argument('--output',default='docs/free_cpu_verification')
    args=parser.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
    client=LlamaCppClient()
    begin=time.perf_counter()
    status=client.prepare()
    load_seconds=time.perf_counter()-begin
    session=memory.new_session()
    cases=[('안녕! 내 이름은 준원이야. 짧게 인사해줘.','chat'),
           ('내 이름이 뭐라고 했지?','chat'),
           ('선체 저항 데이터를 불러와서 설명해줘.','describe'),
           ('선형회귀 해줘.','linear'),
           ('비선형회귀 해줘.','polynomial'),
           ('신경망을 학습해줘.','neural'),
           ('세 모델을 같은 조건으로 비교해줘.','compare'),
           ('50 epoch와 500 epoch를 비교해줘.','epoch_compare'),
           ('오차가 큰 예측 10개 보여줘.','errors'),
           ('보고서 만들어줘.','report'),
           ('신경망이 뭔지 짧게 설명해줘. 학습은 실행하지 마.','chat'),
           ('Fn=0.1, Rn=9870390, Cp=0.5674, Cb=0.3667, Cw=0.6908, L_B=3.6232, B_T=3.043, LCB_L=-0.023으로 저항을 예측해줘.','predict')]
    checks=[]
    for text,expected in cases:
        try:
            answer=handle(text,session,client)
            passed=answer['action']==expected
            if text=='내 이름이 뭐라고 했지?': passed=passed and '준원' in answer['reply']
            if expected=='predict' and answer['action']=='predict':
                passed=passed and answer['evidence']['prediction']['inputs']==dict(Fn=.1,Rn=9870390,Cp=.5674,Cb=.3667,Cw=.6908,L_B=3.6232,B_T=3.043,LCB_L=-.023)
            checks.append({'prompt':text,'expected':expected,'actual':answer['action'],
                           'passed':passed,'reply':answer['reply'],'seconds':answer['evidence']['seconds']})
        except Exception as exc:
            checks.append({'prompt':text,'expected':expected,'actual':None,'passed':False,
                           'error':type(exc).__name__+': '+str(exc)})
        print(json.dumps(checks[-1],ensure_ascii=False),flush=True)
    # 디스크에 기록한 JSON을 새 객체로 읽은 후 후속 요청.
    session=memory.load(session['id'])
    answer=handle('이전 분석 결과를 다시 보여줘.',session,client)
    checks.append({'prompt':'이전 분석 결과를 다시 보여줘.','expected':'last',
                   'actual':answer['action'],'passed':answer['action']=='last'})
    directory=Path(args.output)
    directory.mkdir(parents=True,exist_ok=True)
    import numpy,pandas,sklearn,streamlit
    evidence={'environment':{'python':sys.version,'platform':platform.platform(),'runtime':status,'model_load_seconds':load_seconds,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               'numpy':numpy.__version__,'pandas':pandas.__version__,'sklearn':sklearn.__version__,
               'streamlit':streamlit.__version__},'checks':checks,'session':session,
               'all_routing_checks_passed':all(c['passed'] for c in checks)}
    (directory/'actual_execution.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    (directory/'actual_report.md').write_text(memory.report(session),encoding='utf-8')
    print('Saved to',directory,flush=True)
    if not evidence['all_routing_checks_passed']: raise SystemExit(1)

if __name__=='__main__':main()
