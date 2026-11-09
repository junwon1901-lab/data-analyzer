"""실제 모델과 Streamlit 화면 검사. 오답은 passed=false로 남긴다."""
import os,sys,json,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest
from llm_lab import session as memory
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:])
a=AppTest.from_file(str(ROOT/'llm_chat.py'),default_timeout=180).run()
assert not a.exception
checks=[{'case':'initial page','passed':True}]
for q,action in [('안녕. 내 이름은 준원이야.','chat'),('내 이름이 뭐였지?','chat'),('데이터를 설명해줘','describe'),('선형회귀 해줘','linear')]:
 t=time.perf_counter();a.chat_input[0].set_value(q).run(timeout=180)
 assert not a.exception and not a.error
 s=a.session_state['hull_llm_session']
 assert s['events'][-1]['action']==action
 ok='준원' in s['messages'][-1]['content'] if '이름이 뭐' in q else True
 checks.append({'prompt':q,'action':action,'passed':ok,'ui_flow_passed':True,'reply':s['messages'][-1]['content'],'wall_seconds':time.perf_counter()-t})
assert len(a.dataframe)>=2 and len(a.get('plotly_chart'))>=1
assert memory.load(s['id'])['events']==s['events']
b=AppTest.from_file(str(ROOT/'llm_chat.py'),default_timeout=180).run()
b.chat_input[0].set_value('안녕. 이름을 아직 말하지 않았어.').run(timeout=180)
assert not b.exception and not b.error
s2=b.session_state['hull_llm_session']
assert s2['id']!=s['id'] and s2['runs']==[] and '준원' not in s2['messages'][-1]['content']
checks.append({'case':'different visitor has no previous name/results','passed':True,'reply':s2['messages'][-1]['content']})
r={'scope':'Development Linux CPU, not hosted Streamlit','checks':checks,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'all_passed':all(x['passed'] for x in checks)}
out=ROOT/'docs/free_cpu_verification';out.mkdir(exist_ok=True)
(out/'ui_execution.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print(json.dumps(r,ensure_ascii=False,indent=2))
