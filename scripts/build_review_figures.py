import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from llm_lab import analysis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
out=Path('docs/review_20261111');out.mkdir(exist_ok=True)
r=analysis.run_analysis('compare');e=analysis.run_analysis('epoch_compare')
(out/'comparison.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
(out/'epochs.json').write_text(json.dumps(e,ensure_ascii=False,indent=2))
colors=['#367eaa','#e49c44','#288a76']
fig,axes=plt.subplots(1,3,figsize=(13,4.8),layout='constrained')
for ax,m,c in zip(axes,r['models'],colors):
 p=m['test_points'];x=[i['actual'] for i in p];y=[i['predicted'] for i in p];metric=next(v for v in m['metrics'] if v['split']=='테스트')
 ax.scatter(x,y,s=15,alpha=.65,color=c);hi=max(x+y);ax.plot([0,hi],[0,hi],'--',color='#8c98a8');ax.set_title(m['name']+'\nR² %.4f / RMSE %.4f'%(metric['R2'],metric['RMSE']));ax.set_xlabel('실제 정규화 저항');ax.set_ylabel('예측 정규화 저항');ax.grid(alpha=.15)
fig.suptitle('같은 선형 분할에서 비교한 모델 3가지',fontsize=18)
fig.savefig(out/'predictions.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
for q,c in zip(e['epoch_candidates'],['#367eaa','#288a76']):
 ax.plot([h['epoch'] for h in q['history']],[h['검증 MSE(표준화)'] for h in q['history']],label=f"상한 {q['max_epochs']}회 · 실행 {q['executed_epochs']}회 · 채택 {q['best_epoch']}회",color=c)
ax.set(title='신경망 검증 손실: 학습 상한 50회와 500회',xlabel='학습 횟수 (epoch)',ylabel='검증 MSE (표준화한 로그 목표값)');ax.legend();ax.grid(alpha=.2)
fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
print(json.dumps({'metrics':[next(v for v in m['metrics'] if v['split']=='테스트') for m in r['models']],'epochs':[{k:v for k,v in q.items() if k!='history'} for q in e['epoch_candidates']]},ensure_ascii=False))
