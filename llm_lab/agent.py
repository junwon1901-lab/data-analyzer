"""해석 → 검증 → 허용 함수 실행 → 실제 결과 기록."""
import uuid
from llm_lab import analysis, session as memory


def handle(text, session, client):
    from llm_lab.input_data import command
    plan = command(text, session)
    if plan is None:
        plan, evidence = client.plan(text, session['messages'], memory.context(session))
    else:
        evidence = {'model': '명시적 채팅 명령', 'seconds': 0}
    dataset = session.get('dataset')
    action = plan['action']
    result = None
    if action == 'chat':
        reply = plan['reply'] or '질문을 조금 더 구체적으로 입력해 주세요.'
    elif action == 'describe':
        info = analysis.describe()
        if dataset is not None:
            info = {'rows': len(dataset['rows']), 'columns': len(dataset['columns']), 'hulls': '해당 없음', 'target': dataset['columns'][-1], 'split_rows': '최대 5-fold 교차검증' if dataset.get('small_sample') else '동일 무작위 분할'}
        reply = (f"현재 데이터 {info['rows']:,}행, {info['columns']}열, "
                 f"선형 {info['hulls']}개를 불러왔습니다. 입력 변수는 화면 가이드를 참고하세요. 목표는 {info['target']}입니다. "
                 f"학습·검증·테스트 행 수는 {info['split_rows']}입니다.")
    elif action in ('linear','polynomial','neural','compare','epoch_compare'):
        if len(session['runs']) >= 20:
            raise ValueError('한 대화의 실습 기록 한도는 20회입니다. 기록을 내려받은 뒤 새 대화를 시작하세요.')
        result = analysis.run_analysis(action, plan['degree'], dataset=dataset)
        result['time'] = memory.now()
        result['id'] = uuid.uuid4().hex
        session['runs'].append(result)
        evaluation = result.get('evaluation_label', '테스트')
        lines = [f'분석을 완료했습니다. 아래 수치는 Python이 계산한 **{evaluation} 결과**입니다.']
        for model in result['models']:
            m = next(m for m in model['metrics'] if m['split']==evaluation)
            lines.append(f"• {m['model']}: R² {m['R2']:.4f}, RMSE {m['RMSE']:.4f}, MAE {m['MAE']:.4f}")
            if m['negative_predictions'] and dataset is None:
                lines.append(f"  음수 예측 {m['negative_predictions']}개: 저항의 물리적 의미에 맞지 않는 예측입니다.")
        if action=='compare':
            lines.append(f"{'교차검증' if result.get('small_sample') else '검증'} RMSE로 선택한 모델: **{analysis.NAMES[result['selected']]}**.")
        if action=='epoch_compare':
            lines.append('50은 비교할 상한이며 정답으로 지정된 값이 아닙니다. 두 후보의 검증 RMSE를 비교했습니다.')
            for row in result['epoch_candidates']:
                lines.append(f"상한 {row['max_epochs']} / 실행 {row['executed_epochs']} / 채택 {row['best_epoch']} epoch: 검증 RMSE {row['validation_RMSE']:.4f}")
            lines.append(f"선택한 상한: {result['selected_max_epochs']}. 선택 후 이 모델만 테스트 평가했습니다.")
        reply = '\n\n'.join(lines)
    elif action in ('last','report','errors'):
        if not session['runs']:
            reply = '저장된 분석 결과가 없습니다. 먼저 “선형회귀 해줘” 또는 “모델 3가지 비교해줘”라고 입력하세요.'
        else:
            result = session['runs'][-1]
            reply = (f"{result['time']}의 분석 결과를 불러왔습니다. 모델을 새로 학습하지 않았습니다."
                     if action=='last' else '실제 분석 기록을 보고서로 정리했습니다. 아래 대화에서 확인하세요.')
            if action=='errors':
                point = analysis.worst_errors(result, 1)[0]
                reply = (f"선택 모델의 테스트 예측을 절대오차 순으로 정렬했습니다. 가장 큰 오차는 원자료 {point['row']}행: "
                         f"실제 {point['actual']:.4f}, 예측 {point['predicted']:.4f}, 절대오차 {point['absolute_error']:.4f}입니다. "
                         '아래 표에서 10개를 확인하세요. R²는 정답률이 아니며 개별 예측의 정확성을 보장하지 않습니다.')
                reply = reply.replace('테스트 예측', f"{result.get('evaluation_label', '테스트')} 예측")
    elif action == 'predict':
        previous = session['runs'][-1] if session['runs'] else None
        prediction = analysis.predict_custom(plan['values'], previous, dataset) if dataset is not None else analysis.predict(plan['values'], previous)
        reply = (f"{prediction['model']} 예측: {prediction['target']} = {prediction['value']:.6f}\n\n"
                 f"입력: {prediction['inputs']}\n\n" + prediction['warning'])
        if prediction['outside_training_range']:
            reply += '\n\n학습 범위를 벗어난 입력: ' + ', '.join(prediction['outside_training_range'])
        if prediction['value'] < 0 and dataset is None:
            reply += '\n\n음수 저항은 물리적으로 해석할 수 없으므로 이 예측을 사용하지 마세요.'
        evidence['prediction'] = prediction
    message = {'role': 'assistant', 'content': reply, 'action': action}
    if result is not None:
        message['run_id'] = result.get('id')
    if action == 'predict':
        message['prediction'] = prediction
    session['messages'].extend([{'role':'user','content':text}, message])
    evidence.update({'time':memory.now(), 'action':action, 'prompt':text, 'reply':reply})
    session['events'].append(evidence)
    if action == 'report' and session['runs']:
        message['report'] = memory.report(session)
    # 과거 대화는 200개 메시지까지 저장. 분석 기록은 별도로 보존.
    session['messages'] = session['messages'][-200:]
    session['events'] = session['events'][-100:]
    memory.save(session)
    return {'reply':reply,'action':action,'result':result,'evidence':evidence}
