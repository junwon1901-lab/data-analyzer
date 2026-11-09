"""무료 CPU 서버용 GGUF 추론. 유료 API나 Ollama 서버 없이 실행한다."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen
from llm_lab.client import LLMError, SYSTEM, SCHEMA, validate_plan

MODEL_REPO='Qwen/Qwen2.5-1.5B-Instruct-GGUF'
MODEL_REVISION='91cad51170dc346986eccefdc2dd33a9da36ead9'
MODEL_FILENAME='qwen2.5-1.5b-instruct-q4_k_m.gguf'
MODEL_SHA256='6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e'
MODEL_BYTES=1117320736
MODEL_URL=f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILENAME}'
CONTEXT_TOKENS=4096
MAX_OUTPUT_TOKENS=256
MAX_INPUT_CHARS=1200
_LOCK=threading.Lock()
_MODEL=None

def model_path():
    return Path(os.environ.get('LLM_MODEL_DIR',str(Path(__file__).resolve().parents[1]/'.models')))/MODEL_FILENAME

def _digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def ensure_model():
    path=model_path()
    if path.exists() and path.stat().st_size==MODEL_BYTES and _digest(path)==MODEL_SHA256:return path
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix('.part')
    try:
        digest=hashlib.sha256();size=0
        with urlopen(Request(MODEL_URL,headers={'User-Agent':'HullLLMAssignment/2.0'}),timeout=120) as source,partial.open('wb') as dest:
            while chunk:=source.read(1024*1024):
                size+=len(chunk)
                if size>MODEL_BYTES:raise ValueError('모델 파일 크기 불일치')
                digest.update(chunk);dest.write(chunk)
        if size!=MODEL_BYTES or digest.hexdigest()!=MODEL_SHA256:raise ValueError('모델 무결성 검사 실패')
        partial.replace(path)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise LLMError('모델 다운로드·검사에 실패했습니다. 잠시 뒤 다시 시도하세요. 분석은 실행하지 않았습니다.') from exc
    return path

def _load_locked():
    global _MODEL
    if _MODEL is None:
        try:
            from llama_cpp import Llama
            path=ensure_model()
            _MODEL=Llama(model_path=str(path),n_ctx=CONTEXT_TOKENS,n_threads=2,n_threads_batch=2,
                         n_batch=128,n_ubatch=128,n_gpu_layers=0,use_mmap=True,verbose=False,chat_format='chatml',seed=42)
        except LLMError:raise
        except (ImportError,OSError,ValueError,RuntimeError) as exc:
            raise LLMError('CPU 모델을 불러오지 못했습니다. 패키지 설치 로그와 서버 메모리를 확인하세요.') from exc
    return _MODEL

def _token_count(llm,messages):
    text=''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)+'<|im_start|>assistant\n'
    return len(llm.tokenize(text.encode(),add_bos=False,special=True))

def build_messages(llm,text,history,context):
    last=context.get('last_run') or {}
    saved={'saved_runs':context.get('saved_runs',0),'last_action':last.get('action'),'selected_model':last.get('selected')}
    messages=[{'role':'system','content':SYSTEM+'\nSaved context: '+json.dumps(saved)},{'role':'user','content':text}]
    if _token_count(llm,messages)>CONTEXT_TOKENS-MAX_OUTPUT_TOKENS-32:raise ValueError('질문을 짧게 나누어 입력하세요.')
    return messages

def build_chat_messages(llm,text,history):
    system={'role':'system','content':
        'You are a helpful assistant. Reply in Korean in one or two sentences, under 120 characters. '
        'The user and assistant are different people. When the USER asks "내 이름", it means the USER name, '
        'NOT the assistant name. Read the previous USER messages to answer. If the user has not told you '
        'their name, say you do not know it. Never introduce yourself with the user name. '
        'Explain concepts briefly. Do not claim to run functions or invent numerical results. '
        'This program predicts normalized sailboat hull resistance using Fn,Rn,Cp,Cb,Cw,L/B,B/T,LCB/L.'}
    recent=[{'role':m['role'],'content':str(m.get('content',''))[:240]} for m in history[-6:] if m.get('role') in ('user','assistant')]
    current={'role':'user','content':text}
    while recent and _token_count(llm,[system]+recent+[current])>CONTEXT_TOKENS-MAX_OUTPUT_TOKENS-32:recent=recent[2:]
    messages=[system]+recent+[current]
    if _token_count(llm,messages)>CONTEXT_TOKENS-MAX_OUTPUT_TOKENS-32:raise ValueError('질문을 짧게 나누어 입력하세요.')
    return messages

class LlamaCppClient:
    model='Qwen2.5-1.5B-Instruct / Q4_K_M / CPU'
    def status(self):
        path=model_path()
        return {'models':[self.model],'backend':'llama-cpp-python','loaded':_MODEL is not None,
                'downloaded':path.exists() and path.stat().st_size==MODEL_BYTES}
    def prepare(self):
        if not _LOCK.acquire(blocking=False):raise LLMError('다른 요청을 처리 중입니다. 잠시 뒤 다시 눌러 주세요.')
        try:_load_locked();return self.status()
        finally:_LOCK.release()
    def plan(self,text,messages,context):
        if not isinstance(text,str) or not text.strip() or len(text)>MAX_INPUT_CHARS:raise ValueError('질문은 1~1,200자로 입력하세요.')
        if not _LOCK.acquire(blocking=False):raise LLMError('다른 요청을 처리 중입니다. 잠시 뒤 다시 입력해 주세요.')
        start=time.perf_counter()
        try:
            llm=_load_locked();prompt=build_messages(llm,text,messages,context)
            llm.reset()
            raw=llm.create_chat_completion(messages=prompt,response_format={'type':'json_object','schema':SCHEMA},temperature=0,max_tokens=MAX_OUTPUT_TOKENS)
            if raw['choices'][0].get('finish_reason')!='stop':raise LLMError('응답이 길어 중단되었습니다. 분석은 실행하지 않았습니다.')
            content=raw['choices'][0]['message']['content'];plan=json.loads(content);validate_plan(plan)
            chat_text=None;count=raw.get('usage',{}).get('completion_tokens',0)
            if plan['action']=='chat':
                llm.reset()
                chat=llm.create_chat_completion(messages=build_chat_messages(llm,text,messages),temperature=0,max_tokens=192)
                if chat['choices'][0].get('finish_reason')!='stop':raise LLMError('대화 답변이 길어 중단되었습니다. 짧게 질문해 주세요.')
                chat_text=chat['choices'][0]['message']['content'].strip();plan['reply']=chat_text;validate_plan(plan)
                count+=chat.get('usage',{}).get('completion_tokens',0)
            return plan,{'model':self.model,'backend':'llama-cpp-python','seconds':round(time.perf_counter()-start,3),
                         'eval_count':count,'prompt_tokens':raw.get('usage',{}).get('prompt_tokens'),
                         'chat_raw_text':chat_text,'model_sha256':MODEL_SHA256,'raw_json':content}
        except LLMError:raise
        except (KeyError,TypeError,ValueError,RuntimeError) as exc:
            raise LLMError('명령을 해석하지 못했습니다. 짧고 구체적으로 다시 입력하세요. 분석은 실행하지 않았습니다.') from exc
        finally:_LOCK.release()
