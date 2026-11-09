import hashlib
import pytest
from llm_lab import cloud_client as c
from llm_lab.client import LLMError

def test_busy_request_does_not_load(monkeypatch):
    monkeypatch.setattr(c,'_load_locked',lambda: pytest.fail('must not load'))
    with c._LOCK:
        with pytest.raises(LLMError):c.LlamaCppClient().plan('안녕',[],{})

def test_input_limit_before_load(monkeypatch):
    monkeypatch.setattr(c,'_load_locked',lambda: pytest.fail('must not load'))
    with pytest.raises(ValueError):c.LlamaCppClient().plan('가'*1201,[],{})

def test_download_integrity(tmp_path,monkeypatch):
    from io import BytesIO
    monkeypatch.setenv('LLM_MODEL_DIR',str(tmp_path))
    monkeypatch.setattr(c,'MODEL_BYTES',4);monkeypatch.setattr(c,'MODEL_SHA256',hashlib.sha256(b'good').hexdigest())
    monkeypatch.setattr(c,'urlopen',lambda *a,**k:BytesIO(b'bad!'))
    with pytest.raises(LLMError):c.ensure_model()
    assert not c.model_path().exists()
    assert not c.model_path().with_suffix('.part').exists()

def test_verified_download_reused(tmp_path,monkeypatch):
    monkeypatch.setenv('LLM_MODEL_DIR',str(tmp_path))
    monkeypatch.setattr(c,'MODEL_BYTES',4);monkeypatch.setattr(c,'MODEL_SHA256',hashlib.sha256(b'good').hexdigest())
    c.model_path().write_bytes(b'good')
    monkeypatch.setattr(c,'urlopen',lambda *a,**k:pytest.fail('must not download'))
    assert c.ensure_model()==c.model_path()

def test_history_bounded_and_current_retained():
    class Tokens:
        def tokenize(self,text,**kw):return list(range(len(text)//4))
    history=[{'role':'user','content':'z'*600},{'role':'assistant','content':'z'*600}]*10
    out=c.build_chat_messages(Tokens(),'현재 요청',history)
    assert out[-1]['content']=='현재 요청'
    assert len(out)<=8

def test_router_ignores_old_instructions():
    class Tokens:
        def tokenize(self,text,**kw):return list(range(len(text)//4))
    out=c.build_messages(Tokens(),'현재 요청',[{'role':'user','content':'OLD_INSTRUCTION'}],{})
    assert len(out)==2 and 'OLD_INSTRUCTION' not in str(out)
