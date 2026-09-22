"""Offline-testable fail-closed transport and durable host lifecycle."""
import hashlib,json,os,sqlite3,threading,time,uuid
from decimal import Decimal
from pathlib import Path
from http.client import HTTPConnection

MAX_BODY=900*1024 # Modal queue item limit is 1MiB; leave envelope headroom
MAX_REQUESTS=543522 # E13 logical ceiling plus two runtime smokes
class Halted(RuntimeError): pass

def need(ok,message):
    if not ok: raise Halted(message)

def sha(data): return hashlib.sha256(data).hexdigest()

def campaign_cap(value,mode=None):
    if value is None:
        need(mode=='unlimited','unlimited campaign requires explicit mode')
        return Decimal('Infinity')
    cap=Decimal(str(value));need(cap==75,'unsupported finite campaign cap')
    return cap

def remainder_matches(value,cap,total):
    return value is None if cap.is_infinite() else Decimal(str(value))==cap-total

def validate_profile(r,now=None):
    now=time.time() if now is None else now
    need(r['status']=='RESERVED' and r['provider']=='modal' and r['unit']=='USD','real USD Modal reservation required')
    need(r['reserved_units']==8 and r['max_wall_seconds']==3600,'exact $8/3600s profile required')
    need(0<r['max_hourly_units']<=8 and 2<=r['max_requests']<=MAX_REQUESTS,'rate/request ceiling')
    cap=campaign_cap(r['authorization_cap_usd'],r.get('budget_mode')); total=Decimal(str(r['cumulative_reserved_and_spent_usd']))
    need(total.is_finite() and 0<=total<=cap,'cumulative campaign authorization')
    need(r['expires_epoch']>now+3600 and r['reservation_id'],'reservation expires too early')
    need(r['ledger_ref']['sha256'] and r['authorization_ref']['sha256'],'ledger and authorization refs required')

def verify_refs(r):
    for key in ('ledger_ref','authorization_ref'):
        ref=r[key]; need(sha(Path(ref['path']).read_bytes())==ref['sha256'],key+' changed')
    auth=json.loads(Path(r['authorization_ref']['path']).read_text())
    cap=campaign_cap(auth['total_cap_usd'],auth.get('budget_mode'))
    need(cap==campaign_cap(r.get('authorization_cap_usd',75),r.get('budget_mode')),'authorization cap mismatch')
    ledger=json.loads(Path(r['ledger_ref']['path']).read_text())
    need(ledger['status']=='RESERVED_NOT_DISPATCHED','reservation ledger already dispatched or invalid')
    need(ledger['authorization']['sha256']==r['authorization_ref']['sha256'] and Path(ledger['authorization']['path']).resolve()==Path(r['authorization_ref']['path']).resolve(),'ledger authorization binding mismatch')
    rows=ledger['reservations'];need(len({x['id'] for x in rows})==len(rows),'duplicate ledger reservation ids')
    selected=[x for x in rows if x['id']==r['reservation_id']];need(len(selected)==1,'actual reservation id absent')
    row=selected[0];scope=row['scope']
    need(scope in {'One bounded exact BF16 actor session; E1 native execution','One bounded exact BF16 actor session; E13 native execution','One bounded exact BF16 actor session; E5 native execution'},'unsupported native execution scope')
    need(Decimal(str(row['maximum_usd']))==Decimal(str(r['reserved_units']))==8,'ledger reservation amount mismatch')
    need(row['maximum_wall_seconds']==r['max_wall_seconds']==3600 and row['provider']==r['provider']=='modal','ledger wall/provider mismatch')
    need(row['scope']==scope and r.get('scope',scope)==scope,'ledger reservation scope mismatch')
    continuation_ref=ledger.get('continuation_ledger_ref')
    if continuation_ref:
        cp=Path(continuation_ref['path']);need(sha(cp.read_bytes())==continuation_ref['sha256'],'continuation ledger changed')
        continuation=json.loads(cp.read_text())
        need(continuation['status']=='RESERVED' and campaign_cap(continuation['cumulative_cap_usd'],continuation.get('budget_mode'))==cap,'continuation not reserved under75')
        need(continuation['authorization_ref']==r['authorization_ref'],'continuation authorization mismatch')
        analytical=continuation['analytical_reconciliation_ref'];need(sha(Path(analytical['path']).read_bytes())==analytical['sha256'],'analytical reconciliation changed')
        accounting=json.loads(Path(analytical['path']).read_text())
        need(accounting['invoice_verified_actual_charge_usd'] is None and accounting['historical_ledgers_mutated'] is False,'analytical basis boundary changed')
        allocations=continuation['allocations'];need(len({x['id'] for x in allocations})==len(allocations),'duplicate continuation ids')
        amounts=[Decimal(str(x['reserved_usd'])) for x in allocations]
        need(all(x.is_finite() and x>=0 for x in amounts),'invalid continuation amount')
        selected=[x for x in allocations if x['id']==r['reservation_id']]
        need(len(selected)==1 and Decimal(str(selected[0]['reserved_usd']))==8,'actor allocation absent from continuation')
        cumulative=sum(amounts)
        need(cumulative==Decimal(str(continuation['cumulative_counted_and_reserved_usd']))<=cap,'continuation math mismatch')
        need(remainder_matches(continuation['remaining_unreserved_usd'],cap,cumulative),'continuation remainder mismatch')
        prior=cumulative-8
        need(len(rows)==1,'continuation-bound actor wrapper must contain only current actor; all other allocations are prior')
    else:
        prior=Decimal(str(auth['prior_counted_and_reserved_usd']))
    need(Decimal(str(ledger['prior_counted_and_reserved_usd']))==prior,'ledger prior balance drift')
    amounts=[Decimal(str(x['maximum_usd'])) for x in rows]
    need(all(x.is_finite() and x>0 for x in amounts),'invalid reservation amount')
    total=prior+sum(amounts)
    need(total==Decimal(str(ledger['cumulative_counted_and_reserved_usd']))==Decimal(str(r['cumulative_reserved_and_spent_usd']))<=cap,'cumulative reservation total mismatch')
    need(campaign_cap(ledger['cumulative_cap_usd'],ledger.get('budget_mode'))==cap and remainder_matches(ledger['remaining_after_reservations_usd'],cap,total),'ledger cap/remainder mismatch')

def write_new(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

class Ledger:
    def __init__(self,directory,limit,deadline,clock=time.time):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.db=sqlite3.connect(self.directory/'ledger.sqlite',check_same_thread=False)
        self.db.execute('PRAGMA synchronous=FULL');self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE attempts(id TEXT PRIMARY KEY,state TEXT,request_sha TEXT,response_sha TEXT)')
        self.lock=threading.RLock();self.limit=limit;self.deadline=deadline;self.clock=clock;self.state='NEW';self.count=0;self.submitted=set();self.on_halt=None;self.halt_notified=False
    def start(self):
        with self.lock:
            need(self.state=='NEW','session cannot restart');self.state='ACTIVE'
            write_new(self.directory/'launch-intent.json',json.dumps({'at':self.clock(),'deadline':self.deadline}).encode())
    def halt(self,reason):
        with self.lock:
            self.state='HALTED'
            notify=self.on_halt if not self.halt_notified else None
            self.halt_notified=True
        # Signal remote stop even if writing halt.json subsequently fails.
        if notify is not None:notify()
        with self.lock:
            p=self.directory/'halt.json'
            if not p.exists():write_new(p,json.dumps({'at':self.clock(),'reason':str(reason)}).encode())
    def submit(self,rid,enqueue):
        with self.lock:
            need(self.state=='ACTIVE' and self.clock()<self.deadline,'host halted before enqueue')
            need(rid not in self.submitted and self.db.execute('SELECT state FROM attempts WHERE id=?',(rid,)).fetchone()==('INTENT',),'unknown/duplicate queue submission')
            self.submitted.add(rid)
            try:return enqueue()
            except BaseException:
                self.state='HALTED';raise
    def admit(self,body):
        with self.lock:
            need(self.state=='ACTIVE' and self.clock()<self.deadline,'session halted or expired')
            need(self.count<self.limit,'request limit reached');need(type(body) is bytes and len(body)<=MAX_BODY,'body size/type')
            rid=uuid.uuid4().hex;write_new(self.directory/(rid+'.request'),body)
            self.db.execute('INSERT INTO attempts VALUES(?,?,?,NULL)',(rid,'INTENT',sha(body)));self.db.commit();self.count+=1
            return rid
    def complete(self,rid,response):
        with self.lock:
            need(self.state=='ACTIVE','outcome arrived after halt')
            need(type(response['body']) is bytes and len(response['body'])<=MAX_BODY,'response bounds')
            need(self.db.execute('SELECT state FROM attempts WHERE id=?',(rid,)).fetchone()==('INTENT',),'duplicate/unknown response')
            write_new(self.directory/(rid+'.response'),response['body'])
            write_new(self.directory/(rid+'.response-meta.json'),json.dumps({k:v for k,v in response.items() if k!='body'}).encode())
            self.db.execute('UPDATE attempts SET state=?,response_sha=? WHERE id=?',('RETURNED',sha(response['body']),rid));self.db.commit()

class Relay:
    def __init__(self,ledger,dispatch):self.ledger=ledger;self.dispatch=dispatch
    def request(self,body,headers):
        rid=self.ledger.admit(body)
        try:
            response=self.dispatch({'id':rid,'body':body,'headers':headers})
            need(response['id']==rid and not response.get('error'),'uncertain/mismatched remote outcome')
            self.ledger.complete(rid,response);return response
        except BaseException as e:self.ledger.halt(e);raise

class WorkerGuard:
    def __init__(self,limit,deadline):self.limit=limit;self.deadline=deadline;self.seen=set();self.dispatched=set();self.lock=threading.RLock();self.halted=False
    def halt(self):
        with self.lock:self.halted=True
    def admit(self,rid):
        with self.lock:
            need(not self.halted and time.time()<self.deadline,'worker halted/expired')
            if rid in self.seen or len(self.seen)>=self.limit:
                self.halted=True;raise Halted('duplicate request or budget exhausted')
            self.seen.add(rid)
    def send_start(self,rid,send):
        # Only the actual socket send is locked, not model inference/response read.
        # halt() shares this lock: queued/admitted work cannot send after halt.
        with self.lock:
            need(not self.halted and time.time()<self.deadline,'worker halted before send')
            need(rid in self.seen and rid not in self.dispatched,'unadmitted/duplicate dispatch')
            self.dispatched.add(rid)
            try:return send()
            except BaseException:
                self.halted=True;raise

# Preserve payload bytes, duplicate response headers and status. HTTP framing is
# recreated at each hop; request/response JSON is never parsed or reserialized.
def forward_raw(port,item,timeout,guard=None):
    need(type(item['body']) is bytes and len(item['body'])<=MAX_BODY,'request bound')
    conn=HTTPConnection('127.0.0.1',port,timeout=timeout)
    try:
        conn.putrequest('POST','/v1/chat/completions',skip_accept_encoding=True)
        for k,v in item['headers']:
            if k.lower() not in {'host','content-length','connection','transfer-encoding','authorization'}:conn.putheader(k,v)
        conn.putheader('Content-Length',str(len(item['body'])))
        conn.connect() # Establish TCP without sending any request bytes.
        if guard is None:conn.endheaders(item['body'])
        else:guard.send_start(item['id'],lambda:conn.endheaders(item['body']))
        res=conn.getresponse();headers=res.getheaders()
        lengths=[v for k,v in headers if k.lower()=='content-length']
        transfers=[v for k,v in headers if k.lower()=='transfer-encoding']
        need(len(lengths)<=1,'duplicate Content-Length is ambiguous')
        need(not(lengths and transfers),'Content-Length plus Transfer-Encoding is ambiguous')
        need(not transfers or (len(transfers)==1 and transfers[0].strip().lower()=='chunked'),'unsupported transfer framing')
        declared=None
        if lengths:
            value=lengths[0].strip();need(value.isascii() and value.isdecimal(),'invalid Content-Length')
            declared=int(value);need(declared<=MAX_BODY,'declared response exceeds bound')
        body=res.read(MAX_BODY+1);need(len(body)<=MAX_BODY,'response bound')
        need(declared is None or len(body)==declared,'truncated Content-Length response')
        return {'id':item['id'],'status':res.status,'reason':res.reason,'headers':res.getheaders(),'body':body}
    except BaseException:
        if guard is not None:guard.halt()
        raise
    finally:conn.close()


def validate_endpoint(endpoint,port,proof=None,now=None):
    from urllib.parse import urlsplit
    now=time.time() if now is None else now
    u=urlsplit(endpoint)
    need(u.scheme=='http' and u.path=='/v1' and not u.username and not u.password and not u.query and not u.fragment,'exact credential-free endpoint required')
    need(u.hostname in {'127.0.0.1','host.lima.internal'} and u.port==port,'only same-port loopback/Colima endpoint permitted')
    if u.hostname=='host.lima.internal':
        need(isinstance(proof,dict),'actual Colima connection proof required')
        if 'native_endpoint' in proof:
            need(proof.get('native_endpoint')==endpoint and proof.get('host_bind')==f'127.0.0.1:{port}','root connectivity address mismatch')
            need(proof.get('matched') is True and proof.get('exit_code')==0 and proof.get('error')=='' and proof.get('model_calls')==0,'root connectivity probe failed')
            return endpoint
        need(proof.get('endpoint')==endpoint and proof.get('host_bind')=='127.0.0.1','connection proof address mismatch')
        need(proof.get('verified_from')=='colima_container' and proof.get('connection_verified') is True,'proof must come from actual native container')
        need(0<=now-proof.get('verified_at_epoch',0)<=300,'stale/future connectivity proof')
        nonce=proof.get('expected_nonce','')
        need(isinstance(nonce,str) and len(nonce)>=32 and proof.get('observed_nonce')==nonce,'round-trip nonce proof required')
    return endpoint


class StopSignal:
    """One prompt best-effort stop and cancel, independent of server/response waits."""
    def __init__(self,control,cancel):self.control=control;self.cancel=cancel;self.lock=threading.Lock();self.sent=False;self.threads=[]
    def request(self):
        with self.lock:
            if self.sent:return
            self.sent=True
            def attempt(fn):
                try:fn()
                except BaseException:pass # Unknown stop never admits retry/replay.
            for fn in (self.control,self.cancel):
                t=threading.Thread(target=attempt,args=(fn,),daemon=True);self.threads.append(t);t.start()
