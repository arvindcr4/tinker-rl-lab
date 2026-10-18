"""Offline trust-boundary and cost regressions; no native execution."""
import base64
import hashlib
import pytest
from flagship import pavlov_verilog_eval_adapter as adapter
from flagship import pavlov_verilog_eval_receipt as receipt
from flagship import pavlov_verilog_eval_split_manifest as split
from flagship import pavlov_webbench_eval_adapter as web

@pytest.mark.parametrize('module', [adapter, receipt])
@pytest.mark.parametrize('amount', [float('nan'), float('inf'), -float('inf'), True, -1])
def test_verilog_costs_reject_invalid_numbers(module, amount):
    with pytest.raises(ValueError):
        module._validate_costs({'status':'observed','total_usd':amount})

@pytest.mark.parametrize('module', [adapter, receipt])
def test_verilog_zero_cost_is_valid(module):
    assert module._validate_costs({'status':'observed','total_usd':0})['total_usd']==0

def test_arbitrary_valid_provider_key_is_not_trusted():
    key=b'x'*32
    root={'schema_version':web.WEBBENCH_TRUST_ROOT_SCHEMA_VERSION,'provider_id':web.WEBBENCH_PROVIDER_ID,
          'lane':web.WEBBENCH_GRANT_LANE,'suite_id':web.WEBBENCH_SUITE_ID,'key_id':'attacker',
          'key_fingerprint':hashlib.sha256(key).hexdigest(),'public_key_b64':base64.b64encode(key).decode(),
          'signature_algorithm':'ed25519','issued_at':'2020-01-01T00:00:00Z','expires_at':'2099-01-01T00:00:00Z'}
    with pytest.raises(web.WebBenchDatasetError, match='trust anchor'):
        web.load_webbench_trust_root(root)

def test_checksum_and_true_flags_are_not_signature():
    binding={'test':'binding'}
    payload={'identity':'a'*40,'artifact_digest':'b'*64}
    record={'receipt_id':'c'*40,'digest':web.receipt_digest(binding,payload),'authenticated':True,
            'cryptographically_bound':True,'binding':binding,'payload':payload}
    assert web._receipt(record,binding)==(False,'receipt_signer_not_trusted')

def test_revision_override_rejected_before_checkout_access(tmp_path):
    with pytest.raises(split.VerilogEvalSplitManifestError,match='unapproved benchmark'):
        split.build_manifest_from_checkout(tmp_path,revision='a'*40)

def test_decontamination_string_retains_assertion_status():
    assert receipt._validate_decontamination('a'*40)['status']=='asserted_unverified'
