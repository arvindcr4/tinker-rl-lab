#!/usr/bin/env python3
import json
import sys
import glob
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def sign_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Generate key if we don't have one, or just generate a new one for signing
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    
    # Sign the content (without signature field)
    content_to_sign = json.dumps(data, sort_keys=True).encode()
    signature = private_key.sign(content_to_sign)
    
    # In a real scenario, we'd save the public key somewhere.
    data['signature'] = {
        'algorithm': 'ed25519',
        'pubkey': public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        ).hex(),
        'sig': signature.hex()
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Signed {filepath}")

for path in glob.glob('registry/provenance/*.provenance.json'):
    sign_file(path)
