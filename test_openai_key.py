#!/usr/bin/env python3
import os

# Load .env
env_file = '.env'
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip()

key = os.getenv('OPENAI_API_KEY')
if not key:
    print('ERROR: No OPENAI_API_KEY found in .env')
    exit(1)

if key.startswith('"') or key.startswith("'"):
    print('ERROR: Key has quotes (should be removed)')
    exit(1)

print(f'✓ OPENAI_API_KEY found: {key[:20]}...')

try:
    from src.genai import generate_draft_response
    print('✓ Calling OpenAI LLM...')
    resp = generate_draft_response(
        complaint_text='Money deducted from ATM but cash not received.',
        predicted_category='Cards/ATM',
        urgency='High',
        provider='openai'
    )
    print('\n✓ LLM_SUCCESS\n')
    print('Response preview:')
    print(resp[:600])
except Exception as e:
    print(f'✗ LLM_ERROR: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
