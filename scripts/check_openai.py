import os

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

print('HAS_KEY' if os.getenv('OPENAI_API_KEY') else 'NO_KEY')

try:
    from src.genai import generate_draft_response
    out = generate_draft_response(
        complaint_text='Money deducted from ATM but cash not received.',
        predicted_category='Cards/ATM',
        urgency='High',
        provider='openai'
    )
    print('LLM_OK')
    print(out[:800])
except Exception as e:
    print('LLM_ERROR', type(e).__name__, str(e))
