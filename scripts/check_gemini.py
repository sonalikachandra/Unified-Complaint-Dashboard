import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load .env
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_path):
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

print('GEMINI_KEY_PRESENT=' + str(bool(os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'))))

try:
    from src.genai import generate_draft_response, get_last_provider_status
    print('Calling generate_draft_response(provider=gemini)')
    out = generate_draft_response(
        complaint_text='Money deducted from ATM but cash not received.',
        predicted_category='Cards/ATM',
        urgency='High',
        provider='gemini'
    )
    status = get_last_provider_status()
    print('STATUS:', status)
    print('SUCCESS:' if status.get('used') == 'gemini' else 'FALLBACK:')
    print(out[:1000])
except Exception as e:
    print('ERROR:', type(e).__name__, str(e))
    import traceback
    traceback.print_exc()
