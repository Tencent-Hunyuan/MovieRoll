"""Keep the file:// catalog in sync with the editable JSON catalog."""
import json
from pathlib import Path

def build():
    root=Path(__file__).resolve().parents[1]
    data=json.loads((root/'content.json').read_text(encoding='utf-8'))
    (root/'content-data.js').write_text('// Generated from content.json by scripts/build_offline_data.py.\nwindow.MOVIEROLL_CONTENT = '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')

if __name__=='__main__':
    build()
    print('Offline catalog updated.')
