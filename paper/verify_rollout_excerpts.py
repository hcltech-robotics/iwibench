from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
m=json.loads((root/'rollout-excerpts/manifest.json').read_text())
for e in m['files']:
 p=root/e['file']
 assert p.is_file(),e['file']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==e['sha256'], f"Protected excerpt changed: {e['file']}"
print(f"Verified {len(m['files'])} protected rollout payloads byte-for-byte against the recorded hashes.")
