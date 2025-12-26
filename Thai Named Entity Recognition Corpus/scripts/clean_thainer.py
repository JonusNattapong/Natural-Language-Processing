import json
from pathlib import Path

dirpath = Path(__file__).resolve().parents[1]
input_path = dirpath / 'data' / 'ThaiNER.jsonl'
backup_path = dirpath / 'data' / 'ThaiNER_backup_pre_clean.jsonl'
clean_path = dirpath / 'data' / 'ThaiNER_clean.jsonl'
keep = {'id','domain','tokens','tags'}

print(f'Reading: {input_path}')
if not input_path.exists():
    raise SystemExit('Input file not found')

# scan and clean
total = 0
modified = 0
with input_path.open('r', encoding='utf-8') as inf, clean_path.open('w', encoding='utf-8') as outf:
    for line in inf:
        total += 1
        obj = json.loads(line)
        keys = set(obj.keys())
        if not keys <= keep:
            modified += 1
        new = {k: obj.get(k) for k in keep if k in obj}
        outf.write(json.dumps(new, ensure_ascii=False) + '\n')

print(f'Total lines: {total}, Modified lines: {modified}')

# backup original and replace
import shutil
shutil.copy2(input_path, backup_path)
shutil.move(str(clean_path), str(input_path))
print(f'Backup saved to: {backup_path}')
print(f'Cleaned file written to: {input_path}')
print('Done')