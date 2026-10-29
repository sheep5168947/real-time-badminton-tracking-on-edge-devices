"""Validate new Conventional Commit subjects without external dependencies."""
import re
import sys
from pathlib import Path

PATTERN=re.compile(r"^(feat|fix|docs|refactor|perf|test|chore|build|ci|style|revert)(\([a-z0-9_./-]+\))?!?: \S.*$")

def valid_subject(subject):
    return bool(PATTERN.fullmatch(subject)) and len(subject)<=100 and subject==subject.rstrip()

def self_test():
    for subject in ['feat(roi): add recovery log','fix: correct path','refactor(pc)!: move entry points','docs: 補充評估規則']:
        assert valid_subject(subject),subject
    for subject in ['backup','update code','fix: ','Fix: wrong case','fix: trailing space ','docs: '+('x'*100)]:
        assert not valid_subject(subject),subject

def main():
    if len(sys.argv)==2 and sys.argv[1]=='--self-test':
        self_test()
        print('Commit message checks passed.')
        return
    if len(sys.argv)!=2:
        raise SystemExit('Usage: check_commit_message.py COMMIT_EDITMSG')
    lines=Path(sys.argv[1]).read_text(encoding='utf-8-sig').splitlines()
    subject=next((line for line in lines if line and not line.startswith('#')),'')
    if subject.startswith('Merge pull request ') or subject.startswith("Merge branch "):
        return
    if not valid_subject(subject):
        raise SystemExit('Use type(scope): description, e.g. fix(nano): correct engine path (max 100 characters).')

if __name__=='__main__':
    main()
