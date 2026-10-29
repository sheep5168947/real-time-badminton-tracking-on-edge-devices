"""Fast local push guard. Does not run inference, training, or network calls."""
import ast
import platform
import re
import subprocess
import sys

def allowed_ref(ref):
    return bool(re.fullmatch(r'refs/heads/(feat|fix|docs|refactor|perf|test|chore|build|ci|style)/[a-z0-9][a-z0-9._/-]*',ref))

def main():
    if len(sys.argv)>1 and sys.argv[1]=='--self-test':
        assert allowed_ref('refs/heads/fix/nano-input')
        assert allowed_ref('refs/heads/chore/establish-thesis-version-control')
        assert not allowed_ref('refs/heads/main')
        assert not allowed_ref('refs/heads/deploy/jetson-nano')
        assert not allowed_ref('refs/heads/backup/snapshot')
        print('Push branch checks passed.')
        return
    count=0
    for line in sys.stdin:
        local_ref,local_sha,remote_ref,remote_sha=line.split()
        if set(local_sha)=={'0'}:
            raise SystemExit('Branch deletion must be handled explicitly, not by this workflow hook.')
        if not allowed_ref(remote_ref):
            raise SystemExit('Push a feat/fix/docs/refactor/perf/test/chore/build/ci/style task branch; merge main/deploy through a PR.')
        base=remote_sha
        if set(base)=={'0'}:
            base=subprocess.check_output(['git','merge-base',local_sha,'origin/main'],universal_newlines=True).strip()
        names=subprocess.check_output(['git','diff','--name-only','--diff-filter=ACMR',base,local_sha],universal_newlines=True).splitlines()
        nano=platform.machine().lower() in ('aarch64','arm64') or sys.version_info<(3,10)
        for name in names:
            if not name.endswith('.py'):
                continue
            if nano and not (name.startswith('thesis/platforms/jetson_nano/') or name in ['scripts/check_push.py','scripts/check_commit_message.py','scripts/install_git_hooks.py']):
                continue
            source=subprocess.check_output(['git','show',local_sha+':'+name]).decode('utf-8-sig')
            ast.parse(source,filename=name)
            count+=1
    print('Push syntax checks passed: '+str(count)+' Python files.')

if __name__=='__main__':
    main()
