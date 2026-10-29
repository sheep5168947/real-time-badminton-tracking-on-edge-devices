"""Enable the versioned hooks for this clone only."""
import subprocess
import sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    subprocess.check_call(['git','-C',str(root),'config','core.hooksPath','.githooks'])
    subprocess.check_call(['git','-C',str(root),'config','thesis.python',sys.executable.replace(chr(92),'/')])
    print('Enabled .githooks for this repository. See CONTRIBUTING.md.')

if __name__=='__main__':
    main()
