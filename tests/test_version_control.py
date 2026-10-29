from scripts.check_commit_message import self_test as message_self_test
from scripts.check_push import allowed_ref

def test_commit_message_convention():
    message_self_test()

def test_task_branches_and_protected_targets():
    assert allowed_ref('refs/heads/fix/nano-input')
    assert allowed_ref('refs/heads/docs/update-evaluation')
    assert not allowed_ref('refs/heads/main')
    assert not allowed_ref('refs/heads/deploy/jetson-nano')
    assert not allowed_ref('refs/heads/backup/snapshot')
