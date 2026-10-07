"""evals/local_env.py use_local_chroma_dir() 단위 테스트.

LLM·네트워크 없음. os.environ을 건드리므로 매 테스트 후 원상복구한다."""
import os

import pytest

from evals.local_env import use_local_chroma_dir


@pytest.fixture(autouse=True)
def _restore_chroma_dir():
    had = "CHROMA_PERSIST_DIR" in os.environ
    before = os.environ.get("CHROMA_PERSIST_DIR")
    yield
    if had:
        os.environ["CHROMA_PERSIST_DIR"] = before
    else:
        os.environ.pop("CHROMA_PERSIST_DIR", None)


def test_shell_explicit_value_respected():
    """셸에서 이미 CHROMA_PERSIST_DIR을 명시했으면(had=True) 손대지 않는다."""
    os.environ["CHROMA_PERSIST_DIR"] = "/data/chroma_db"
    use_local_chroma_dir(had_chroma_dir_before_dotenv=True)
    assert os.environ["CHROMA_PERSIST_DIR"] == "/data/chroma_db"


def test_deploy_path_from_dotenv_is_replaced_with_local():
    """셸 명시가 없고 .env 값이 /data로 시작하면 레포 루트 기준 ./chroma_db로 바꾼다."""
    os.environ["CHROMA_PERSIST_DIR"] = "/data/chroma_db"
    use_local_chroma_dir(had_chroma_dir_before_dotenv=False)
    result = os.environ["CHROMA_PERSIST_DIR"]
    assert result != "/data/chroma_db"
    assert result.endswith("chroma_db")
    assert os.path.isabs(result)


def test_already_local_path_untouched():
    """셸 명시가 없어도 .env 값이 이미 로컬 경로면 그대로 둔다."""
    os.environ["CHROMA_PERSIST_DIR"] = "./chroma_db"
    use_local_chroma_dir(had_chroma_dir_before_dotenv=False)
    assert os.environ["CHROMA_PERSIST_DIR"] == "./chroma_db"
