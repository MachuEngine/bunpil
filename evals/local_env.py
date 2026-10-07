"""로컬 실행에서 배포용 CHROMA_PERSIST_DIR을 피하기 위한 공용 헬퍼 (2026-10-07).

배경: 로컬 `.env`의 CHROMA_PERSIST_DIR은 배포 경로(`/data/chroma_db`)로 설정돼
있다(EC2 배포 기준). 이 값을 그대로 쓰는 평가·실험·생성 스크립트를 맥 등 로컬에서
돌리면 `RAGStore` 초기화가 `OSError: [Errno 30] Read-only file system: '/data'`로
조용히(또는 요란하게) 실패한다 — golden_gen/gen_item_quality_golden.py가 2026-10
생성 모델 비교 내내 이 문제로 search_standards가 실패했고, evals/eval_exam.py도
같은 이유로 매번 `CHROMA_PERSIST_DIR=./chroma_db`를 수동으로 줘야 했다.

호출 순서(중요) — load_dotenv() 호출 **전**에 셸 명시 여부를 캡처하고, **후**에
이 함수를 불러야 한다. 셸 명시 여부를 load_dotenv() 뒤에 판정하면 .env가 채운
값과 구분할 수 없다:

    had_chroma_dir = "CHROMA_PERSIST_DIR" in os.environ  # load_dotenv() *전*
    load_dotenv()
    use_local_chroma_dir(had_chroma_dir)                  # load_dotenv() *후*
"""
import os


def use_local_chroma_dir(had_chroma_dir_before_dotenv: bool) -> None:
    """셸에서 CHROMA_PERSIST_DIR을 명시했다면(had_chroma_dir_before_dotenv=True) 그 값을
    존중하고 아무 것도 하지 않는다. 명시하지 않았는데 .env 값이 배포 경로(`/data`로
    시작)이면 레포 루트 기준 `./chroma_db`(절대경로)로 바꾸고 한 줄 출력한다. 이미
    로컬 경로면 그대로 둔다."""
    if had_chroma_dir_before_dotenv:
        return
    current = os.environ.get("CHROMA_PERSIST_DIR", "")
    if current.startswith("/data"):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_dir = os.path.join(repo_root, "chroma_db")
        os.environ["CHROMA_PERSIST_DIR"] = local_dir
        print(f"[local_env] CHROMA_PERSIST_DIR이 배포 경로({current!r})라 로컬 경로로 전환: {local_dir}")
