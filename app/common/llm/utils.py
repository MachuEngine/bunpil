import asyncio
import concurrent.futures


def run_async(coro):
    """async coroutine을 동기 컨텍스트에서 안전하게 실행한다."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=300)
