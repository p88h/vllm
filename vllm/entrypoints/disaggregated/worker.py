# SPDX-License-Identifier: Apache-2.0

import uvloop

from vllm.disaggregated.pd_worker import PDWorker
from vllm.engine.async_llm_engine import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.logger import init_logger
from vllm.utils import FlexibleArgumentParser
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


async def run(args, engine: EngineClient):
    try:
        worker = PDWorker(engine, args.worker_addr, args.controller_addr)
        await worker.run_busy_loop()
    finally:
        worker.shutdown()


async def main(args) -> None:
    logger.info("vLLM P/D Worker Server %s", VLLM_VERSION)
    logger.info("Args: %s", args)

    args.disable_frontend_multiprocessing = False
    async with build_async_engine_client(args) as engine:
        await run(args, engine)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument('--controller-addr',
                        type=str,
                        required=True,
                        help='The address of the controller.')
    parser.add_argument('--worker-addr',
                        type=str,
                        required=True,
                        help='The address of the worker.')
    AsyncEngineArgs.add_cli_args(parser)
    uvloop.run(main(parser.parse_args()))
