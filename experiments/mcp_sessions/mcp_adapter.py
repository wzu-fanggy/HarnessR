"""将真实MCP会话接到现有同步资源生命周期接口。

同一工作线程/异步任务拥有会话上下文，确保打开和关闭发生在同一任务。
官方SDK处理协议与stdio，独立Windows进程句柄确认服务实际退出。
"""
from __future__ import annotations
from release_paths import release_path

import asyncio
from concurrent.futures import Future
import ctypes
from ctypes import wintypes
from datetime import timedelta
import json
from pathlib import Path
import sys
import threading
import time

sys.path[:0] = [str(Path(__file__).resolve().parents[2]/'provenance/frozen_sources/mcp_sessions'), release_path('${ORION_SOURCE}'), release_path('${PROJECT_SOURCE}')]
from capvm.adapters.base import BaseCapabilityAdapter
from capvm.model import ResidencyLevel, ResourceVector
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parent
import os, shutil
NODE = os.environ.get('NODE_BINARY') or shutil.which('node') or 'node'
ENTRY = ROOT / 'server/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js'
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel.OpenProcess.restype = wintypes.HANDLE
kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel.WaitForSingleObject.restype = wintypes.DWORD
kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
kernel.CloseHandle.argtypes = [wintypes.HANDLE]


class MCPSessionAdapter(BaseCapabilityAdapter):
    """一个固定授权目录对应一个会话身份，服务进程属于该会话。"""

    def __init__(self, workspace: Path, records: Path):
        self.workspace = workspace.resolve()
        self.records = records
        self.records.mkdir(parents=True, exist_ok=True)
        self.generation = 0
        self.live = False
        self.events = []
        self._thread = None
        self._handle = None
        self.info = {}

    def declared_cost(self, level):
        return ResourceVector(connection_slots=1 if level >= ResidencyLevel.BOUND else 0)

    def measured_cost(self, level):
        # 这里计量实际存活的受管会话实例，不是操作系统网络连接数。
        return ResourceVector(connection_slots=int(self.live))

    def promote(self, current, target):
        if target >= ResidencyLevel.BOUND and not self.live:
            self._open()

    def demote(self, current, target):
        if target < ResidencyLevel.BOUND:
            self.close()

    def _open(self):
        self.generation += 1
        started = time.perf_counter()
        self._ready = Future()
        self._done = Future()
        self._pid_file = self.records / f'g{self.generation}.pid.json'
        self._thread = threading.Thread(target=self._thread_main, name=f'mcp-{self.workspace.name}', daemon=True)
        self._thread.start()
        try:
            self.info = self._ready.result(timeout=35)
            pid = json.loads(self._pid_file.read_text(encoding='utf-8'))['pid']
            self._handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
            if not self._handle:
                raise ctypes.WinError(ctypes.get_last_error())
            self.live = True
            self.events.append({'type': 'open', 'generation': self.generation, 'pid': pid,
                                'elapsed_s': time.perf_counter() - started, **self.info})
        except BaseException:
            if self._thread.is_alive() and hasattr(self, '_queue'):
                self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
                self._thread.join(timeout=15)
            raise

    def _thread_main(self):
        try:
            asyncio.run(self._owner())
            self._done.set_result(True)
        except BaseException as error:
            if not self._ready.done():
                self._ready.set_exception(error)
            self._done.set_exception(error)

    async def _owner(self):
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        params = StdioServerParameters(
            command=NODE,
            args=['--require', str(ROOT / 'pid_probe.cjs'), str(ENTRY), str(self.workspace)],
            env={'CWSR_MCP_PID_FILE': str(self._pid_file)},
        )
        with (self.records / f'g{self.generation}.stderr.log').open('w', encoding='utf-8') as log:
            async with stdio_client(params, errlog=log) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=20)) as session:
                    initialized = await session.initialize()
                    listed = await session.list_tools()
                    self._ready.set_result({
                        'protocol_version': initialized.protocolVersion,
                        'server_info': initialized.serverInfo.model_dump(mode='json'),
                        'tool_names': sorted(tool.name for tool in listed.tools),
                    })
                    while True:
                        command = await self._queue.get()
                        if command is None:
                            break
                        name, arguments, result = command
                        try:
                            response = await session.call_tool(name, arguments)
                            result.set_result(response.model_dump(mode='json'))
                        except BaseException as error:
                            result.set_exception(error)
                            if not isinstance(error, Exception):
                                raise

    def call(self, name, arguments):
        if not self.live:
            raise RuntimeError('MCP session is closed')
        result = Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (name, arguments, result))
        response = result.result(timeout=30)
        self.events.append({'type': 'call', 'generation': self.generation, 'tool': name,
                            'arguments': arguments, 'is_error': bool(response.get('isError'))})
        return response

    def close(self):
        if not self.live:
            return
        started = time.perf_counter()
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            self._done.result(timeout=35)
            self._thread.join(timeout=5)
            wait = kernel.WaitForSingleObject(self._handle, 5000)
            code = wintypes.DWORD()
            ok = kernel.GetExitCodeProcess(self._handle, ctypes.byref(code))
            confirmed = wait == 0 and bool(ok) and code.value != 259 and not self._thread.is_alive()
            self.events.append({'type': 'close', 'generation': self.generation,
                                'process_exit_confirmed': confirmed, 'exit_code': code.value,
                                'elapsed_s': time.perf_counter() - started})
            if not confirmed:
                raise RuntimeError('Server process exit was not confirmed')
        finally:
            kernel.CloseHandle(self._handle)
            self._handle = None
            self.live = False
