// 只记录本实验拥有的服务进程，不改变工具或协议实现。
const fs = require('node:fs');
const target = process.env.CWSR_MCP_PID_FILE;
if (!target) throw new Error('CWSR_MCP_PID_FILE is required');
fs.writeFileSync(target, JSON.stringify({pid: process.pid, argv: process.argv}));
