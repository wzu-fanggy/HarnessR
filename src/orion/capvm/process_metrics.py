"""操作系统进程内存计量：端点与生命周期峰值分开，不推断子进程占用。"""
import os
from pathlib import Path

def process_memory_snapshot(pid):
    if isinstance(pid,bool) or not isinstance(pid,int) or pid<=0:raise ValueError('invalid PID')
    if os.name=='nt':return _windows(pid)
    if Path('/proc').exists():
        values={}
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            key,_,value=line.partition(':')
            if key in ('VmRSS','VmHWM'):values[key]=float(value.split()[0])/1024
        if 'VmRSS' not in values:raise OSError('RSS unavailable')
        return {'pid':pid,'scope':'single_process','source':'proc_status',
            'rss_mb':values['VmRSS'],'peak_rss_mb':values.get('VmHWM'),
            'commit_mb':None,'peak_commit_mb':None,'peak_scope':'process_lifetime'}
    raise NotImplementedError('process memory telemetry unsupported')

def _windows(pid):
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);psapi=ctypes.WinDLL('psapi',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME)];kernel.GetProcessTimes.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        c=Counters();c.cb=ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(c),c.cb):raise ctypes.WinError(ctypes.get_last_error())
        created,exited,kernel_time,user_time=(wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle,ctypes.byref(created),ctypes.byref(exited),ctypes.byref(kernel_time),ctypes.byref(user_time)):raise ctypes.WinError(ctypes.get_last_error())
        return {'pid':pid,'creation_filetime':(created.dwHighDateTime<<32)|created.dwLowDateTime,
            'scope':'single_process','source':'GetProcessMemoryInfo','rss_mb':c.WorkingSetSize/1048576,
            'peak_rss_mb':c.PeakWorkingSetSize/1048576,'commit_mb':c.PrivateUsage/1048576,
            'peak_commit_mb':c.PeakPagefileUsage/1048576,'peak_scope':'process_lifetime'}
    finally:kernel.CloseHandle(handle)

