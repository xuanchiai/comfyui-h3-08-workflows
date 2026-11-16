import ctypes
from ctypes import wintypes
import logging
import os

import psutil

def release_resident_pages(threshold_mib=1536):
    if os.name != "nt" or psutil.Process().memory_info().rss < threshold_mib * 1024**2:
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = ()
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.K32EmptyWorkingSet.argtypes = (wintypes.HANDLE,)
    kernel.K32EmptyWorkingSet.restype = wintypes.BOOL
    if not kernel.K32EmptyWorkingSet(kernel.GetCurrentProcess()):
        logging.warning("MOSS 2K 驻留页回收失败，Windows 错误 %d", ctypes.get_last_error())
