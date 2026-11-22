"""The container sandbox: for a machine that is a container of its own.

The first sandbox (runner.py, bubblewrap) is made of namespaces. Rented GPUs are mostly
Docker containers that may not make any: no CAP_SYS_ADMIN, and Docker's system-call filter
refuses unshare(). This one is made of what such a container does let root do, and it is
for a container that runs stet and nothing else, such as one of the organisers' rented
machines (stet-cloud). It takes over the container's top-level /editor, /corpus,
/assets, /out and /scratch, and empties /tmp and /dev/shm of the curator's files.

  Who        The curator runs as a user of its own, with no capabilities and no way to
             gain any (no_new_privs). Everything stet keeps (its key, its cache, other
             curators' bundles and scripts) is under /root, which that user cannot enter.
  Files      Its bundle at /editor, the corpus at /corpus and its assets under /assets,
             read-only: root's files, linked there. /out and /scratch are its own. The
             same paths as in the other sandbox.
  Network    A system-call filter hands every internet socket the curator asks for to
             stet, which makes it itself, TCP only, bound to the loopback device: nothing
             sent on it leaves the machine. Landlock lets it connect only to ports that
             nothing else on the machine was listening on when it started, and from 1024
             up, so not to Docker's resolver, which would carry queries out. UDP, raw and
             every other family but Unix and netlink are refused, and so are unbinding a
             socket and TCP Fast Open (a connect that Landlock would not see).
  Limits     The same cores (affinity, and a cgroup's CPU quota where the container lets
             stet make one), the same memory (that cgroup's limit, or else stet adds up
             what the curator holds four times a second and stops it at the limit, and the
             kernel's out-of-memory killer takes the curator first), the same number of
             processes, and the same GPU memory (PyTorch's own limit).
  Afterwards Every process the curator's user has is killed, and what it left in /out,
             /scratch, /tmp, /var/tmp, /dev/shm, /dev/mqueue, /run/lock and System V IPC
             is removed, so that nothing of one curator reaches the next.

What it does not do that the other sandbox does:

  - It does not hide the machine's other processes. /proc lists them, and their command
    lines; never their environments, files or memory.
  - /tmp and /dev/shm are the machine's own, emptied of the curator's files afterwards,
    and /scratch is on disk, not in memory: it does not come out of the memory limit.
  - Loopback is TCP only.
  - Where the container lets stet make no cgroup, the memory limit is a watch, not the
    kernel's: a curator that grabs memory faster than it is looked at can pass it for a
    moment, and is then stopped.

Like the other sandbox it is not a virtual machine: a flaw in the kernel or the GPU driver
is outside it.
"""

from __future__ import annotations

import ctypes
import errno
import os
import select
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

UID = GID = 61000  # the curator's user and group: no name, no home, no files of its own until a run
EDITOR, CORPUS, ASSETS, OUT, SCRATCH = (Path(p) for p in ("/editor", "/corpus", "/assets", "/out", "/scratch"))
# Every directory on a stock image that anyone may write in. What the curator leaves in them is
# removed after each run.
SHARED = (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"), Path("/dev/mqueue"), Path("/run/lock"))
SHARED_LIMIT_BYTES = 2 << 30  # /tmp and /dev/shm are 2 GB each in the other sandbox
WATCH_SECONDS = 0.25

# x86-64 system calls and constants the filter and the supervisor use.
NR_SOCKET, NR_SENDTO, NR_SENDMSG, NR_SETSOCKOPT, NR_SENDMMSG = 41, 44, 46, 54, 307
DENIED = {
    101: errno.EPERM,  # ptrace
    248: errno.EPERM, 249: errno.EPERM, 250: errno.EPERM,  # add_key, request_key, keyctl: keyrings outlive a run
    298: errno.EPERM,  # perf_event_open
    310: errno.EPERM, 311: errno.EPERM,  # process_vm_readv, process_vm_writev
    321: errno.EPERM,  # bpf
    323: errno.EPERM,  # userfaultfd
    425: errno.ENOSYS, 426: errno.ENOSYS, 427: errno.ENOSYS,  # io_uring: it makes sockets and sends without these checks
    438: errno.EPERM,  # pidfd_getfd
}  # fmt: skip
NR_SECCOMP, NR_LANDLOCK_CREATE, NR_LANDLOCK_ADD, NR_LANDLOCK_RESTRICT = 317, 444, 445, 446
AUDIT_ARCH_X86_64 = 0xC000003E
AF_UNIX, AF_INET, AF_INET6, AF_NETLINK = 1, 2, 10, 16
SOL_SOCKET, SO_BINDTODEVICE, SO_BINDTOIFINDEX = 1, 25, 62
IPPROTO_TCP, TCP_FASTOPEN_CONNECT = 6, 30
MSG_FASTOPEN = 0x20000000
SOCK_STREAM, SOCK_NONBLOCK, SOCK_CLOEXEC = 1, 0o4000, 0o2000000

RET_KILL, RET_NOTIFY, RET_ERRNO, RET_ALLOW = 0x80000000, 0x7FC00000, 0x00050000, 0x7FFF0000
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_NEW_LISTENER = 1, 1 << 3
LANDLOCK_CREATE_RULESET_VERSION, LANDLOCK_RULE_NET_PORT, LANDLOCK_ACCESS_NET_CONNECT_TCP = 1, 2, 1 << 1


class CageError(RuntimeError):
    """The container sandbox cannot be made here, or could not be put away."""


# --- the system-call filter -------------------------------------------------------------------


def program() -> list[tuple[int, int, int, int]]:
    """The seccomp filter, as classic BPF (code, jt, jf, k).

    seccomp_data: nr at 0, arch at 4, args[i] at 16 + 8i (their low halves, which is all
    an int argument is).
    """
    LD, JEQ, JGE, JSET, RET = 0x20, 0x15, 0x35, 0x45, 0x06
    lines: list = []  # instructions, with jumps by label to be resolved below
    labels: dict[str, int] = {}

    def at(name):
        labels[name] = len(lines)

    def op(code, k, jt=None, jf=None):
        lines.append((code, k, jt, jf))

    op(LD, 4)
    op(JEQ, AUDIT_ARCH_X86_64, None, "kill")
    op(LD, 0)
    op(JGE, 0x40000000, "kill", None)  # the x32 numbering, which would get round all of this
    op(JEQ, NR_SOCKET, "socket", None)
    op(JEQ, NR_SETSOCKOPT, "setsockopt", None)
    op(JEQ, NR_SENDTO, "flags3", None)
    op(JEQ, NR_SENDMMSG, "flags3", None)
    op(JEQ, NR_SENDMSG, "flags2", None)
    for nr, err in sorted(DENIED.items()):
        op(JEQ, nr, f"err{err}", None)
    op(RET, RET_ALLOW)
    at("socket")
    op(LD, 16)
    op(JEQ, AF_UNIX, "allow", None)
    op(JEQ, AF_NETLINK, "allow", None)
    op(JEQ, AF_INET, "notify", None)
    op(JEQ, AF_INET6, "notify", None)
    op(RET, RET_ERRNO | errno.EAFNOSUPPORT)
    at("setsockopt")
    op(LD, 24)
    op(JEQ, SOL_SOCKET, None, "tcpopt")
    op(LD, 32)
    op(JEQ, SO_BINDTODEVICE, f"err{errno.EPERM}", None)
    op(JEQ, SO_BINDTOIFINDEX, f"err{errno.EPERM}", None)
    op(RET, RET_ALLOW)
    at("tcpopt")
    op(JEQ, IPPROTO_TCP, None, "allow")
    op(LD, 32)
    op(JEQ, TCP_FASTOPEN_CONNECT, f"err{errno.EPERM}", None)
    op(RET, RET_ALLOW)
    at("flags3")  # sendto and sendmmsg: flags are the fourth argument
    op(LD, 40)
    op(JSET, MSG_FASTOPEN, f"err{errno.EPERM}", None)
    op(RET, RET_ALLOW)
    at("flags2")  # sendmsg: the third
    op(LD, 32)
    op(JSET, MSG_FASTOPEN, f"err{errno.EPERM}", None)
    op(RET, RET_ALLOW)
    at("allow")
    op(RET, RET_ALLOW)
    at("notify")
    op(RET, RET_NOTIFY)
    at("kill")
    op(RET, RET_KILL)
    for err in sorted({errno.EPERM, *DENIED.values()}):
        at(f"err{err}")
        op(RET, RET_ERRNO | err)

    def rel(i, target):
        if target is None:
            return 0
        d = labels[target] - i - 1
        if not 0 <= d < 256:
            raise ValueError(f"jump to {target} out of reach")
        return d

    return [(code, rel(i, jt), rel(i, jf), k) for i, (code, k, jt, jf) in enumerate(lines)]


def run_bpf(prog, nr: int, args=(0, 0, 0, 0, 0, 0), arch: int = AUDIT_ARCH_X86_64) -> int:
    """What the filter returns for one call. For testing it without a kernel."""
    data = [nr, arch, 0, 0] + [w for a in args for w in (a & 0xFFFFFFFF, a >> 32)]
    acc, pc = 0, 0
    while True:
        code, jt, jf, k = prog[pc]
        if code == 0x20:
            acc = data[k // 4]
        elif code == 0x06:
            return k
        else:
            hit = {0x15: acc == k, 0x35: acc >= k, 0x45: bool(acc & k)}[code]
            pc += jt if hit else jf
        pc += 1


# --- inside: what the launcher does before the curator starts ------------------------------------
#
# The launcher is a fresh Python process, started by the harness as root. It puts itself in
# the curator's cgroup and on its cores, takes its limits, closes the network to it, hands
# the filter's listener to the harness, and becomes the curator's user before it runs the
# curator. Nothing of it runs as root once the curator's code does.

_libc = None


def libc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.syscall.restype = ctypes.c_long
        _libc.ioctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
    return _libc


def _check(r: int, what: str) -> int:
    if r < 0:
        e = ctypes.get_errno()
        raise OSError(e, f"{what}: {os.strerror(e)}")
    return r


class _Filter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint)]


class _Fprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_Filter))]


def landlock_abi() -> int:
    """Landlock's ABI version here, or 0 where there is none."""
    r = libc().syscall(NR_LANDLOCK_CREATE, None, ctypes.c_size_t(0), LANDLOCK_CREATE_RULESET_VERSION)
    return max(0, int(r))


def listening_ports() -> set[int]:
    """TCP ports something on this machine is listening on now, by any address."""
    ports: set[int] = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            if len(f) > 3 and f[3] == "0A":  # LISTEN
                ports.add(int(f[1].rsplit(":", 1)[1], 16))
    return ports


def _close_ports(open_ports) -> None:
    """Let this process and its children connect over TCP to these ports and no others."""
    class Attr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64)]

    class Port(ctypes.Structure):
        _fields_ = [("allowed_access", ctypes.c_uint64), ("port", ctypes.c_uint64)]

    attr = Attr(0, LANDLOCK_ACCESS_NET_CONNECT_TCP)
    fd = _check(libc().syscall(NR_LANDLOCK_CREATE, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), 0), "landlock_create_ruleset")
    try:
        rule = Port(LANDLOCK_ACCESS_NET_CONNECT_TCP, 0)
        add = libc().syscall
        for p in open_ports:
            rule.port = p
            _check(add(NR_LANDLOCK_ADD, fd, LANDLOCK_RULE_NET_PORT, ctypes.byref(rule), 0), "landlock_add_rule")
        _check(libc().syscall(NR_LANDLOCK_RESTRICT, fd, 0), "landlock_restrict_self")
    finally:
        os.close(fd)


def _filter() -> int:
    """Install the filter on this process (and so the curator); returns its listener."""
    prog = program()
    arr = (_Filter * len(prog))(*[_Filter(c, t, f, k) for c, t, f, k in prog])
    fprog = _Fprog(len(prog), arr)
    return _check(libc().syscall(NR_SECCOMP, SECCOMP_SET_MODE_FILTER, SECCOMP_FILTER_FLAG_NEW_LISTENER, ctypes.byref(fprog)), "seccomp")


def launch() -> None:
    """The launcher's body. argv: up-fd channel-fd uid cpus cgroup-procs nproc closed-ports cwd -- curator..."""
    import resource

    a = sys.argv
    up, chan, uid = int(a[1]), int(a[2]), int(a[3])
    cpus = [int(c) for c in a[4].split(",") if c]
    procs, nproc = a[5], int(a[6])
    closed = {int(p) for p in a[7].split(",") if p}
    cwd, argv = a[8], a[10:]
    try:
        if procs:
            Path(procs).write_text(str(os.getpid()))
        if cpus:
            os.sched_setaffinity(0, cpus)
        Path("/proc/self/oom_score_adj").write_text("1000")  # what the kernel kills first, if memory runs out
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        _check(libc().prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "no_new_privs")
        _close_ports(p for p in range(1024, 65536) if p not in closed)
        listener = _filter()
        socket.send_fds(socket.socket(fileno=chan), [b"L"], [listener])
        os.close(listener)
        os.setgroups([])
        os.setgid(uid)
        os.setuid(uid)
        if 0 in (os.getuid(), os.geteuid(), os.getgid(), os.getegid()):
            raise OSError("still root")
        os.umask(0o022)
        os.chdir(cwd)
    except OSError as e:
        sys.stderr.write(f"stet: the container sandbox could not be set up: {e}\n")
        sys.exit(126)
    os.write(up, b"up")
    os.close(up)
    try:
        os.execvp(argv[0], argv)
    except OSError as e:
        sys.stderr.write(f"stet: cannot start {argv[0]!r}: {e.strerror}\n")
        sys.exit(127)


# --- outside: the harness's side ----------------------------------------------------------------


class _Data(ctypes.Structure):
    _fields_ = [("nr", ctypes.c_int), ("arch", ctypes.c_uint32), ("ip", ctypes.c_uint64), ("args", ctypes.c_uint64 * 6)]


class _Notif(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("pid", ctypes.c_uint32), ("flags", ctypes.c_uint32), ("data", _Data)]


class _Resp(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("val", ctypes.c_int64), ("error", ctypes.c_int32), ("flags", ctypes.c_uint32)]


class _AddFd(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64), ("flags", ctypes.c_uint32), ("srcfd", ctypes.c_uint32), ("newfd", ctypes.c_uint32), ("newfd_flags", ctypes.c_uint32)]


def _ioc(direction: int, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord("!") << 8) | nr


NOTIF_RECV = _ioc(3, 0, ctypes.sizeof(_Notif))
NOTIF_SEND = _ioc(3, 1, ctypes.sizeof(_Resp))
NOTIF_ADDFD = _ioc(1, 3, ctypes.sizeof(_AddFd))
ADDFD_FLAG_SEND = 1 << 1


def loopback_socket(domain: int, kind: int, proto: int) -> int:
    """A TCP socket made by stet for the curator, bound to the loopback device, or -errno.

    Made here, not checked and let through: the curator could change what it asked for
    between the check and the call, but it cannot change a socket it was handed.
    """
    base, flags = kind & 0xF, kind & ~0xF
    if domain not in (AF_INET, AF_INET6) or base != SOCK_STREAM or proto not in (0, IPPROTO_TCP) or flags & ~(SOCK_NONBLOCK | SOCK_CLOEXEC):
        return -errno.EPERM  # UDP, raw, ping and the rest: there is nothing on the loopback for them
    fd = libc().socket(domain, SOCK_STREAM | (flags & SOCK_NONBLOCK) | SOCK_CLOEXEC, proto)
    if fd < 0:
        return -ctypes.get_errno()
    lo = ctypes.create_string_buffer(b"lo")
    if libc().setsockopt(fd, SOL_SOCKET, SO_BINDTODEVICE, lo, ctypes.c_uint(3)) != 0:
        e = ctypes.get_errno()
        os.close(fd)
        return -e
    return fd


class Supervisor(threading.Thread):
    """Answers the curator's requests for internet sockets with loopback ones, until it is gone."""

    def __init__(self, listener: int):
        super().__init__(daemon=True)
        self.listener = listener
        self.stop = threading.Event()
        self.handed = 0
        self.refused = 0

    def run(self):
        poll = select.poll()
        poll.register(self.listener, select.POLLIN)
        try:
            while not self.stop.is_set():
                events = poll.poll(200)
                if not events:
                    continue
                if any(ev & (select.POLLHUP | select.POLLERR) for _, ev in events) and not any(ev & select.POLLIN for _, ev in events):
                    return  # nothing is left that the filter applies to
                n = _Notif()
                if libc().ioctl(self.listener, NOTIF_RECV, ctypes.byref(n)) != 0:
                    continue  # the caller went away before it could be answered
                self.answer(n)
        finally:
            os.close(self.listener)

    def answer(self, n: _Notif) -> None:
        a = n.data.args
        fd = loopback_socket(a[0] & 0xFFFFFFFF, a[1] & 0xFFFFFFFF, a[2] & 0xFFFFFFFF) if n.data.nr == NR_SOCKET else -errno.EPERM
        if fd < 0:
            self.refused += 1
            libc().ioctl(self.listener, NOTIF_SEND, ctypes.byref(_Resp(n.id, 0, fd, 0)))
            return
        try:
            cloexec = os.O_CLOEXEC if (a[1] & SOCK_CLOEXEC) else 0
            if libc().ioctl(self.listener, NOTIF_ADDFD, ctypes.byref(_AddFd(n.id, ADDFD_FLAG_SEND, fd, 0, cloexec))) >= 0:
                self.handed += 1
        finally:
            os.close(fd)


def in_container() -> bool:
    return os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv") or bool(os.environ.get("container"))


def _blocked(path: Path, need_read: bool = True) -> Path | None:
    """The first directory on the way to `path` that the curator's user cannot pass, or the
    path itself if it cannot read it; None if it can reach and read it."""
    p = Path(os.path.realpath(path))
    for d in [*reversed(p.parents)]:
        try:
            st = os.stat(d)
        except OSError:
            return d
        if not _may(st, 0o1):
            return d
    try:
        st = os.stat(p)
    except OSError:
        return p
    want = (0o5 if stat.S_ISDIR(st.st_mode) else 0o4) if need_read else 0o1
    return None if _may(st, want) else p


def _may(st: os.stat_result, bits: int) -> bool:
    """Whether the curator's user may do `bits` (rwx as 4/2/1) to a file with this stat."""
    if st.st_uid == UID:
        return (st.st_mode >> 6) & bits == bits
    if st.st_gid == GID:
        return (st.st_mode >> 3) & bits == bits
    return st.st_mode & bits == bits


def available() -> tuple[bool, str]:
    """Whether the container sandbox can be made here, and why not."""
    if sys.platform != "linux" or os.uname().machine != "x86_64":
        return False, "the container sandbox is for Linux on x86-64"
    if os.geteuid() != 0:
        return False, "the container sandbox needs stet to run as root, in a container that runs nothing else"
    if not in_container():
        return False, "the container sandbox is only for a container that runs stet and nothing else, and this does not look like a container (no /.dockerenv)"
    major, minor = (int(x) for x in os.uname().release.split(".")[:2])
    if (major, minor) < (5, 14):
        return False, f"the container sandbox needs Linux 5.14 or newer, and this is {os.uname().release}"
    if landlock_abi() < 4:
        return False, "the container sandbox needs Landlock's network rules (Linux 6.7 or newer, with Landlock on), to keep the curator off this machine's other ports"
    import sysconfig

    from .keys import config_dir

    # The curator runs this Python, with these packages: its user has to reach them.
    for need in {sys.executable, sys.base_prefix, sysconfig.get_paths()["purelib"], str(Path(__file__).parent)}:
        if (where := _blocked(Path(need))) is not None:
            return False, (f"the curator's user cannot reach {need} (it stops at {where}). Install stet where any user can read it, "
                           "outside /root: UV_TOOL_DIR=/opt/stet/tools UV_PYTHON_INSTALL_DIR=/opt/stet/python before the installer")  # fmt: skip
    # And must not reach what is this machine's: its key and its cache (a round's exam is kept there).
    from .data import cache_dir

    for what, path in (("the key", config_dir() / "key.json"), ("the config directory", config_dir()), ("the cache", cache_dir())):
        if path.exists() and _blocked(path) is None:
            return False, f"the curator's user could read {what} ({path}): keep it under a directory only root may enter, such as /root"
    for d in (EDITOR, CORPUS, ASSETS, OUT, SCRATCH):
        if d.is_symlink() or (d.exists() and not d.is_dir()):
            return False, f"{d} is in the way: the container sandbox puts the curator's files there"
    return True, ""


# --- the cgroup, where the container lets one be made ---------------------------------------------

_cgroup: list[Path | None] = []


def cgroup(limits: dict, cores: int) -> Path | None:
    """A cgroup for the curator with the season's limits in force, or None.

    Only where /sys/fs/cgroup is this container's own cgroup2 tree, written to by root. Its
    processes are moved into a leaf first, since a cgroup with processes of its own cannot
    hand controllers to children. Made once per process; the limits are written each time.
    """
    if not _cgroup:
        root = Path("/sys/fs/cgroup")
        made = None
        try:
            own = Path("/proc/self/cgroup").read_text().strip()
            if own == "0::/stet-harness" and (root / "stet-curator" / "memory.max").exists():
                made = root / "stet-curator"  # made by an earlier stet in this container
            elif own == "0::/" and (root / "cgroup.controllers").exists() and {"memory", "pids", "cpu"} <= set((root / "cgroup.controllers").read_text().split()):
                harness = root / "stet-harness"
                harness.mkdir(exist_ok=True)
                for pid in (root / "cgroup.procs").read_text().split():
                    try:
                        (harness / "cgroup.procs").write_text(pid)
                    except OSError:
                        pass  # gone already
                (root / "cgroup.subtree_control").write_text("+memory +pids +cpu")
                made = root / "stet-curator"
                made.mkdir(exist_ok=True)
        except OSError:
            made = None
        _cgroup.append(made)
    cg = _cgroup[0]
    if cg is None:
        return None
    try:
        (cg / "memory.max").write_text(str(int(limits["memory_gb"]) << 30))
        if (cg / "memory.swap.max").exists():
            (cg / "memory.swap.max").write_text("0")
        (cg / "pids.max").write_text(str(int(limits["tasks"])))
        (cg / "cpu.max").write_text(f"{cores * 100000} 100000" if cores else "max 100000")
        if int((cg / "memory.max").read_text()) != int(limits["memory_gb"]) << 30:
            return None
    except (OSError, ValueError):
        return None
    return cg


def oom_kills(cg: Path | None) -> int:
    """How many of the cgroup's processes the kernel has killed for memory, so far."""
    if cg is None:
        return 0
    try:
        for line in (cg / "memory.events").read_text().splitlines():
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    except (OSError, ValueError):
        pass
    return 0


# --- before and after ------------------------------------------------------------------------------


def pids() -> list[int]:
    """The curator's user's processes that are alive (not zombies)."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = os.stat(f"/proc/{d}")
            if st.st_uid != UID:
                continue
            with open(f"/proc/{d}/stat") as f:
                if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                    continue
        except (OSError, IndexError):
            continue
        out.append(int(d))
    return out


REAPER = "import os, signal, sys\nu = int(sys.argv[1])\nos.setgroups([]); os.setgid(u); os.setuid(u)\nfor _ in range(50):\n    try:\n        os.kill(-1, signal.SIGKILL)\n    except ProcessLookupError:\n        break\n"


def reap(cg: Path | None = None) -> bool:
    """Kill every process of the curator's user. True once none is left.

    As that user, kill(-1) reaches all of them and nothing else; a fork bomb is bounded by
    its process limit, and the sweep is repeated until nothing answers.
    """
    if cg is not None:
        try:
            (cg / "cgroup.kill").write_text("1")
        except OSError:
            pass
    for _ in range(40):
        if not pids():
            return True
        subprocess.run([sys.executable, "-I", "-c", REAPER, str(UID)], stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
        time.sleep(0.05)
    return not pids()


def _owned(path: Path) -> bool:
    try:
        return os.lstat(path).st_uid == UID
    except OSError:
        return False


# Removing a System V object takes its owner, or CAP_SYS_ADMIN, which root in a container has
# not got: so it is done as the curator's user, which owns them.
IPC_CLEANER = """import ctypes, os, sys
u = int(sys.argv[1])
os.setgroups([]); os.setgid(u); os.setuid(u)
libc = ctypes.CDLL(None)
for kind, column in (("shm", "shmid"), ("sem", "semid"), ("msg", "msqid")):
    try:
        lines = open("/proc/sysvipc/" + kind).read().splitlines()
    except OSError:
        continue
    head = lines[0].split() if lines else []
    for line in lines[1:]:
        f = dict(zip(head, line.split()))
        try:
            if int(f["uid"]) != u and int(f.get("cuid", -1)) != u:
                continue
            ident = int(f[column])
        except (KeyError, ValueError):
            continue
        if kind == "shm":
            libc.shmctl(ident, 0, None)
        elif kind == "sem":
            libc.semctl(ident, 0, 0)
        else:
            libc.msgctl(ident, 0, None)
"""


def _ipc_left() -> bool:
    """Whether the curator's user owns any System V shared memory, semaphores or queues:
    they outlive the processes that made them."""
    for kind in ("shm", "sem", "msg"):
        try:
            lines = Path(f"/proc/sysvipc/{kind}").read_text().splitlines()
        except OSError:
            continue
        head = lines[0].split() if lines else []
        for line in lines[1:]:
            f = dict(zip(head, line.split(), strict=False))
            if f.get("uid") == str(UID) or f.get("cuid") == str(UID):
                return True
    return False


def _ipc_clean() -> bool:
    if not _ipc_left():
        return True
    subprocess.run([sys.executable, "-I", "-c", IPC_CLEANER, str(UID)], stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
    return not _ipc_left()


def clean() -> bool:
    """Remove everything the curator's user left on this machine. True if nothing is left."""
    from .runner import remove_tree

    left = True
    for d in (OUT, SCRATCH):
        left = remove_tree(d) and left
    for d in SHARED:
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            p = d / name
            if _owned(p):
                left = remove_tree(p) and left
    return _ipc_clean() and left


def _copy_bundle(src: Path, dest: Path) -> None:
    """The bundle, as root's read-only files: every file readable, as executable as it was."""
    import shutil

    shutil.copytree(src, dest, symlinks=True)
    for dirpath, _, filenames in os.walk(dest):
        os.chmod(dirpath, 0o755)
        for n in filenames:
            p = os.path.join(dirpath, n)
            if not os.path.islink(p):
                mode = os.lstat(p).st_mode
                os.chmod(p, 0o755 if mode & 0o111 else 0o644)


def _link(src: Path, dest: Path) -> None:
    """A read-only link to one of root's files, or a copy where a link cannot be made."""
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copyfile(src, dest)
    os.chmod(dest, 0o444)


def prepare(bundle: Path, corpus: Path, assets=()) -> None:
    """Lay out the curator's files where it will find them, and nothing of the last one's."""
    from .assets import cached
    from .runner import remove_tree

    if not reap():
        raise CageError("processes of the curator's user are still running from before, and will not die")
    if not clean():
        raise CageError("what a curator left on this machine could not all be removed")
    for d in (EDITOR, CORPUS, ASSETS):
        if not remove_tree(d):
            raise CageError(f"could not clear {d}")
    _copy_bundle(bundle, EDITOR)
    CORPUS.mkdir(mode=0o755)
    for dirpath, _, filenames in os.walk(corpus):
        for n in filenames:
            src = Path(dirpath) / n
            _link(src, CORPUS / src.relative_to(corpus))
    ASSETS.mkdir(mode=0o755)
    for a in assets:
        _link(cached(a), ASSETS / a.path)
    for dirpath, _, _ in os.walk(ASSETS):
        os.chmod(dirpath, 0o755)
    for dirpath, _, _ in os.walk(CORPUS):
        os.chmod(dirpath, 0o755)
    for d in (OUT, SCRATCH):
        d.mkdir(mode=0o700)
        os.chown(d, UID, GID)


def usage_by_curator(d: Path, limit_entries: int = 100_000) -> int:
    """Bytes the curator's user holds in a shared directory."""
    from .runner import _usage

    total = 0
    try:
        names = os.listdir(d)
    except OSError:
        return 0
    for name in names[:limit_entries]:
        p = d / name
        try:
            st = os.lstat(p)
        except OSError:
            continue
        if st.st_uid != UID:
            continue
        if stat.S_ISDIR(st.st_mode):
            total += _usage(p, limit_entries)[0]
        else:
            total += max(st.st_size, st.st_blocks * 512)
    return total


def _container_memory() -> int | None:
    """Anonymous and shared memory in use in this whole container, by its own cgroup's count
    (which counts a page once, however many processes share it), or None."""
    for path, keys in ((Path("/sys/fs/cgroup/memory.stat"), ("anon", "shmem")),
                       (Path("/sys/fs/cgroup/memory/memory.stat"), ("total_rss", "total_shmem"))):  # fmt: skip
        try:
            stats = dict(line.split()[:2] for line in path.read_text().splitlines() if line.strip())
            return sum(int(stats[k]) for k in keys)
        except (OSError, ValueError, KeyError):
            continue
    return None


def _status_bytes(pid: int, fields: tuple[str, ...]) -> int:
    total = 0
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.split(":")[0] in fields:
                    total += int(line.split()[1]) << 10
    except (OSError, ValueError, IndexError):
        pass
    return total


def memory_in_use() -> int:
    """About what the curator holds of this machine's memory.

    Root here may not read another user's memory maps, so the curator's pages are counted
    from outside: the container's own count of anonymous and shared memory, less what the
    harness's processes hold (their own, which root may read). Where the container's count
    cannot be read, the curator's processes' resident anonymous and shared memory is added
    up instead, which counts a page twice where two processes share it after a fork.
    """
    whole = _container_memory()
    if whole is None:
        return sum(_status_bytes(pid, ("RssAnon", "RssShmem")) for pid in pids())
    harness = 0
    mine = set(pids())
    for d in os.listdir("/proc"):
        if d.isdigit() and int(d) not in mine:
            try:
                with open(f"/proc/{d}/smaps_rollup") as f:
                    for line in f:
                        if line.startswith(("Pss_Anon:", "Pss_Shmem:")):
                            harness += int(line.split()[1]) << 10
            except (OSError, ValueError):
                continue
    return max(0, whole - harness)


class Watch(threading.Thread):
    """Stops a curator that holds more than its memory, or fills /tmp or /dev/shm."""

    def __init__(self, memory_bytes: int | None, found: list[str]):
        super().__init__(daemon=True)
        self.memory_bytes, self.found = memory_bytes, found
        self.done = threading.Event()

    def run(self):
        n = 0
        while not self.done.wait(WATCH_SECONDS) and not self.found:
            n += 1
            if self.memory_bytes and memory_in_use() > self.memory_bytes:
                self.found.append(f"the curator used more than {self.memory_bytes >> 30} GB of memory")
            elif n % 8 == 0:
                for d in (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm")):
                    if usage_by_curator(d) > SHARED_LIMIT_BYTES:
                        self.found.append(f"the curator wrote more than {SHARED_LIMIT_BYTES >> 30} GB to {d}")
                        break


def boot_command(py_exe: str, up_w: int, chan: int, cpus: list[int], cg: Path | None, tasks: int, argv: list[str]) -> list[str]:
    """How the harness starts the launcher, which starts the curator."""
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    boot = f"import sys; sys.path.insert(0, {pkg_parent!r}); from stet.cage import launch; launch()"
    closed = sorted(listening_ports())
    return [py_exe, "-I", "-c", boot, str(up_w), str(chan), str(UID), ",".join(map(str, cpus)),
            str(cg / "cgroup.procs") if cg else "", str(tasks), ",".join(map(str, closed)), str(EDITOR), "--", *argv]  # fmt: skip


def receive_listener(sock: socket.socket, timeout: float = 60) -> int | None:
    """The filter's listener, as the launcher hands it over, or None if it never does."""
    sock.settimeout(timeout)
    try:
        _, fds, _, _ = socket.recv_fds(sock, 4, 1)
    except OSError:
        return None
    return fds[0] if fds else None


def kill_all(signum=signal.SIGKILL) -> None:
    for pid in pids():
        try:
            os.kill(pid, signum)
        except OSError:
            pass
