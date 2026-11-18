"""The container sandbox, as far as it can be tried without root or a container.

tools/cage_check.py tries it whole, as root in a container, from inside.
"""

import errno
import os
import socket
import stat
import sys

import pytest

from stet import cage

ALLOW, NOTIFY = cage.RET_ALLOW, cage.RET_NOTIFY


def decide(nr, *args, arch=cage.AUDIT_ARCH_X86_64):
    return cage.run_bpf(cage.program(), nr, tuple(args) + (0,) * (6 - len(args)), arch)


def test_the_filter_hands_internet_sockets_to_stet_and_refuses_the_rest():
    assert decide(cage.NR_SOCKET, socket.AF_INET, socket.SOCK_STREAM) == NOTIFY
    assert decide(cage.NR_SOCKET, socket.AF_INET6, socket.SOCK_DGRAM) == NOTIFY  # stet refuses UDP when it is asked
    assert decide(cage.NR_SOCKET, socket.AF_UNIX, socket.SOCK_STREAM) == ALLOW
    assert decide(cage.NR_SOCKET, socket.AF_NETLINK, socket.SOCK_RAW) == ALLOW
    for family in (17, 40, 31, 29):  # packet, vsock, bluetooth, CAN
        assert decide(cage.NR_SOCKET, family, socket.SOCK_RAW) == cage.RET_ERRNO | errno.EAFNOSUPPORT


def test_the_filter_keeps_sockets_on_the_loopback_and_out_of_the_kernels_back_doors():
    eperm = cage.RET_ERRNO | errno.EPERM
    # Unbinding the socket from the loopback device, by name or by index.
    assert decide(cage.NR_SETSOCKOPT, 3, cage.SOL_SOCKET, cage.SO_BINDTODEVICE) == eperm
    assert decide(cage.NR_SETSOCKOPT, 3, cage.SOL_SOCKET, cage.SO_BINDTOIFINDEX) == eperm
    assert decide(cage.NR_SETSOCKOPT, 3, cage.SOL_SOCKET, socket.SO_REUSEADDR) == ALLOW
    # TCP Fast Open connects without connect(), where Landlock looks.
    assert decide(cage.NR_SETSOCKOPT, 3, cage.IPPROTO_TCP, cage.TCP_FASTOPEN_CONNECT) == eperm
    assert decide(cage.NR_SETSOCKOPT, 3, cage.IPPROTO_TCP, socket.TCP_NODELAY) == ALLOW
    assert decide(cage.NR_SENDTO, 3, 0, 0, cage.MSG_FASTOPEN) == eperm
    assert decide(cage.NR_SENDMMSG, 3, 0, 0, cage.MSG_FASTOPEN | 0x40) == eperm
    assert decide(cage.NR_SENDMSG, 3, 0, cage.MSG_FASTOPEN) == eperm
    assert decide(cage.NR_SENDTO, 3, 0, 0, 0x40) == ALLOW
    # io_uring makes sockets and sends with none of these checks; the rest reach other processes or outlive the run.
    for nr in (425, 426, 427):
        assert decide(nr) == cage.RET_ERRNO | errno.ENOSYS
    for nr in (101, 248, 249, 250, 298, 310, 311, 321, 323, 438):
        assert decide(nr) == eperm
    assert decide(0) == ALLOW and decide(1) == ALLOW
    # Another architecture's numbering, or x32's, would get round all of it.
    assert decide(0, arch=0x40000003) == cage.RET_KILL
    assert decide(0x40000000 | cage.NR_SOCKET, socket.AF_INET) == cage.RET_KILL


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sockets")
def test_a_socket_stet_makes_for_a_curator_is_tcp_bound_to_the_loopback():
    assert cage.loopback_socket(socket.AF_INET, socket.SOCK_DGRAM, 0) == -errno.EPERM
    assert cage.loopback_socket(socket.AF_INET, socket.SOCK_RAW, 1) == -errno.EPERM
    assert cage.loopback_socket(socket.AF_INET, socket.SOCK_STREAM, 262) == -errno.EPERM  # MPTCP
    assert cage.loopback_socket(socket.AF_UNIX, socket.SOCK_STREAM, 0) == -errno.EPERM
    fd = cage.loopback_socket(socket.AF_INET, socket.SOCK_STREAM | socket.SOCK_NONBLOCK, 0)
    if fd == -errno.EPERM:
        pytest.skip("this kernel lets only privileged processes bind a socket to a device")
    assert fd >= 0
    s = socket.socket(fileno=fd)
    try:
        assert s.getsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, 16).rstrip(b"\0") == b"lo"
        import fcntl

        assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_NONBLOCK  # as the curator asked
        assert not os.get_inheritable(fd)
    finally:
        s.close()


def test_what_the_curators_user_may_do_is_read_from_the_permission_bits():
    def st(mode, uid=0, gid=0):
        return os.stat_result((mode, 0, 0, 1, uid, gid, 0, 0, 0, 0))

    assert cage._may(st(stat.S_IFDIR | 0o755), 0o5)
    assert not cage._may(st(stat.S_IFDIR | 0o700), 0o1)  # /root
    assert cage._may(st(stat.S_IFDIR | 0o700, uid=cage.UID), 0o7)  # its own /out
    assert cage._may(st(stat.S_IFREG | 0o640, gid=cage.GID), 0o4)
    assert not cage._may(st(stat.S_IFREG | 0o600), 0o4)  # a key file


def test_the_container_sandbox_is_not_made_where_it_does_not_belong(monkeypatch):
    if os.geteuid() == 0:
        monkeypatch.setattr(os, "geteuid", lambda: 1000)
    ok, why = cage.available()
    assert not ok and why


def test_what_listens_on_this_machine_is_kept_from_the_curator(tmp_path, monkeypatch):
    table = (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        "   0: 0B00007F:A93F 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 1\n"
        "   1: 00000000:0016 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 2\n"
        "   2: 0100007F:1F90 0100007F:D431 01 00000000:00000000 00:00000000 00000000  1000        0 3\n"
    )
    real = cage.Path

    class FakePath(type(real())):
        def read_text(self, *a, **k):
            if str(self) == "/proc/net/tcp":
                return table
            if str(self) == "/proc/net/tcp6":
                raise OSError
            return super().read_text(*a, **k)

    monkeypatch.setattr(cage, "Path", FakePath)
    # Docker's resolver behind 127.0.0.11, and sshd; not a connection that is merely established.
    assert cage.listening_ports() == {0xA93F, 22}
