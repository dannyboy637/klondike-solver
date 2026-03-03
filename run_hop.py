#!/usr/bin/env python3
"""Wrapper to run lonecli hop-loop with pseudo-tty for line-buffered output."""
import subprocess
import sys
import signal
import os
import pty
import select

def main():
    if len(sys.argv) < 4:
        print("Usage: run_hop.py <seed_type> <seed> <draw_step> [output_file]")
        sys.exit(1)

    seed_type = sys.argv[1]
    seed = sys.argv[2]
    draw_step = sys.argv[3]
    output_file = sys.argv[4] if len(sys.argv) > 4 else None

    lonecli = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lonelybot/target/release/lonecli")
    cmd = [lonecli, "hop-loop", seed_type, seed, draw_step]

    # Create pseudo-tty so Rust thinks it's writing to terminal (line buffered)
    master_fd, slave_fd = pty.openpty()

    proc = subprocess.Popen(
        cmd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True
    )
    os.close(slave_fd)

    f = open(output_file, "w") if output_file else None

    def handler(sig, frame):
        proc.terminate()
        os.close(master_fd)
        if f:
            f.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    buf = b""
    try:
        while True:
            r, _, _ = select.select([master_fd], [], [], 1.0)
            if r:
                try:
                    data = os.read(master_fd, 4096)
                    if not data:
                        break
                    buf += data
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line_str = line.decode("utf-8", errors="replace").rstrip("\r")
                        print(line_str, flush=True)
                        if f:
                            f.write(line_str + "\n")
                            f.flush()
                except OSError:
                    break
            if proc.poll() is not None:
                break
    except:
        pass
    finally:
        proc.terminate()
        try:
            os.close(master_fd)
        except:
            pass
        if f:
            f.close()

if __name__ == "__main__":
    main()
