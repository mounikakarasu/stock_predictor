import os
import subprocess


def run(cmd):
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise SystemExit(f"Failed: {cmd}")


def main():
    run("python -m src.update_data")
    run("python -m src.dataset")
    run("python -m src.train")
    run("python -m src.predict")


if __name__ == "__main__":
    main()
