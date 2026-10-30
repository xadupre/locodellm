"""Selects all idle visible GPUs and assigns distinct verification CPU cores."""

import csv
import os
import subprocess


def select_devices(device, visible_uuids, busy_uuids):
    """Uses every idle visible GPU, without a task-count or eight-device hard cap."""
    if device != "cuda":
        return [device]
    busy = {str(uuid).removeprefix("GPU-") for uuid in busy_uuids}
    devices = [
        f"cuda:{index}"
        for index, uuid in enumerate(visible_uuids)
        if str(uuid).removeprefix("GPU-") not in busy
    ]
    if not devices:
        raise RuntimeError("No idle visible CUDA GPU. Use cpu or an explicit cuda:N to share.")
    return devices


def discover_devices(device):
    """Queries active compute processes before creating any model replicas."""
    import torch

    if device == "cpu":
        return ["cpu"]
    if device != "cuda":
        if not device.startswith("cuda:") or not device[5:].isdigit():
            raise ValueError("device must be cpu, cuda, or cuda:N.")
        if int(device[5:]) >= torch.cuda.device_count():
            raise ValueError(f"Device is not visible: {device}")
        return [device]
    visible = [
        str(torch.cuda.get_device_properties(index).uuid)
        for index in range(torch.cuda.device_count())
    ]
    if not visible:
        raise RuntimeError("CUDA requested, but no GPU is visible.")
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"GPU availability query failed: {result.stderr.strip()}")
    busy = {
        uuid.strip()
        for uuid, pid in csv.reader(result.stdout.splitlines())
        if int(pid.strip()) != os.getpid()
    }
    return select_devices(device, visible, busy)


def verification_cpus(count):
    """Assigns nonoverlapping physical CPU cores to concurrent timing workers."""
    from pathlib import Path

    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = (
            (topology / "physical_package_id").read_text().strip(),
            (topology / "core_id").read_text().strip(),
        )
        cores.setdefault(key, cpu)
    if len(cores) < count:
        raise RuntimeError(f"Need {count} distinct CPU cores for isolated local timing workers.")
    return list(cores.values())[:count]
