import json
import asyncio
from datetime import timedelta
from datetime import datetime
from pathlib import Path
import aiofiles
import docker
import threading

from core.models.utility_models import TaskStatus
from core.models.payload_models import TrainerProxyRequest, TrainerTaskLog
from validator.utils.logging import get_all_context_tags
from validator.utils.logging import get_logger
from validator.utils.logging import stream_container_logs
from trainer import constants as cst

logger = get_logger(__name__)

task_history: list[TrainerTaskLog] = []
TASK_HISTORY_FILE = Path(cst.TASKS_FILE_PATH)


def start_cleanup_loop_in_thread():
    def run():
        asyncio.run(periodically_cleanup_tasks_and_cache())
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    

async def periodically_cleanup_tasks_and_cache(poll_interval_seconds: int = 600):
    while True:
        if len(task_history) > 0:
            now = datetime.utcnow()
            for task in task_history:
                if task.status != TaskStatus.TRAINING or not task.started_at:
                    continue

                timeout = timedelta(hours=task.training_data.hours_to_complete) + timedelta(minutes=cst.STALE_TASK_GRACE_MINUTES)
                deadline = task.started_at + timeout

                if now > deadline:
                    task.status = TaskStatus.FAILURE
                    task.finished_at = now
                    task.logs.append(f"[{now.isoformat()}] Task marked as FAILED due to timeout.")
                    await save_task_history()

            client = docker.from_env()
            abs_task_path = Path(cst.TASKS_FILE_PATH).resolve()

            if abs_task_path.exists():

                logger.info("Starting cleanup container...")

                container = client.containers.run(
                    image=cst.CACHE_CLEANER_DOCKER_IMAGE,
                    volumes={
                        cst.VOLUME_NAMES[0]: {"bind": "/checkpoints", "mode": "rw"},
                        cst.VOLUME_NAMES[1]: {"bind": "/cache", "mode": "rw"},
                        str(abs_task_path): {"bind": "/app/trainer/task_history.json", "mode": "ro"},
                    },
                    remove=True,
                    detach=True
                )

                log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))

                logger.info("Cleanup container finished.")


            await asyncio.sleep(poll_interval_seconds)


async def start_task(task: TrainerProxyRequest) -> tuple[str, str]:
    log_entry = TrainerTaskLog(**task.dict(), status=TaskStatus.TRAINING, started_at=datetime.utcnow(), finished_at=None)
    task_history.append(log_entry)
    await save_task_history()
    return log_entry.training_data.task_id, log_entry.hotkey


async def complete_task(task_id: str, hotkey: str, success: bool = True):
    task = get_task(task_id, hotkey)
    if task is None:
        return
    task.status = TaskStatus.SUCCESS if success else TaskStatus.FAILURE
    task.finished_at = datetime.utcnow()
    await save_task_history()


def get_task(task_id: str, hotkey: str) -> TrainerTaskLog | None:
    for task in task_history:
        if task.training_data.task_id == task_id and task.hotkey == hotkey:
            return task
    return None


async def log_task(task_id: str, hotkey: str, message: str):
    task = get_task(task_id, hotkey)
    if task:
        timestamped_message = f"[{datetime.utcnow().isoformat()}] {message}"
        task.logs.append(timestamped_message)
        await save_task_history()


def get_running_tasks() -> list[TrainerTaskLog]:
    return [t for t in task_history if t.status == TaskStatus.TRAINING]


async def save_task_history():
    async with aiofiles.open(TASK_HISTORY_FILE, "w") as f:
        data = json.dumps([t.model_dump() for t in task_history], indent=2, default=str)
        await f.write(data)


def load_task_history():
    global task_history
    if TASK_HISTORY_FILE.exists():
        with open(TASK_HISTORY_FILE, "r") as f:
            data = json.load(f)
            task_history.clear()
            task_history.extend(TrainerTaskLog(**item) for item in data)
