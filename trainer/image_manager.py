import asyncio
import json
import os
import uuid

import docker
from docker.errors import APIError
from docker.errors import BuildError
from docker.models.containers import Container

from core.models.payload_models import TrainerProxyRequest
from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
from trainer import constants as cst
from trainer.tasks import complete_task
from trainer.tasks import log_task
from validator.utils.logging import get_all_context_tags
from validator.utils.logging import get_logger
from validator.utils.logging import stream_container_logs
from validator.utils.logging import stream_image_build_logs


logger = get_logger(__name__)


def build_docker_image(
    dockerfile_path: str, context_path: str = ".", is_image_task: bool = False, tag: str = None, no_cache: bool = True
) -> str:
    client: docker.DockerClient = docker.from_env()

    if tag is None:
        tag = f"standalone-image-trainer:{uuid.uuid4()}" if is_image_task else f"standalone-text-trainer:{uuid.uuid4()}"

    logger.info(f"Building Docker image '{tag}', Dockerfile path: {dockerfile_path}, Context Path: {context_path}...")

    try:
        build_output = client.api.build(
            path=context_path,
            dockerfile=dockerfile_path,
            tag=tag,
            nocache=no_cache,
            decode=True,
        )
        stream_image_build_logs(build_output, get_all_context_tags())

        logger.info("Docker image built successfully.")
        return tag
    except (BuildError, APIError) as e:
        logger.error(f"Docker build failed: {str(e)}")
        raise


def delete_image_and_cleanup(tag: str):
    client = docker.from_env()
    try:
        client.images.remove(image=tag, force=True)
        logger.info(f"Deleted Docker image with tag: {tag}")
    except docker.errors.ImageNotFound:
        logger.error(f"No Docker image found with tag: {tag}")
    except Exception as e:
        logger.error(f"Failed to delete image '{tag}': {e}")

    try:
        client.images.prune(filters={"dangling": True})
        client.api.prune_builds()
        logger.info("Cleaned up dangling images and build cache.")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


async def run_trainer_container_image(
    task_id: str,
    tag: str,
    model: str,
    dataset_zip: str,
    model_type: str,
    expected_repo_name: str,
    hours_to_complete: float,
    gpu_ids: list[int] = [0]
) -> Container:
    client: docker.DockerClient = docker.from_env()

    command: list[str] = [
        "--task-id",
        task_id,
        "--model",
        model,
        "--dataset-zip",
        dataset_zip,
        "--model-type",
        model_type,
        "--expected-repo-name",
        expected_repo_name,
        "--hours-to-complete",
        str(hours_to_complete),
    ]

    container_name = f"image-trainer-{uuid.uuid4().hex}"

    try:
        container: Container = client.containers.run(
            image=tag,
            command=command,
            volumes={
                cst.VOLUME_NAMES[0]: {"bind": cst.IMAGE_CONTAINER_SAVE_PATH, "mode": "rw"},
                cst.VOLUME_NAMES[1]: {"bind": "/cache", "mode": "rw"},
            },
            remove=False,
            name=container_name,
            mem_limit=cst.DEFAULT_TRAINING_CONTAINER_MEM_LIMIT,
            nano_cpus=cst.DEFAULT_TRAINING_CONTAINER_NANO_CPUS * 1_000_000_000,
            device_requests=[docker.types.DeviceRequest(device_ids=[str(i) for i in gpu_ids], capabilities=[["gpu"]])],
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            network_mode="none",
            environment={"TRANSFORMERS_CACHE": "/cache/hf_cache"},
            detach=True,
        )

        log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))
        return container
    except Exception as e:
        logger.error(e)
        return e


async def run_trainer_container_text(
    task_id: str,
    tag: str,
    model: str,
    dataset: str,
    dataset_type: InstructTextDatasetType | DpoDatasetType,
    task_type: TaskType,
    file_format: FileFormat,
    expected_repo_name: str,
    hours_to_complete: float,
    gpu_ids: list[int] = [0],
) -> Container:
    client: docker.DockerClient = docker.from_env()

    environment = {"WANDB_MODE": "disabled", "WANDB_DISABLED": "true"}

    command: list[str] = [
        "--task-id",
        task_id,
        "--model",
        model,
        "--dataset",
        dataset,
        "--dataset-type",
        json.dumps(dataset_type.model_dump()),
        "--task-type",
        task_type,
        "--file-format",
        file_format,
        "--expected-repo-name",
        expected_repo_name,
        "--hours-to-complete",
        str(hours_to_complete)
    ]

    container_name = f"text-trainer-{uuid.uuid4().hex}"

    try:
        container: Container = client.containers.run(
            image=tag,
            command=command,
            volumes={
                cst.VOLUME_NAMES[0]: {"bind": cst.TEXT_CONTAINER_SAVE_PATH, "mode": "rw"},
                cst.VOLUME_NAMES[1]: {"bind": "/cache", "mode": "rw"},
            },
            remove=False,
            name=container_name,
            mem_limit=cst.DEFAULT_TRAINING_CONTAINER_MEM_LIMIT,
            nano_cpus=cst.DEFAULT_TRAINING_CONTAINER_NANO_CPUS * 1_000_000_000,
            device_requests=[docker.types.DeviceRequest(device_ids=[str(i) for i in gpu_ids], capabilities=[["gpu"]])],
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            detach=True,
            network_mode="none",
            environment=environment,
        )

        log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))
        return container
    except Exception as e:
        logger.error(e)
        return e


async def create_volumes_if_dont_exist():
    client: docker.DockerClient = docker.from_env()
    volume_names = cst.VOLUME_NAMES
    for volume_name in volume_names:
        try:
            volume = client.volumes.get(volume_name)
        except docker.errors.NotFound:
            volume = client.volumes.create(name=volume_name)
            logger.info(f"Volume '{volume_name}' created.")


def run_downloader_container(
    task_id: str,
    model: str,
    dataset_url: str,
    task_type: TaskType,
    file_format: FileFormat | None = None,
) -> tuple[int, Exception | None]:
    client = docker.from_env()

    command = [
        "--task-id",
        task_id,
        "--model",
        model,
        "--task-type",
        task_type,
        "--dataset",
        dataset_url,
    ]
    if file_format:
        command += ["--file-format", file_format]

    container_name = f"downloader-{task_id}-{str(uuid.uuid4())[:8]}"
    container = None

    try:
        logger.info(f"Starting downloader container: {container_name}")
        container = client.containers.run(
            image=cst.TRAINER_DOWNLOADER_DOCKER_IMAGE,
            name=container_name,
            command=command,
            volumes={cst.VOLUME_NAMES[1]: {"bind": "/cache", "mode": "rw"}},
            remove=False,
            detach=True,
        )

        stream_container_logs(container, get_all_context_tags())

        result = container.wait()
        exit_code = result.get("StatusCode", -1)

        if exit_code == 0:
            logger.info(f"Download completed successfully for task {task_id}")
        else:
            logger.error(f"Download container exited with code {exit_code} for task {task_id}")

        return exit_code, None

    except docker.errors.ContainerError as e:
        logger.error(f"Downloader container failed for task {task_id}: {e}")
        return 1, e

    except Exception as ex:
        logger.error(f"Unexpected error in downloader for task {task_id}: {ex}")
        return 1, ex

    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception as cleanup_err:
                logger.warning(f"Failed to remove container {container_name}: {cleanup_err}")


async def upload_repo_to_hf(
    task_id: str,
    expected_repo_name: str,
    huggingface_token: str,
    huggingface_username: str,
    task_type: TaskType,
    wandb_token: str | None = None,
    path_in_repo: str | None = None,
):
    try:
        client = docker.from_env()

        local_container_folder = (
            f"{cst.IMAGE_CONTAINER_SAVE_PATH}{task_id}/{expected_repo_name}/"
            if task_type == TaskType.IMAGETASK
            else f"{cst.TEXT_CONTAINER_SAVE_PATH}{task_id}/{expected_repo_name}/"
        )

        environment = {
            "HUGGINGFACE_TOKEN": huggingface_token,
            "HUGGINGFACE_USERNAME": huggingface_username,
            "WANDB_TOKEN": wandb_token or "",
            "LOCAL_FOLDER": local_container_folder,
            "TASK_ID": task_id,
            "EXPECTED_REPO_NAME": expected_repo_name,
            "HF_REPO_SUBFOLDER": path_in_repo,
        }

        container_path = cst.IMAGE_CONTAINER_SAVE_PATH if task_type == TaskType.IMAGETASK else cst.TEXT_CONTAINER_SAVE_PATH

        volumes = {
            cst.VOLUME_NAMES[0]: {"bind": container_path, "mode": "rw"},
            cst.VOLUME_NAMES[1]: {"bind": "/cache", "mode": "rw"},
        }

        container_name = f"hf-upload-{uuid.uuid4().hex}"

        logger.info(f"Starting upload container {container_name} for task {task_id}...")

        container = client.containers.run(
            image=cst.HF_UPLOAD_DOCKER_IMAGE,
            environment=environment,
            volumes=volumes,
            detach=True,
            remove=True,
            name=container_name,
        )

        log_task = asyncio.create_task(asyncio.to_thread(stream_container_logs, container, get_all_context_tags()))

    except Exception as e:
        logger.exception(f"Unexpected error during upload_repo_to_hf for task {task_id}: {e}")
        raise


def get_task_type(request: TrainerProxyRequest) -> TaskType:
    training_data = request.training_data

    if isinstance(training_data, TrainRequestImage):
        return TaskType.IMAGETASK

    elif isinstance(training_data, TrainRequestText):
        if isinstance(training_data.dataset_type, DpoDatasetType):
            return TaskType.DPOTASK
        elif isinstance(training_data.dataset_type, InstructTextDatasetType):
            return TaskType.INSTRUCTTEXTTASK
        elif isinstance(training_data.dataset_type, GrpoDatasetType):
            return TaskType.GRPOTASK
        else:
            raise ValueError(f"Unsupported dataset_type for text task: {type(training_data.dataset_type)}")

    raise ValueError(f"Unsupported training_data type: {type(training_data)}")


async def start_training_task(task: TrainerProxyRequest, local_repo_path: str):
    try:
        training_data = task.training_data
        success = False
        container = None
        tag = None
        timeout_seconds = int(training_data.hours_to_complete * 3600)
        task_type = get_task_type(task)
        logger.info(f"Task Type: {task_type}")
        await create_volumes_if_dont_exist()

        dockerfile_path = (
            f"{local_repo_path}/{cst.DEFAULT_IMAGE_DOCKERFILE_PATH}"
            if task_type == TaskType.IMAGETASK
            else f"{local_repo_path}/{cst.DEFAULT_TEXT_DOCKERFILE_PATH}"
        )

        logger.info("Running Cache Download Container")
        await log_task(training_data.task_id, task.hotkey, "Downloading data")

        download_status, exc = await asyncio.to_thread(
            run_downloader_container,
            task_id=training_data.task_id,
            model=training_data.model,
            dataset_url=training_data.dataset_zip if task_type == TaskType.IMAGETASK else training_data.dataset,
            task_type=task_type,
            file_format=getattr(training_data, "file_format", None),
        )

        if download_status == 0:
            message = "Download container completed successfully"
            await log_task(training_data.task_id, task.hotkey, message)
        else:
            message = f"Download container failed with error: {exc}"
            await log_task(training_data.task_id, task.hotkey, message)
            await complete_task(training_data.task_id, task.hotkey, success=False)
            raise RuntimeError(f"Downloader container failed: {exc}")

        tag = await asyncio.to_thread(
            build_docker_image,
            dockerfile_path=dockerfile_path,
            is_image_task=(task_type == TaskType.IMAGETASK),
            context_path=local_repo_path,
        )

        await log_task(training_data.task_id, task.hotkey, f"Docker image built with tag: {tag}")

        if task_type == TaskType.IMAGETASK:
            container = await asyncio.wait_for(
                run_trainer_container_image(
                    task_id=training_data.task_id,
                    tag=tag,
                    model=training_data.model,
                    dataset_zip=training_data.dataset_zip,
                    model_type=training_data.model_type,
                    expected_repo_name=training_data.expected_repo_name,
                    hours_to_complete=training_data.hours_to_complete,
                    gpu_ids=task.gpu_ids,
                ),
                timeout=60,
            )
        else:
            container = await asyncio.wait_for(
                run_trainer_container_text(
                    task_id=training_data.task_id,
                    tag=tag,
                    model=training_data.model,
                    dataset=training_data.dataset,
                    dataset_type=training_data.dataset_type,
                    task_type=task_type,
                    file_format=training_data.file_format,
                    expected_repo_name=training_data.expected_repo_name,
                    hours_to_complete=training_data.hours_to_complete,
                    gpu_ids=task.gpu_ids,
                ),
                timeout=60,
            )

        await log_task(training_data.task_id, task.hotkey, f"Container started: {container.name}")

        await log_task(training_data.task_id, task.hotkey, f"Waiting for container to finish (timeout={timeout_seconds})...")
        wait_task = asyncio.create_task(asyncio.to_thread(container.wait))
        done, pending = await asyncio.wait({wait_task}, timeout=timeout_seconds)
        await log_task(training_data.task_id, task.hotkey, "Container wait completed or timed out.")

        if wait_task in done:
            result = await wait_task
            logger.info(f"Container.wait() returned: {result}")
            status_code = result.get("StatusCode", -1)
            if status_code == 0:
                log_task(training_data.task_id, task.hotkey, "Training completed successfully.")
                success = True
            else:
                await complete_task(training_data.task_id, task.hotkey, success=success)
                await log_task(training_data.task_id, task.hotkey, f"Training failed with status code {status_code}")
        else:
            await log_task(training_data.task_id, task.hotkey, f"Timeout reached ({timeout_seconds}s). Killing container...")
            success = True
            await complete_task(training_data.task_id, task.hotkey, success=success)

    except Exception as e:
        await log_task(training_data.task_id, task.hotkey, f"Fatal error during training: {e}")
        logger.exception(f"Training job failed: {training_data.task_id}")
        await complete_task(training_data.task_id, task.hotkey, success=success)

    finally:
        if container:
            try:
                container.kill()
                container.remove(force=True)
                await log_task(training_data.task_id, task.hotkey, f"Container {container.name} cleaned up.")

            except Exception as cleanup_err:
                await log_task(training_data.task_id, task.hotkey, f"Error during container cleanup: {cleanup_err}")

        logger.info("Cleaning up")
        if tag:
            delete_image_and_cleanup(tag)
            logger.info("Cleaned up Docker resources.")
        else:
            logger.info("No Docker image to clean up.")

        if success:
            try:
                path_in_repo = cst.IMAGE_TASKS_HF_SUBFOLDER_PATH if task_type == TaskType.IMAGETASK else None
                await upload_repo_to_hf(
                    task_id=training_data.task_id,
                    expected_repo_name=training_data.expected_repo_name,
                    huggingface_username=os.getenv("HUGGINGFACE_USERNAME"),
                    huggingface_token=os.getenv("HUGGINGFACE_TOKEN"),
                    task_type=task_type,
                    wandb_token=os.getenv("WANDB_TOKEN", None),
                    path_in_repo=path_in_repo,
                )
                await log_task(training_data.task_id, task.hotkey, "Repo uploaded successfully.")
            except Exception as upload_err:
                await log_task(training_data.task_id, task.hotkey, f"Upload to HuggingFace failed: {upload_err}")
                success = False

        await complete_task(training_data.task_id, task.hotkey, success=success)
