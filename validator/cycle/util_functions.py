import asyncio
import re

from datasets import get_dataset_infos
from fiber import Keypair
from huggingface_hub import HfApi

from core.models.payload_models import TrainRequestImage
from core.models.payload_models import TrainRequestText
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskStatus
from validator.core import constants as cst
from validator.core.models import AnyTextTypeRawTask
from validator.core.models import DpoRawTask
from validator.core.models import GrpoRawTask
from validator.core.models import ImageRawTask
from validator.core.models import ChatRawTask
from validator.core.models import InstructTextRawTask
from validator.tasks.task_prep import prepare_image_task
from validator.tasks.task_prep import prepare_text_task
from validator.utils.logging import get_logger
from validator.utils.minio import async_minio_client


logger = get_logger(__name__)
hf_api = HfApi()


async def get_fake_text_dataset_size(task: AnyTextTypeRawTask) -> int:
    return 100_000


async def get_total_text_dataset_size(task: AnyTextTypeRawTask) -> int:
    if task.file_format == FileFormat.S3:
        if not task.training_data:
            logger.error(f"Training data is missing from task: {task.task_id}")
            raise ValueError(f"Training data is missing from task: {task.task_id}")
        train_bucket_name, train_object_name = async_minio_client.parse_s3_url(task.training_data)
        train_stats = await async_minio_client.get_stats(train_bucket_name, train_object_name)
        train_ds_size = train_stats.size
        if task.test_data:
            test_bucket_name, test_object_name = async_minio_client.parse_s3_url(task.test_data)
            test_stats = await async_minio_client.get_stats(test_bucket_name, test_object_name)
            test_ds_size = test_stats.size
            return train_ds_size + test_ds_size
        else:
            return train_ds_size

    else:
        loop = asyncio.get_running_loop()
        dataset_infos = await loop.run_in_executor(None, get_dataset_infos, task.ds)
        size = sum(info.dataset_size for info in dataset_infos.values() if info.dataset_size)
    return int(size)


def get_model_num_params(model_id: str) -> int:
    try:
        model_info = hf_api.model_info(model_id)
        size = model_info.safetensors.total
        return size
    except Exception as e:
        logger.warning(f"Error getting model size from safetensors: {e}")
        model_size = re.search(r"(\d+)(?=[bB])", model_id)
        model_size = int(model_size.group(1)) * 1_000_000_000 if model_size else None
        logger.info(f"Model size from regex: {model_size}")
        return model_size


async def get_total_image_dataset_size(task: ImageRawTask) -> int:
    if not task.image_text_pairs:
        return 0
    return len(task.image_text_pairs)


async def run_image_task_prep(task: ImageRawTask, keypair: Keypair) -> ImageRawTask:
    test_url, train_url = await prepare_image_task(task.image_text_pairs)
    task.training_data = train_url
    task.test_data = test_url
    task.status = TaskStatus.LOOKING_FOR_NODES
    logger.info(
        "Data creation is complete - now time to find some miners",
    )
    return task


async def run_text_task_prep(task: AnyTextTypeRawTask, keypair: Keypair) -> AnyTextTypeRawTask:
    # Store original dataset name for processing
    original_ds_name = task.ds

    test_data, synth_data, train_data = await prepare_text_task(task, keypair=keypair)
    task.training_data = train_data
    task.status = TaskStatus.LOOKING_FOR_NODES
    task.synthetic_data = synth_data
    task.test_data = test_data

    # Update dataset name after processing if multiple datasets were used
    if original_ds_name and "," in original_ds_name:
        num_datasets = len([ds.strip() for ds in original_ds_name.split(",")])
        task.ds = f"mix of {num_datasets} datasets"
        logger.info(f"Updated dataset name from '{original_ds_name}' to: {task.ds}")

    if isinstance(task, InstructTextRawTask):
        task.field_instruction = cst.STANDARD_INSTRUCT_COLUMN
        task.field_output = cst.STANDARD_OUTPUT_COLUMN
        task.field_input = cst.STANDARD_INPUT_COLUMN if task.field_input else None
        task.field_system = cst.STANDARD_SYSTEM_COLUMN if task.field_system else None
    elif isinstance(task, DpoRawTask):
        task.field_prompt = cst.STANDARD_DPO_PROMPT_COLUMN
        task.field_chosen = cst.STANDARD_DPO_CHOSEN_COLUMN
        task.field_rejected = cst.STANDARD_DPO_REJECTED_COLUMN
        task.field_system = cst.STANDARD_SYSTEM_COLUMN if task.field_system else None
    elif isinstance(task, GrpoRawTask):
        task.field_prompt = cst.STANDARD_GRPO_PROMPT_COLUMN

    logger.info("Data creation is complete - now time to find some miners")
    return task


def prepare_text_task_request(task: AnyTextTypeRawTask) -> TrainRequestText:
    if isinstance(task, InstructTextRawTask):
        dataset_type = InstructTextDatasetType(
            field_system=task.field_system,
            field_input=task.field_input,
            field_output=task.field_output,
            field_instruction=task.field_instruction,
            format=task.format,
            no_input_format=task.no_input_format,
        )
    elif isinstance(task, DpoRawTask):
        dataset_type = DpoDatasetType(
            field_prompt=task.field_prompt,
            field_system=task.field_system,
            field_chosen=task.field_chosen,
            field_rejected=task.field_rejected,
            prompt_format=task.prompt_format,
            chosen_format=task.chosen_format,
            rejected_format=task.rejected_format,
        )
    elif isinstance(task, GrpoRawTask):
        dataset_type = GrpoDatasetType(
            field_prompt=task.field_prompt,
            reward_functions=task.reward_functions,
        )
    elif isinstance(task, ChatRawTask):
        dataset_type = ChatTemplateDatasetType(
            chat_template=task.chat_template,
            chat_column=task.chat_column,
            chat_role_field=task.chat_role_field,
            chat_content_field=task.chat_content_field,
            chat_user_reference=task.chat_user_reference,
            chat_assistant_reference=task.chat_assistant_reference,
        )

    dataset = task.training_data if task.training_data else "dataset error"
    task_request_body = TrainRequestText(
        dataset=dataset,
        model=task.model_id,
        dataset_type=dataset_type,
        file_format=FileFormat.S3,
        task_id=str(task.task_id),
        hours_to_complete=task.hours_to_complete,
    )

    return task_request_body


def prepare_image_task_request(task: ImageRawTask) -> TrainRequestImage:
    return TrainRequestImage(
        model=task.model_id,
        task_id=str(task.task_id),
        hours_to_complete=task.hours_to_complete,
        dataset_zip=task.training_data,
        model_type=task.model_type,
    )
