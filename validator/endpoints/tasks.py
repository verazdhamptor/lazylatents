from datetime import datetime
from datetime import timedelta
from uuid import UUID

from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import Response

from core.models.payload_models import AllOfNodeResults
from core.models.payload_models import AnyTypeTaskDetails
from core.models.payload_models import LeaderboardRow
from core.models.payload_models import NewTaskRequestDPO
from core.models.payload_models import NewTaskRequestGrpo
from core.models.payload_models import NewTaskRequestChat
from core.models.payload_models import NewTaskRequestImage
from core.models.payload_models import NewTaskRequestInstructText
from core.models.payload_models import NewTaskResponse
from core.models.payload_models import NewTaskWithCustomDatasetRequest
from core.models.payload_models import NewTaskWithFixedDatasetsRequest
from core.models.payload_models import TaskResultResponse
from core.models.utility_models import FileFormat
from core.models.utility_models import MinerTaskResult
from core.models.utility_models import TaskMinerResult
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from validator.core.config import Config
from validator.core.constants import MAX_CONCURRENT_JOBS
from validator.core.dependencies import get_api_key
from validator.core.dependencies import get_config
from validator.core.models import DpoRawTask
from validator.core.models import GrpoRawTask
from validator.core.models import ImageRawTask
from validator.core.models import ChatRawTask
from validator.core.models import InstructTextRawTask
from validator.core.models import NetworkStats
from validator.core.models import DetailedNetworkStats
from validator.db.sql import submissions_and_scoring as submissions_and_scoring_sql
from validator.db.sql import tasks as task_sql
from validator.db.sql.nodes import get_all_nodes
from validator.utils.logging import get_logger
from validator.utils.util import convert_task_to_task_details
from validator.utils.util import hide_sensitive_data_till_finished


logger = get_logger(__name__)


TASKS_CREATE_ENDPOINT_INSTRUCT_TEXT = "/v1/tasks/create"  # TODO: change to create_text after FE changes
TASKS_CREATE_ENDPOINT_IMAGE = "/v1/tasks/create_image"
CREATE_TEXT_TASK_WITH_CUSTOM_DATASET_ENDPOINT = (
    "/v1/tasks/create_custom_dataset_text"  # TODO: this is just for instruct text tasks
)
TASKS_CREATE_ENDPOINT_DPO = "/v1/tasks/create_dpo"
TASKS_CREATE_ENDPOINT_CHAT = "/v1/tasks/create_chat"
TASKS_CREATE_ENDPOINT_GRPO = "/v1/tasks/create_grpo"
TASKS_CREATE_WITH_FIXED_DATASETS_ENDPOINT = "/v1/tasks/create_with_fixed_datasets"  # TODO: this is just for instruct text tasks
GET_TASKS_BY_ACCOUNT_ENDPOINT = "/v1/tasks/account/{account_id}"
GET_TASK_DETAILS_ENDPOINT = "/v1/tasks/{task_id}"
GET_TASKS_RESULTS_ENDPOINT = "/v1/tasks/breakdown/{task_id}"
GET_NETWORK_STATUS = "/v1/network/status"
GET_NODE_RESULTS_ENDPOINT = "/v1/tasks/node_results/{hotkey}"
DELETE_TASK_ENDPOINT = "/v1/tasks/delete/{task_id}"
LEADERBOARD_ENDPOINT = "/v1/leaderboard"
COMPLETED_ORGANIC_TASKS_ENDPOINT = "/v1/tasks/organic/completed"
GET_NETWORK_DETAILED_STATUS = "/v1/network/detailed_status"
UPDATE_TRAINING_REPO_BACKUP_ENDPOINT = "/v1/tasks/{task_id}/training_repo_backup"
UPDATE_RESULT_MODEL_NAME_ENDPOINT = "/v1/tasks/{task_id}/result_model_name"


async def delete_task(
    task_id: UUID,
    config: Config = Depends(get_config),
) -> Response:
    task = await task_sql.get_task(task_id, config.psql_db)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    await task_sql.delete_task(task_id, config.psql_db)
    return Response(content="Success")


async def create_task_dpo(
    request: NewTaskRequestDPO,
    config: Config = Depends(get_config),
) -> NewTaskResponse:
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=request.hours_to_complete)

    task = DpoRawTask(
        model_id=request.model_repo,
        ds=request.ds_repo,
        file_format=request.file_format,
        field_prompt=request.field_prompt,
        field_system=request.field_system,
        field_chosen=request.field_chosen,
        field_rejected=request.field_rejected,
        prompt_format=request.prompt_format,
        chosen_format=request.chosen_format,
        rejected_format=request.rejected_format,
        is_organic=True,
        status=TaskStatus.PENDING,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=request.hours_to_complete,
        account_id=request.account_id,
        task_type=TaskType.DPOTASK,
        result_model_name=request.result_model_name,
    )

    task = await task_sql.add_task(task, config.psql_db)

    logger.info(f"Task of type {task.task_type} created: {task.task_id}")
    return NewTaskResponse(success=True, task_id=task.task_id, created_at=task.created_at, account_id=task.account_id)


async def create_task_grpo(
    request: NewTaskRequestGrpo,
    config: Config = Depends(get_config),
) -> NewTaskResponse:
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=request.hours_to_complete)

    task = GrpoRawTask(
        model_id=request.model_repo,
        ds=request.ds_repo,
        file_format=request.file_format,
        field_prompt=request.field_prompt,
        reward_functions=request.reward_functions,
        is_organic=True,
        status=TaskStatus.PENDING,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=request.hours_to_complete,
        account_id=request.account_id,
        task_type=TaskType.GRPOTASK,
        result_model_name=request.result_model_name,
    )

    task = await task_sql.add_task(task, config.psql_db)

    logger.info(f"Task of type {task.task_type} created: {task.task_id}")
    return NewTaskResponse(success=True, task_id=task.task_id, created_at=task.created_at, account_id=task.account_id)


async def create_task_chat(
    request: NewTaskRequestChat,
    config: Config = Depends(get_config),
) -> NewTaskResponse:
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=request.hours_to_complete)

    task = ChatRawTask(
        model_id=request.model_repo,
        ds=request.ds_repo,
        file_format=request.file_format,
        chat_template=request.chat_template,
        chat_column=request.chat_column,
        chat_role_field=request.chat_role_field,
        chat_content_field=request.chat_content_field,
        chat_user_reference=request.chat_user_reference,
        chat_assistant_reference=request.chat_assistant_reference,
        is_organic=True,
        status=TaskStatus.PENDING,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=request.hours_to_complete,
        account_id=request.account_id,
        task_type=TaskType.CHATTASK,
        result_model_name=request.result_model_name,
    )

    task = await task_sql.add_task(task, config.psql_db)

    logger.info(f"Task of type {task.task_type} created: {task.task_id}")
    return NewTaskResponse(success=True, task_id=task.task_id, created_at=task.created_at, account_id=task.account_id)


async def create_task_instruct_text(
    request: NewTaskRequestInstructText,
    config: Config = Depends(get_config),
) -> NewTaskResponse:
    existing_tasks = await task_sql.get_successful_matching_tasks(
        request.model_repo,
        request.ds_repo,
        request.field_instruction,
        request.field_input,
        request.field_output,
        config.psql_db,
    )

    existing_tasks = [task for task in existing_tasks if task.training_repo_backup or task.trained_model_repository]

    if existing_tasks:
        existing_task = existing_tasks[0]
        logger.info(f"Found matching successful task {existing_task.task_id}, reusing results")
        result_repo = None
        if existing_task.training_repo_backup:
            result_repo = existing_task.training_repo_backup
        else:
            result_repo = existing_task.trained_model_repository

        if not result_repo:
            raise HTTPException(status_code=400, detail="Found matching task but no result repo found")

        new_task = InstructTextRawTask(
            account_id=request.account_id,
            model_id=request.model_repo,
            ds=request.ds_repo,
            field_system=existing_task.field_system,
            field_instruction=existing_task.field_instruction,
            field_input=existing_task.field_input,
            field_output=existing_task.field_output,
            format=existing_task.format,
            no_input_format=existing_task.no_input_format,
            system_format=existing_task.system_format,
            status=TaskStatus.SUCCESS,
            is_organic=True,
            created_at=datetime.utcnow(),
            termination_at=datetime.utcnow(),
            hours_to_complete=request.hours_to_complete,
            training_repo_backup=result_repo,
            test_data=existing_task.test_data,
            training_data=existing_task.training_data,
            synthetic_data=existing_task.synthetic_data,
            task_type=TaskType.INSTRUCTTEXTTASK,
            result_model_name=existing_task.result_model_name,
            file_format=existing_task.file_format,
            started_at=datetime.utcnow(),
        )

        new_task = await task_sql.add_task(new_task, config.psql_db)
        return NewTaskResponse(
            success=True, task_id=new_task.task_id, created_at=new_task.created_at, account_id=new_task.account_id
        )

    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=request.hours_to_complete)

    # if there are any queued jobs that are organic we can't accept any more to avoid overloading the network
    #    queued_tasks = await task_sql.get_tasks_with_status(TaskStatus.DELAYED, config.psql_db, include_not_ready_tasks=True)
    #    if len(queued_tasks) > 0:
    #        logger.info("We already have some queued organic jobs, we can't a accept any more")
    #        return NewTaskResponse(success=False, task_id=None)

    task = InstructTextRawTask(
        model_id=request.model_repo,
        ds=request.ds_repo,
        file_format=request.file_format,
        field_system=request.field_system,
        field_instruction=request.field_instruction,
        field_input=request.field_input,
        field_output=request.field_output,
        format=request.format,
        no_input_format=request.no_input_format,
        is_organic=True,
        status=TaskStatus.PENDING,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=request.hours_to_complete,
        account_id=request.account_id,
        task_type=TaskType.INSTRUCTTEXTTASK,
        result_model_name=request.result_model_name,
    )

    task = await task_sql.add_task(task, config.psql_db)

    logger.info(f"Task of type {task.task_type} created: {task.task_id}")
    return NewTaskResponse(success=True, task_id=task.task_id, created_at=task.created_at, account_id=task.account_id)


async def create_task_image(
    request: NewTaskRequestImage,
    config: Config = Depends(get_config),
) -> NewTaskResponse:
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=request.hours_to_complete)

    # if there are any queued jobs that are organic we can't accept any more to avoid overloading the network
    #    queued_tasks = await task_sql.get_tasks_with_status(TaskStatus.DELAYED, config.psql_db, include_not_ready_tasks=True)
    #    if len(queued_tasks) > 0:
    #        logger.info("We already have some queued organic jobs, we can't a accept any more")
    #        return NewTaskResponse(success=False, task_id=None)

    task = ImageRawTask(
        model_id=request.model_repo,
        image_text_pairs=request.image_text_pairs,
        ds=request.ds_id,
        is_organic=True,
        status=TaskStatus.PENDING,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=request.hours_to_complete,
        account_id=request.account_id,
        task_type=TaskType.IMAGETASK,
        result_model_name=request.result_model_name,
        model_type=request.model_type,
    )

    task = await task_sql.add_task(task, config.psql_db)

    logger.info(f"Task of type {task.task_type} created: {task.task_id}")
    return NewTaskResponse(success=True, task_id=task.task_id, created_at=task.created_at, account_id=task.account_id)


async def create_text_task_with_custom_dataset(
    request: NewTaskWithCustomDatasetRequest,
    config: Config = Depends(get_config),
) -> NewTaskResponse:
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=request.hours_to_complete)

    task = InstructTextRawTask(
        model_id=request.model_repo,
        ds=request.ds_repo or "custom",
        file_format=request.file_format if request.ds_repo else FileFormat.S3,
        field_system=request.field_system,
        field_instruction=request.field_instruction,
        field_input=request.field_input,
        field_output=request.field_output,
        format=request.format,
        is_organic=True,
        no_input_format=request.no_input_format,
        status=TaskStatus.PENDING,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=request.hours_to_complete,
        account_id=request.account_id,
        task_type=TaskType.INSTRUCTTEXTTASK,
        result_model_name=request.result_model_name,
        training_data=request.training_data,
        test_data=request.test_data,
    )

    task = await task_sql.add_task(task, config.psql_db)
    logger.info(f"Task of type {task.task_type} created: {task.task_id}")
    return NewTaskResponse(success=True, task_id=task.task_id, created_at=task.created_at, account_id=task.account_id)


async def create_task_with_fixed_datasets(
    request: NewTaskWithFixedDatasetsRequest,
    config: Config = Depends(get_config),
) -> NewTaskResponse:
    current_time = datetime.utcnow()
    end_timestamp = current_time + timedelta(hours=request.hours_to_complete)

    task = InstructTextRawTask(
        model_id=request.model_repo,
        ds=request.ds_repo or request.training_data,
        file_format=request.file_format if request.ds_repo else FileFormat.S3,
        field_system=request.field_system,
        field_instruction=request.field_instruction,
        field_input=request.field_input,
        field_output=request.field_output,
        format=request.format,
        is_organic=True,
        no_input_format=request.no_input_format,
        status=TaskStatus.LOOKING_FOR_NODES,
        created_at=current_time,
        termination_at=end_timestamp,
        hours_to_complete=request.hours_to_complete,
        account_id=request.account_id,
        result_model_name=request.result_model_name,
    )

    # NOTE: feels weird to add the task and then update it immediately
    await task_sql.add_task(task, config.psql_db)
    task.training_data = request.training_data
    task.synthetic_data = request.synthetic_data
    task.test_data = request.test_data
    await task_sql.update_task(task, config.psql_db)

    logger.info(task.task_id)
    return NewTaskResponse(success=True, task_id=task.task_id, created_at=task.created_at, account_id=task.account_id)


async def get_node_results(
    hotkey: str,
    config: Config = Depends(get_config),
) -> AllOfNodeResults:
    try:
        logger.info(f"The hotkey is {hotkey}")
        miner_results = [
            TaskMinerResult(**result)
            for result in await submissions_and_scoring_sql.get_all_scores_for_hotkey(hotkey, config.psql_db)
        ]
    except Exception as e:
        logger.info(e)
        raise HTTPException(status_code=404, detail="Hotkey not found")
    return AllOfNodeResults(success=True, hotkey=hotkey, task_results=miner_results)


async def get_task_details_by_account(
    account_id: UUID,
    limit: int = 100,
    page: int = 1,
    config: Config = Depends(get_config),
) -> list[AnyTypeTaskDetails]:
    offset = (page - 1) * limit
    tasks = await task_sql.get_tasks_by_account_id(config.psql_db, account_id, limit, offset)
    tasks = [hide_sensitive_data_till_finished(task) for task in tasks]

    return [convert_task_to_task_details(task) for task in tasks]


async def get_task_details(
    task_id: UUID,
    config: Config = Depends(get_config),
) -> AnyTypeTaskDetails:
    task = await task_sql.get_task_by_id(task_id, config.psql_db)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    task = hide_sensitive_data_till_finished(task)
    return convert_task_to_task_details(task)


async def get_miner_breakdown(
    task_id: UUID,
    config: Config = Depends(get_config),
) -> TaskResultResponse:
    try:
        results = await submissions_and_scoring_sql.get_all_scores_and_losses_for_task(task_id, config.psql_db)
        miner_results = [MinerTaskResult(**result) for result in results]
    except Exception as e:
        logger.info(e)
        raise HTTPException(status_code=404, detail="Task not found.")
    return TaskResultResponse(id=task_id, miner_results=miner_results)


async def get_leaderboard(
    config: Config = Depends(get_config),
) -> list[LeaderboardRow]:
    nodes = await get_all_nodes(config.psql_db)
    if not nodes:
        return []

    hotkeys = [node.hotkey for node in nodes]
    try:
        all_stats = await submissions_and_scoring_sql.get_all_node_stats_batched(hotkeys, config.psql_db)
        return [
            LeaderboardRow(hotkey=node.hotkey, stats=all_stats.get(node.hotkey)) for node in nodes if node.hotkey in all_stats
        ]
    except Exception as e:
        logger.error(f"Error processing leaderboard stats: {e}")
        return []


async def get_network_status(
    config: Config = Depends(get_config),
) -> NetworkStats:
    try:
        logger.debug("IN get network status")
        current_task_stats = await task_sql.get_current_task_stats(config.psql_db)
        if current_task_stats.number_of_jobs_training >= MAX_CONCURRENT_JOBS:
            current_task_stats.job_can_be_made = False
        return current_task_stats
    except Exception as e:
        logger.info(f"There was an issue with getting training status {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_network_detailed_status(
    config: Config = Depends(get_config),
) -> DetailedNetworkStats:
    try:
        logger.debug("IN get network detailed status")
        detailed_stats = await task_sql.get_detailed_task_stats(config.psql_db)
        if detailed_stats.number_of_jobs_training >= MAX_CONCURRENT_JOBS:
            detailed_stats.job_can_be_made = False
        return detailed_stats
    except Exception as e:
        logger.info(f"There was an issue with getting detailed status {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_completed_organic_tasks(
    hours: int | None = Query(default=None, description="Number of hours to look back for completed organic tasks", ge=1),
    task_type: TaskType | None = Query(
        default=None,
        description=f"Filter by task type: {', '.join([t.value for t in TaskType])}",
    ),
    search_model_name: str | None = Query(default=None, description="Search term to filter models by name"),
    limit: int = Query(default=100, description="Number of tasks per page", ge=1),
    page: int = Query(default=1, description="Page number", ge=1),
    config: Config = Depends(get_config),
) -> list[AnyTypeTaskDetails]:
    """Get completed organic tasks with optional time filter and task type filter"""
    tasks = await task_sql.get_completed_organic_tasks(
        config.psql_db,
        hours=hours,
        task_type=task_type,
        search_model_name=search_model_name,
        limit=limit,
        offset=(page - 1) * limit,
    )

    return [convert_task_to_task_details(task) for task in tasks]


async def update_training_repo_backup(
    task_id: UUID,
    training_repo: str,
    config: Config = Depends(get_config),
) -> Response:
    task = await task_sql.get_task(task_id, config.psql_db)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    task.training_repo_backup = training_repo
    await task_sql.update_task(task, config.psql_db)
    return Response(status_code=200)


async def update_result_model_name(
    task_id: UUID,
    result_model_name: str,
    config: Config = Depends(get_config),
) -> Response:
    task = await task_sql.get_task(task_id, config.psql_db)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    task.result_model_name = result_model_name
    await task_sql.update_task(task, config.psql_db)
    return Response(status_code=200)


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Gradients On Demand"], dependencies=[Depends(get_api_key)])
    router.add_api_route(TASKS_CREATE_ENDPOINT_INSTRUCT_TEXT, create_task_instruct_text, methods=["POST"])
    router.add_api_route(TASKS_CREATE_ENDPOINT_IMAGE, create_task_image, methods=["POST"])
    router.add_api_route(TASKS_CREATE_ENDPOINT_DPO, create_task_dpo, methods=["POST"])
    router.add_api_route(TASKS_CREATE_ENDPOINT_CHAT, create_task_chat, methods=["POST"])
    # router.add_api_route(TASKS_CREATE_ENDPOINT_GRPO, create_task_grpo, methods=["POST"])
    router.add_api_route(TASKS_CREATE_WITH_FIXED_DATASETS_ENDPOINT, create_task_with_fixed_datasets, methods=["POST"])
    router.add_api_route(CREATE_TEXT_TASK_WITH_CUSTOM_DATASET_ENDPOINT, create_text_task_with_custom_dataset, methods=["POST"])
    router.add_api_route(GET_TASK_DETAILS_ENDPOINT, get_task_details, methods=["GET"])
    router.add_api_route(DELETE_TASK_ENDPOINT, delete_task, methods=["DELETE"])
    router.add_api_route(GET_TASKS_RESULTS_ENDPOINT, get_miner_breakdown, methods=["GET"])
    router.add_api_route(GET_NODE_RESULTS_ENDPOINT, get_node_results, methods=["GET"])
    router.add_api_route(GET_TASKS_BY_ACCOUNT_ENDPOINT, get_task_details_by_account, methods=["GET"])
    router.add_api_route(LEADERBOARD_ENDPOINT, get_leaderboard, methods=["GET"])
    router.add_api_route(GET_NETWORK_STATUS, get_network_status, methods=["GET"])
    router.add_api_route(GET_NETWORK_DETAILED_STATUS, get_network_detailed_status, methods=["GET"])
    router.add_api_route(COMPLETED_ORGANIC_TASKS_ENDPOINT, get_completed_organic_tasks, methods=["GET"])
    router.add_api_route(UPDATE_TRAINING_REPO_BACKUP_ENDPOINT, update_training_repo_backup, methods=["PUT"])
    router.add_api_route(UPDATE_RESULT_MODEL_NAME_ENDPOINT, update_result_model_name, methods=["PUT"])
    return router
