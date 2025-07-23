import ast
import hashlib
from datetime import datetime
from uuid import UUID
from uuid import uuid4

from fiber.logging_utils import get_logger
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from core import constants as cst

from core.models.utility_models import DpoDatasetType
from core.models.utility_models import TextDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import ChatTemplateDatasetType
from core.models.utility_models import ImageModelType
from core.models.utility_models import ImageTextPair
from core.models.utility_models import JobStatus
from core.models.utility_models import MinerTaskResult
from core.models.utility_models import RewardFunction
from core.models.utility_models import TaskMinerResult
from core.models.utility_models import TaskStatus
from core.models.utility_models import TaskType
from core.models.utility_models import TextDatasetType
from validator.core.models import AllNodeStats


logger = get_logger(__name__)


class MinerTaskOffer(BaseModel):
    ds_size: int | None = None
    model: str
    hours_to_complete: float
    task_id: str
    task_type: TaskType
    model_params_count: int | None = None

    model_config = ConfigDict(protected_namespaces=())


class TrainRequest(BaseModel):
    model: str = Field(..., description="Name or path of the model to be trained", min_length=1)
    task_id: str
    hours_to_complete: float
    expected_repo_name: str | None = None


class TrainRequestText(TrainRequest):
    dataset: str = Field(
        ...,
        description="Path to the dataset file or Hugging Face dataset name",
        min_length=1,
    )
    dataset_type: TextDatasetType
    file_format: FileFormat


class TrainRequestGrpo(TrainRequest):
    dataset: str = Field(
        ...,
        description="Path to the dataset file or Hugging Face dataset name",
        min_length=1,
    )
    dataset_type: GrpoDatasetType
    file_format: FileFormat


class TrainRequestImage(TrainRequest):
    model_config = ConfigDict(protected_namespaces=())
    dataset_zip: str = Field(
        ...,
        description="Link to dataset zip file",
        min_length=1,
    )
    model_type: ImageModelType = ImageModelType.SDXL


class TrainerProxyRequest(BaseModel):
    training_data: TrainRequestImage | TrainRequestText
    github_repo: str
    gpu_ids: list[int]
    hotkey: str
    github_branch: str | None = None
    github_commit_hash: str | None = None


class TrainerTaskLog(TrainerProxyRequest):
    status: TaskStatus
    started_at: datetime | None
    finished_at: datetime | None
    logs: list[str] = []


class TrainResponse(BaseModel):
    message: str
    task_id: UUID


class TrainingRepoResponse(BaseModel):
    github_repo: str = Field(..., description="The GitHub repository URL")
    commit_hash: str = Field(..., description="The commit hash of the repository")


class JobStatusPayload(BaseModel):
    task_id: UUID


class JobStatusResponse(BaseModel):
    task_id: UUID
    status: JobStatus


class EvaluationRequest(TrainRequest):
    original_model: str


class EvaluationRequestDiffusion(BaseModel):
    test_split_url: str
    original_model_repo: str
    models: list[str]


class DiffusionLosses(BaseModel):
    text_guided_losses: list[float]
    no_text_losses: list[float]


class EvaluationResultImage(BaseModel):
    eval_loss: DiffusionLosses | float
    is_finetune: bool | None = None


class EvaluationResultText(BaseModel):
    is_finetune: bool
    eval_loss: float


class DockerEvaluationResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    results: dict[str, EvaluationResultText | EvaluationResultImage | Exception]
    base_model_params_count: int = 0


class MinerTaskResponse(BaseModel):
    message: str
    accepted: bool


class DpoDatasetColumnsResponse(BaseModel):
    field_prompt: str
    field_chosen: str | None = None
    field_rejected: str | None = None


class InstructTextDatasetColumnsResponse(BaseModel):
    field_instruction: str
    field_input: str | None = None
    field_output: str | None = None


class NewTaskRequest(BaseModel):
    account_id: UUID
    hours_to_complete: float = Field(..., description="The number of hours to complete the task", examples=[1])
    result_model_name: str | None = Field(None, description="The name to give to a model that is created by this task")


class NewTaskRequestInstructText(NewTaskRequest):
    field_instruction: str = Field(..., description="The column name for the instruction", examples=["instruction"])
    field_input: str | None = Field(None, description="The column name for the input", examples=["input"])
    field_output: str | None = Field(None, description="The column name for the output", examples=["output"])
    field_system: str | None = Field(None, description="The column name for the system (prompt)", examples=["system"])

    ds_repo: str = Field(..., description="The repository for the dataset", examples=["yahma/alpaca-cleaned"])
    file_format: FileFormat = Field(
        FileFormat.HF, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )
    model_repo: str = Field(..., description="The repository for the model", examples=["Qwen/Qwen2.5-Coder-32B-Instruct"])
    format: None = None
    no_input_format: None = None

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    def convert_empty_strings(cls, values: dict) -> dict:
        string_fields = ["field_instruction", "field_input", "field_output", "field_system"]
        for field in string_fields:
            if field in values and isinstance(values[field], str):
                values[field] = values[field].strip() or None
        return values


class NewTaskRequestChat(NewTaskRequest):
    chat_template: str = Field(..., description="The chat template of the dataset", examples=["chatml"])
    chat_column: str | None = Field(None, description="The column name containing the conversations", examples=["conversations"])
    chat_role_field: str | None = Field(None, description="The column name for the role", examples=["from"])
    chat_content_field: str | None = Field(None, description="The column name for the content", examples=["value"])
    chat_user_reference: str | None = Field(None, description="The user reference", examples=["user"])
    chat_assistant_reference: str | None = Field(None, description="The assistant reference", examples=["assistant"])

    ds_repo: str = Field(..., description="The repository for the dataset", examples=["Magpie-Align/Magpie-Pro-300K-Filtered"])
    file_format: FileFormat = Field(
        FileFormat.HF, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )
    model_repo: str = Field(..., description="The repository for the model", examples=["Qwen/Qwen2.5-Coder-32B-Instruct"])

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    def convert_empty_strings(cls, values):
        string_fields = [
            "chat_column",
            "chat_role_field",
            "chat_content_field",
            "chat_user_reference",
            "chat_assistant_reference",
        ]

        for field in string_fields:
            if field in values and isinstance(values[field], str):
                values[field] = values[field].strip() or None
        return values


class NewTaskRequestDPO(NewTaskRequest):
    field_prompt: str = Field(..., description="The column name for the prompt", examples=["prompt"])
    field_system: str | None = Field(None, description="The column name for the system (prompt)", examples=["system"])
    field_chosen: str = Field(..., description="The column name for the chosen response", examples=["chosen"])
    field_rejected: str = Field(..., description="The column name for the rejected response", examples=["rejected"])

    prompt_format: str | None = Field(None, description="The format of the prompt", examples=["{system} {prompt}"])
    chosen_format: str | None = Field(None, description="The format of the chosen response", examples=["{chosen} <|endoftext|>"])
    rejected_format: str | None = Field(
        None, description="The format of the rejected response", examples=["{rejected} <|endoftext|>"]
    )

    ds_repo: str = Field(..., description="The repository for the dataset", examples=["Intel/orca_dpo_pairs"])
    file_format: FileFormat = Field(
        FileFormat.HF, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )
    model_repo: str = Field(..., description="The repository for the model", examples=["Qwen/Qwen2.5-Coder-32B-Instruct"])

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    def convert_empty_strings(cls, values: dict) -> dict:
        string_fields = ["field_prompt", "field_system", "field_chosen", "field_rejected"]
        for field in string_fields:
            if field in values and isinstance(values[field], str):
                values[field] = values[field].strip() or None
        return values


class NewTaskRequestGrpo(NewTaskRequest):
    field_prompt: str = Field(..., description="The column name for the prompt", examples=["prompt"])

    ds_repo: str = Field(..., description="The repository for the dataset", examples=["trl-lib/tldr"])
    file_format: FileFormat = Field(
        FileFormat.HF, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )
    model_repo: str = Field(..., description="The repository for the model", examples=["Qwen/Qwen2.5-Coder-32B-Instruct"])

    reward_functions: list[RewardFunction]

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    def convert_empty_strings(cls, values: dict) -> dict:
        string_fields = ["field_prompt"]
        for field in string_fields:
            if field in values and isinstance(values[field], str):
                values[field] = values[field].strip() or None
        return values

    @model_validator(mode="after")
    def validate_reward_lists(self) -> "NewTaskRequestGrpo":
        if len(self.reward_functions) == 0:
            raise ValueError("reward_functions must not be empty")
        return self

    @model_validator(mode="after")
    def validate_reward_functions(self) -> "NewTaskRequestGrpo":
        for reward_function in self.reward_functions:
            try:
                # Check if it's valid Python code
                parsed = ast.parse(reward_function.reward_func)

                # Check if it contains a function definition
                function_found = False
                for node in ast.walk(parsed):
                    if isinstance(node, ast.FunctionDef):
                        function_found = True
                        arg_names = [arg.arg for arg in node.args.args]
                        has_completions = "completions" in arg_names
                        has_kwargs = node.args.kwarg is not None

                        if not has_completions:
                            raise ValueError(f"Reward function {node.name} must have a 'completions' parameter")
                        if not has_kwargs:
                            raise ValueError(f"Reward function {node.name} must have a '**kwargs' parameter")

                        if reward_function.is_generic is None:
                            allowed_params = {"completions", "prompts"}
                            reward_function.is_generic = set(arg_names) <= allowed_params

                        if reward_function.func_hash is None:
                            reward_function.func_hash = hashlib.sha256(reward_function.reward_func.encode()).hexdigest()

                        break

                if not function_found:
                    raise ValueError("Each reward function must be a proper Python function")

            except Exception as e:
                raise ValueError(f"Invalid Python syntax: {reward_function.reward_func[:50]}... {e}")

        return self


class NewTaskRequestImage(NewTaskRequest):
    model_config = ConfigDict(protected_namespaces=())
    model_repo: str = Field(..., description="The model repository to use")
    image_text_pairs: list[ImageTextPair] = Field(
        ...,
        description="List of image and text file pairs",
        min_length=cst.MIN_IMAGE_TEXT_PAIRS,
        max_length=cst.MAX_IMAGE_TEXT_PAIRS,
    )
    ds_id: str = Field(
        default_factory=lambda: str(uuid4()), description="A ds name. The actual dataset is provided via the image_text_pairs"
    )
    model_type: ImageModelType = ImageModelType.SDXL


class NewTaskWithFixedDatasetsRequest(NewTaskRequestInstructText):
    ds_repo: str | None = Field(None, description="Optional: The original repository of the dataset")
    file_format: FileFormat = Field(
        FileFormat.S3, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )
    training_data: str = Field(..., description="The prepared training dataset")
    synthetic_data: str = Field(..., description="The prepared synthetic dataset")
    test_data: str = Field(..., description="The prepared test dataset")


class NewTaskWithCustomDatasetRequest(NewTaskRequestInstructText):
    ds_repo: str | None = Field(None, description="Optional: The original repository of the dataset")
    training_data: str = Field(..., description="The prepared training dataset")
    test_data: str | None = Field(None, description="The prepared test dataset")
    file_format: FileFormat = Field(
        FileFormat.S3, description="The format of the dataset", examples=[FileFormat.HF, FileFormat.S3]
    )


class NewTaskResponse(BaseModel):
    success: bool = Field(..., description="Whether the task was created successfully")
    task_id: UUID | None = Field(..., description="The ID of the task")
    created_at: datetime = Field(..., description="The creation time of the task")
    account_id: UUID | None = Field(..., description="The account ID who owns the task")


class TaskResultResponse(BaseModel):
    id: UUID
    miner_results: list[MinerTaskResult] | None


class AllOfNodeResults(BaseModel):
    success: bool
    hotkey: str
    task_results: list[TaskMinerResult] | None


class TaskDetails(BaseModel):
    id: UUID
    account_id: UUID
    status: TaskStatus
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    hours_to_complete: float
    trained_model_repository: str | None
    task_type: TaskType
    result_model_name: str | None = None


class InstructTextTaskDetails(TaskDetails):
    task_type: TaskType = TaskType.INSTRUCTTEXTTASK
    base_model_repository: str
    ds_repo: str

    field_system: str | None = Field(None, description="The column name for the `system (prompt)`", examples=["system"])
    field_instruction: str = Field(
        ..., description="The column name for the instruction - always needs to be provided", examples=["instruction"]
    )
    field_input: str | None = Field(None, description="The column name for the `input`", examples=["input"])
    field_output: str | None = Field(None, description="The column name for the `output`", examples=["output"])

    # NOTE: ATM can not be defined by the user, but should be able to in the future
    format: None = Field(None, description="The column name for the `format`", examples=["{instruction} {input}"])
    no_input_format: None = Field(
        None, description="If the field_input is not provided, what format should we use? ", examples=["{instruction}"]
    )
    system_format: None = Field(None, description="How to format the `system (prompt)`", examples=["{system}"])

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())


class DpoTaskDetails(TaskDetails):
    task_type: TaskType = TaskType.DPOTASK
    base_model_repository: str
    ds_repo: str

    field_prompt: str = Field(..., description="The column name for the prompt", examples=["prompt"])
    field_system: str | None = Field(None, description="The column name for the `system (prompt)`", examples=["system"])
    field_chosen: str = Field(..., description="The column name for the chosen response", examples=["chosen"])
    field_rejected: str = Field(..., description="The column name for the rejected response", examples=["rejected"])

    prompt_format: str | None = Field(None, description="The format of the prompt", examples=["{system} {prompt}"])
    chosen_format: str | None = Field(None, description="The format of the chosen response", examples=["{chosen} <|endoftext|>"])
    rejected_format: str | None = Field(
        None, description="The format of the rejected response", examples=["{rejected} <|endoftext|>"]
    )

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())


class GrpoTaskDetails(TaskDetails):
    task_type: TaskType = TaskType.GRPOTASK
    base_model_repository: str
    ds_repo: str

    field_prompt: str = Field(..., description="The column name for the prompt", examples=["prompt"])
    reward_functions: list[RewardFunction]

    # Turn off protected namespace for model
    model_config = ConfigDict(protected_namespaces=())


class ImageTaskDetails(TaskDetails):
    task_type: TaskType = TaskType.IMAGETASK
    image_text_pairs: list[ImageTextPair]
    base_model_repository: str = Field(..., description="The repository for the model")
    model_type: ImageModelType = ImageModelType.SDXL

    model_config = ConfigDict(protected_namespaces=())


class TaskListResponse(BaseModel):
    success: bool
    task_id: UUID
    status: TaskStatus


class LeaderboardRow(BaseModel):
    hotkey: str
    stats: AllNodeStats


class ImageModelInfo(BaseModel):
    model_id: str
    model_type: ImageModelType

    model_config = ConfigDict(protected_namespaces=())


class ImageModelsResponse(BaseModel):
    models: list[ImageModelInfo]


class GpuRequirementSummary(BaseModel):
    gpu_type: str
    count: int
    total_hours: float


class TournamentGpuRequirementsResponse(BaseModel):
    gpu_requirements: list[GpuRequirementSummary]
    total_tasks: int
    total_hours: float


# Type alias for task details types
AnyTypeTaskDetails = InstructTextTaskDetails | ImageTaskDetails | DpoTaskDetails | GrpoTaskDetails
