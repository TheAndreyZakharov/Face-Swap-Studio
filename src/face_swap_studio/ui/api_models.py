from __future__ import annotations

from pydantic import BaseModel, Field


class SessionCreateResponse(BaseModel):
    session_id: str


class UploadedImageResponse(BaseModel):
    id: str
    name: str
    url: str
    selected: bool = False


class FaceResponse(BaseModel):
    id: str
    index: int
    image_url: str
    label: str


class ModelResponse(BaseModel):
    id: str
    name: str
    description: str
    available: bool


class DetectionRequest(BaseModel):
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.95,
    )


class TargetSelectionRequest(BaseModel):
    target_image_id: str


class FaceMappingItem(BaseModel):
    target_face_index: int = Field(
        ge=0,
    )
    source_face_index: int | None = Field(
        default=None,
        ge=0,
    )


class FaceMappingRequest(BaseModel):
    mappings: list[FaceMappingItem]


class GenerationRequest(BaseModel):
    model_id: str

    enhance_faces: bool = False
    enhancement_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
    )

    upscale_image: bool = False
    upscale_factor: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
    )
    tile_size: int = Field(
        default=256,
        ge=64,
        le=1024,
    )


class SessionResponse(BaseModel):
    session_id: str

    source_images: list[UploadedImageResponse]
    target_images: list[UploadedImageResponse]

    source_faces: list[FaceResponse]
    target_faces: list[FaceResponse]

    mappings: dict[int, int | None]

    active_target_image_id: str | None
    analysis_completed: bool

    result_url: str | None
    download_url: str | None