from dataclasses import dataclass, asdict
import os
from typing import Dict, List, Optional
from src.utils import crop_face, hash_embedding


@dataclass
class FaceAnalysisResult:
    face_detected: bool
    facial_area: Dict[str, int]
    confidence: float
    embedding: List[float]
    face_hash: str
    cropped_image_path: str
    detector: str
    model: str

    def to_dict(self) -> dict:
        d = asdict(self)
        # Avoid flooding JSON dumps with all 512 embedding floats unless requested
        d["embedding_dim"] = len(self.embedding)
        d["embedding_sample"] = self.embedding[:5]
        del d["embedding"]
        return d


class FaceEngine:
    """
    Handles facial detection and high-dimensional biometric representation.
    Default configuration:
      - Detector: RetinaFace (state-of-the-art multi-task face detection & landmark alignment)
      - Representation Model: ArcFace (512-dimensional feature embedding)
    """

    def __init__(
        self,
        detector: str = "retinaface",
        model: str = "ArcFace",
    ):
        self.detector = detector
        self.model = model

    def analyze(self, image_path: str, enforce_detection: bool = True) -> FaceAnalysisResult:
        """
        Executes end-to-end detection and encoding on the target image.
        1. Detects face location & landmarks using RetinaFace (or fallback)
        2. Crops the detected face region with margin for optimal reverse search
        3. Generates 512-d biometric embedding using ArcFace
        4. Computes deterministic SHA-256 hash of the embedding vector
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Lazy import DeepFace to keep CLI startup instant
        from deepface import DeepFace

        # Step 1: Detect face location
        try:
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector,
                enforce_detection=enforce_detection,
                align=True,
            )
        except Exception as e:
            # If specified detector encounters an issue, attempt graceful fallback to opencv
            if self.detector != "opencv":
                face_objs = DeepFace.extract_faces(
                    img_path=image_path,
                    detector_backend="opencv",
                    enforce_detection=enforce_detection,
                    align=True,
                )
                self.detector = "opencv (fallback)"
            else:
                raise RuntimeError(f"Face detection failed: {e}")

        if not face_objs:
            raise ValueError(f"No face detected in {image_path}")

        # Pick the most prominent / highest confidence face
        primary_face = max(face_objs, key=lambda f: f.get("confidence", 0.0))
        facial_area = primary_face.get("facial_area", {})
        confidence = float(primary_face.get("confidence", 1.0))

        # Step 2: Crop the detected face for reverse image searching
        cropped_path = crop_face(image_path, facial_area)

        # Step 3: Extract facial embedding using ArcFace
        embedding_objs = DeepFace.represent(
            img_path=cropped_path,
            model_name=self.model,
            detector_backend="skip",  # Already cropped and aligned
            enforce_detection=False,
        )

        if not embedding_objs or "embedding" not in embedding_objs[0]:
            raise ValueError(f"Failed to generate biometric embedding for {image_path}")

        embedding = embedding_objs[0]["embedding"]
        face_hash = hash_embedding(embedding)

        return FaceAnalysisResult(
            face_detected=True,
            facial_area=facial_area,
            confidence=confidence,
            embedding=embedding,
            face_hash=face_hash,
            cropped_image_path=cropped_path,
            detector=self.detector,
            model=self.model,
        )
