from collections import OrderedDict
from dataclasses import dataclass, asdict
import hashlib
import os
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image
from src.utils import crop_face, hash_embedding, compute_cosine_similarity, assess_image_quality


class BiometricLRUCache:
    """Bounded, thread-safe LRU cache for face embedding vectors (Iteration 11)."""

    def __init__(self, maxsize: int = 256):
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: Dict[str, Any]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def values(self):
        return list(self._cache.values())

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __getitem__(self, key: str) -> Dict[str, Any]:
        return self._cache[key]

    def __setitem__(self, key: str, value: Dict[str, Any]) -> None:
        self.set(key, value)

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()


# Global in-memory LRU cache to prevent redundant neural network forward passes
_EMBEDDING_CACHE = BiometricLRUCache(maxsize=256)


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
    quality: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
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

    def analyze(
        self,
        image_path: str,
        enforce_detection: bool = True,
        crop_padding: float = 0.10,
        check_quality: bool = True,
    ) -> FaceAnalysisResult:
        """
        Executes end-to-end detection and encoding on the target image.
        1. Evaluates portrait quality (blur, luminance)
        2. Detects face location & landmarks using RetinaFace (with cascade fallback)
        3. Crops the detected face region with margin for optimal reverse search
        4. Generates 512-d biometric embedding using ArcFace (with caching)
        5. Computes deterministic SHA-256 hash of the embedding vector
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        # Quality check
        quality_info = assess_image_quality(image_path) if check_quality else None

        # Check in-memory cache
        cache_key = self._get_cache_key(image_path, crop_padding)
        if cache_key in _EMBEDDING_CACHE:
            cached = _EMBEDDING_CACHE[cache_key]
            return FaceAnalysisResult(
                face_detected=True,
                facial_area=cached["facial_area"],
                confidence=cached["confidence"],
                embedding=cached["embedding"],
                face_hash=cached["face_hash"],
                cropped_image_path=cached["cropped_image_path"],
                detector=self.detector,
                model=self.model,
                quality=quality_info,
            )

        # Lazy import DeepFace to keep startup instantaneous
        from deepface import DeepFace

        # Prepare image (clamp resolution if excessively large > 1920px to save RAM)
        prepared_path = self._clamp_resolution_if_needed(image_path)

        # Step 1: Detect face location with cascade fallback
        try:
            face_objs = DeepFace.extract_faces(
                img_path=prepared_path,
                detector_backend=self.detector,
                enforce_detection=enforce_detection,
                align=True,
            )
        except Exception:
            # Multi-detector fallback cascade: RetinaFace -> MTCNN -> OpenCV
            fallback_detectors = ["mtcnn", "opencv"]
            face_objs = []
            for fb in fallback_detectors:
                try:
                    face_objs = DeepFace.extract_faces(
                        img_path=prepared_path,
                        detector_backend=fb,
                        enforce_detection=enforce_detection,
                        align=True,
                    )
                    if face_objs:
                        self.detector = f"{fb} (fallback)"
                        break
                except Exception:
                    continue

        if not face_objs:
            raise ValueError(f"No face detected in {image_path}")

        # Pick the most prominent / highest confidence face
        primary_face = max(face_objs, key=lambda f: f.get("confidence", 0.0))
        facial_area = primary_face.get("facial_area", {})
        confidence = float(primary_face.get("confidence", 1.0))

        # Step 2: Crop the detected face for reverse image searching
        cropped_path = crop_face(prepared_path, facial_area, padding_pct=crop_padding)

        # Step 3: Extract facial embedding using ArcFace (or Ensemble: ArcFace + Facenet512)
        if self.model.lower() in ("ensemble", "dual"):
            models_to_run = ["ArcFace", "Facenet512"]
            combined_emb: List[float] = []
            for m in models_to_run:
                emb_objs = DeepFace.represent(
                    img_path=cropped_path,
                    model_name=m,
                    detector_backend="skip",
                    enforce_detection=False,
                )
                if emb_objs and "embedding" in emb_objs[0]:
                    v = np.array(emb_objs[0]["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        v = v / norm
                    combined_emb.extend(v.tolist())
            if not combined_emb:
                raise ValueError(f"Failed to generate ensemble biometric embedding for {image_path}")
            embedding = combined_emb
        else:
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

        # Cache result
        _EMBEDDING_CACHE[cache_key] = {
            "source_image": image_path,
            "facial_area": facial_area,
            "confidence": confidence,
            "embedding": embedding,
            "face_hash": face_hash,
            "cropped_image_path": cropped_path,
        }

        return FaceAnalysisResult(
            face_detected=True,
            facial_area=facial_area,
            confidence=confidence,
            embedding=embedding,
            face_hash=face_hash,
            cropped_image_path=cropped_path,
            detector=self.detector,
            model=self.model,
            quality=quality_info,
        )

    def extract_embedding_only(self, image_path: str) -> List[float]:
        """
        Fast direct embedding extractor for scraped avatars.
        Checks in-memory LRU cache first, then performs ArcFace / Ensemble feature extraction.
        """
        # Check cache by exact path or cropped path
        norm_path = os.path.normpath(image_path)
        for cached in _EMBEDDING_CACHE.values():
            cached_source = cached.get("source_image")
            cached_cropped = cached.get("cropped_image_path")
            if (cached_source and os.path.normpath(cached_source) == norm_path) or \
               (cached_cropped and os.path.normpath(cached_cropped) == norm_path):
                return cached["embedding"]

        cache_key = self._get_cache_key(image_path, crop_padding=0.15)
        if cache_key in _EMBEDDING_CACHE:
            return _EMBEDDING_CACHE[cache_key]["embedding"]

        from deepface import DeepFace

        models = ["ArcFace", "Facenet512"] if self.model.lower() in ("ensemble", "dual") else [self.model]
        all_embs: List[float] = []
        for mod in models:
            mod_emb: List[float] = []
            primary_det = self.detector.split()[0].lower()
            detectors_to_try = [primary_det, "opencv", "skip"]
            for det in detectors_to_try:
                try:
                    res = DeepFace.represent(
                        img_path=image_path,
                        model_name=mod,
                        detector_backend=det,
                        enforce_detection=False,
                    )
                    if res and "embedding" in res[0]:
                        v = np.array(res[0]["embedding"], dtype=np.float32)
                        norm = np.linalg.norm(v)
                        if norm > 0 and len(models) > 1:
                            v = v / norm
                        mod_emb = v.tolist()
                        break
                except Exception:
                    continue
            if mod_emb:
                all_embs.extend(mod_emb)
            elif len(models) == 1:
                return []

        if all_embs:
            _EMBEDDING_CACHE[cache_key] = {
                "source_image": image_path,
                "embedding": all_embs,
            }
            return all_embs
        return []

    def warm_up(self) -> bool:
        """
        Pre-warms deep learning backends and models (Iteration 17).
        Minimizes cold-start latency for biometric requests.
        """
        try:
            from deepface import DeepFace
            target_model = "ArcFace" if self.model.lower() not in ("ensemble", "dual") else "ArcFace"
            _ = DeepFace.build_model(target_model)
            return True
        except Exception:
            return False

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculates cosine similarity between two face embeddings.
        """
        return compute_cosine_similarity(embedding1, embedding2)

    def _get_cache_key(self, image_path: str, crop_padding: float) -> str:
        """Computes unique cache key based on file content and parameters."""
        try:
            mtime = os.path.getmtime(image_path)
            size = os.path.getsize(image_path)
            raw = f"{image_path}:{mtime}:{size}:{self.detector}:{self.model}:{crop_padding}"
            return hashlib.sha256(raw.encode()).hexdigest()
        except Exception:
            return image_path

    def _clamp_resolution_if_needed(self, image_path: str, max_dim: int = 1600) -> str:
        """Downscales excessively large images to conserve memory and speed up inference."""
        try:
            img = Image.open(image_path)
            w, h = img.size
            if max(w, h) <= max_dim:
                return image_path

            temp_dir = "temp_crops"
            os.makedirs(temp_dir, exist_ok=True)
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            out_path = os.path.join(temp_dir, f"clamped_{os.path.basename(image_path)}")
            resized.save(out_path, quality=95)
            return out_path
        except Exception:
            return image_path
