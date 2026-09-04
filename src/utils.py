import hashlib
import json
import os
import struct
from typing import Any, Dict, List
from urllib.parse import urlparse
from PIL import Image


def hash_embedding(embedding: List[float]) -> str:
    """
    Computes a deterministic SHA-256 hex digest of a floating-point embedding vector.
    Packs floats into IEEE 754 binary representation for platform independence.
    """
    if not embedding:
        raise ValueError("Embedding vector cannot be empty.")
    
    # Pack as 32-bit single-precision floats (standard for deep learning embeddings)
    byte_repr = struct.pack(f"{len(embedding)}f", *[float(x) for x in embedding])
    return hashlib.sha256(byte_repr).hexdigest()


def hash_payload(payload: Any) -> str:
    """
    Computes a deterministic SHA-256 hex digest of any string or JSON-serializable dictionary.
    """
    if isinstance(payload, (dict, list)):
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        serialized = str(payload)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def to_bytes32(hex_str: str) -> bytes:
    """
    Converts a hex string (with or without '0x') into exactly 32 bytes for Solidity bytes32.
    """
    clean_hex = hex_str[2:] if hex_str.startswith("0x") else hex_str
    clean_hex = clean_hex.ljust(64, "0")[:64]
    return bytes.fromhex(clean_hex)


def format_bytes32_hex(b: bytes) -> str:
    """Formats bytes to 0x-prefixed 64-char hex string."""
    return "0x" + b.hex()


def identify_platform(url: str) -> str:
    """
    Identifies the social media platform or web source from a given URL.
    """
    if not url:
        return "Unknown"
    
    parsed = urlparse(url.lower())
    domain = parsed.netloc or parsed.path
    
    platform_map = {
        "instagram.com": "Instagram",
        "twitter.com": "Twitter/X",
        "x.com": "Twitter/X",
        "linkedin.com": "LinkedIn",
        "facebook.com": "Facebook",
        "reddit.com": "Reddit",
        "tiktok.com": "TikTok",
        "youtube.com": "YouTube",
        "pinterest.com": "Pinterest",
        "github.com": "GitHub",
        "threads.net": "Threads",
        "medium.com": "Medium",
        "quora.com": "Quora",
        "wikipedia.org": "Wikipedia",
    }
    
    for key, name in platform_map.items():
        if key in domain:
            return name
            
    return domain.replace("www.", "") if domain else "Web"


def crop_face(
    image_path: str,
    facial_area: Dict[str, int],
    output_dir: str = "temp_crops",
    padding_pct: float = 0.1,
) -> str:
    """
    Crops the detected facial region with an optional margin and saves it to disk.
    Returns the path to the cropped image file.
    """
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    x = facial_area.get("x", 0)
    y = facial_area.get("y", 0)
    w = facial_area.get("w", width)
    h = facial_area.get("h", height)

    pad_x = int(w * padding_pct)
    pad_y = int(h * padding_pct)

    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(width, x + w + pad_x)
    bottom = min(height, y + h + pad_y)

    cropped = img.crop((left, top, right, bottom))
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(output_dir, f"{base_name}_face_crop.jpg")
    cropped.save(out_path, format="JPEG", quality=95)
    return os.path.abspath(out_path)
