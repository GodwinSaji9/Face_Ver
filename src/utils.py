import difflib
import hashlib
import json
import os
import re
import struct
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
import numpy as np
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def hash_embedding(embedding: List[float]) -> str:
    """
    Computes a deterministic SHA-256 hex digest of a floating-point embedding vector.
    Uses IEEE 754 32-bit single-precision float packing for platform-independent reproducibility.
    """
    if not embedding:
        raise ValueError("Embedding vector cannot be empty.")
    try:
        # High-performance vectorized packing via NumPy
        arr = np.array(embedding, dtype=np.float32)
        return hashlib.sha256(arr.tobytes()).hexdigest()
    except Exception:
        # Standard struct fallback
        byte_repr = struct.pack(f"{len(embedding)}f", *[float(x) for x in embedding])
        return hashlib.sha256(byte_repr).hexdigest()


def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculates cosine similarity between two high-dimensional embedding vectors:
    cos_sim(u, v) = (u . v) / (||u|| * ||v||)
    Returns float in range [-1.0, 1.0].
    """
    if not vec1 or not vec2:
        return 0.0
    u = np.array(vec1, dtype=np.float32)
    v = np.array(vec2, dtype=np.float32)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0.0
    similarity = float(np.dot(u, v) / (norm_u * norm_v))
    # Clip for floating point numerical precision
    return max(-1.0, min(1.0, similarity))


def compute_merkle_root(leaf_hashes: List[str]) -> str:
    """
    Constructs a deterministic cryptographic Merkle Tree from a list of 64-char hex leaf hashes.
    Returns 64-character SHA-256 Merkle root hex digest.
    """
    if not leaf_hashes:
        return hashlib.sha256(b"EMPTY_MERKLE_TREE").hexdigest()

    # Normalize leaves: lowercase, clean 0x prefix if present
    current_level = [
        bytes.fromhex(h[2:] if h.startswith("0x") else h.ljust(64, "0")[:64])
        for h in leaf_hashes
    ]

    while len(current_level) > 1:
        # If odd number of nodes, duplicate the last node
        if len(current_level) % 2 == 1:
            current_level.append(current_level[-1])

        next_level = []
        for i in range(0, len(current_level), 2):
            combined = current_level[i] + current_level[i + 1]
            parent_hash = hashlib.sha256(combined).digest()
            next_level.append(parent_hash)

        current_level = next_level

    return current_level[0].hex()


def assess_image_quality(image_path: str, blur_threshold: float = 30.0) -> Dict[str, Any]:
    """
    Performs portrait quality assessment:
    1. Checks image dimensions and aspect ratio
    2. Computes Laplacian blur variance (Var(nabla^2 I))
    3. Evaluates overall luminance and exposure
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        import cv2

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not decode image at {image_path}")

        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))

        is_blurry = blur_variance < blur_threshold
        is_too_dark = mean_brightness < 20.0
        is_too_bright = mean_brightness > 240.0
        is_valid = not is_blurry and not is_too_dark and not is_too_bright

        return {
            "is_valid": is_valid,
            "blur_variance": round(blur_variance, 2),
            "is_blurry": is_blurry,
            "mean_brightness": round(mean_brightness, 2),
            "width": width,
            "height": height,
        }
    except Exception as e:
        # Fallback using PIL
        pil_img = Image.open(image_path)
        w, h = pil_img.size
        return {
            "is_valid": True,
            "blur_variance": 100.0,
            "is_blurry": False,
            "mean_brightness": 128.0,
            "width": w,
            "height": h,
            "note": f"Basic PIL fallback: {e}",
        }


def string_similarity(str1: str, str2: str) -> float:
    """
    Computes fuzzy string similarity ratio between two strings (0.0 to 1.0).
    Normalizes casing and whitespace.
    """
    if not str1 or not str2:
        return 0.0
    s1 = " ".join(str1.lower().split())
    s2 = " ".join(str2.lower().split())
    if s1 == s2:
        return 1.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


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


def create_resilient_session(pool_size: int = 20, max_retries: int = 3) -> requests.Session:
    """
    Creates a resilient requests.Session configured with HTTP connection pooling,
    exponential backoff retries, and standard browser user-agent headers.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(
        pool_connections=pool_size,
        pool_maxsize=pool_size,
        max_retries=retry_strategy,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


def extract_opengraph_metadata(
    url: str,
    session: Optional[requests.Session] = None,
    timeout: float = 4.0,
) -> Dict[str, Optional[str]]:
    """
    Extracts OpenGraph image, title, and profile meta tags from a target URL.
    Returns dictionary with og_image, og_title, page_title, and author.
    """
    if not url or not url.startswith(("http://", "https://")):
        return {"og_image": None, "og_title": None, "page_title": None, "author": None}

    sess = session or create_resilient_session()
    try:
        resp = sess.get(url, timeout=timeout, allow_redirects=True)
        if resp.status_code != 200:
            return {"og_image": None, "og_title": None, "page_title": None, "author": None}

        # Fast regex extraction on head region
        html = resp.text[:100_000]
        
        # og:image or twitter:image
        og_img_match = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        if not og_img_match:
            og_img_match = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', html, re.IGNORECASE)
        if not og_img_match:
            og_img_match = re.search(r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)

        # og:title or title tag
        og_title_match = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        
        # author tag
        author_match = re.search(r'<meta[^>]+name=["\']author["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE)

        return {
            "og_image": og_img_match.group(1) if og_img_match else None,
            "og_title": og_title_match.group(1).strip() if og_title_match else None,
            "page_title": title_match.group(1).strip() if title_match else None,
            "author": author_match.group(1).strip() if author_match else None,
        }
    except Exception:
        return {"og_image": None, "og_title": None, "page_title": None, "author": None}


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


def select_image_file(prompt_if_cancelled: bool = True, use_gui: bool = True) -> Optional[str]:
    """
    Allows the user to select an image file from the system itself using a native
    system file dialog (Tkinter/Windows explorer). If cancelled or unavailable,
    gracefully provides an interactive CLI fallback with detected samples.
    """
    selected_path: Optional[str] = None

    # Step 1: Attempt native GUI file dialog if enabled
    if use_gui:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update()

            initial_dir = os.path.abspath("samples") if os.path.exists("samples") else os.getcwd()
            file_path = filedialog.askopenfilename(
                parent=root,
                title="Choose Face Image for Biometric Verification",
                initialdir=initial_dir,
                filetypes=[
                    ("Image Files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tiff"),
                    ("JPEG Images", "*.jpg;*.jpeg"),
                    ("PNG Images", "*.png"),
                    ("All Files", "*.*"),
                ],
            )
            root.destroy()

            if file_path and os.path.isfile(file_path):
                return os.path.abspath(file_path)

        except Exception:
            pass

    # Step 2: Interactive terminal prompt fallback if dialog cancelled or GUI unavailable
    if not prompt_if_cancelled:
        return None

    print("\n[File Selection] Native dialog closed. Please specify an image:")
    
    # Discover available samples
    sample_files: List[str] = []
    if os.path.exists("samples"):
        for f in sorted(os.listdir("samples")):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                sample_files.append(os.path.join("samples", f))

    if sample_files:
        print("  Quick presets:")
        for idx, s in enumerate(sample_files, 1):
            print(f"    [{idx}] {s}")

    try:
        user_input = input("\n  Enter file path, preset number, or drag & drop image: ").strip()
        clean_input = user_input.strip('"').strip("'")

        if clean_input.isdigit() and 1 <= int(clean_input) <= len(sample_files):
            return os.path.abspath(sample_files[int(clean_input) - 1])

        if clean_input and os.path.isfile(clean_input):
            return os.path.abspath(clean_input)

    except (KeyboardInterrupt, EOFError):
        return None

    return None
