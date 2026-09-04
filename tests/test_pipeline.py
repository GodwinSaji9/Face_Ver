import os
import sys
import pytest

# Ensure UTF-8 output encoding for Windows compatibility
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.utils import hash_embedding, hash_payload, to_bytes32, identify_platform
from src.face_engine import FaceEngine
from src.search_engine import SearchEngine
from src.blockchain_engine import BlockchainEngine


def test_hashing_utilities():
    # Determinism test for float embeddings
    vec = [0.123456, -0.654321, 0.987654]
    h1 = hash_embedding(vec)
    h2 = hash_embedding(vec)
    assert h1 == h2
    assert len(h1) == 64

    # Bytes32 conversion
    b32 = to_bytes32(h1)
    assert isinstance(b32, bytes)
    assert len(b32) == 32
    assert b32.hex() == h1


def test_platform_identification():
    assert identify_platform("https://twitter.com/user/status/123") == "Twitter/X"
    assert identify_platform("https://x.com/user/status/123") == "Twitter/X"
    assert identify_platform("https://www.instagram.com/p/B_123/") == "Instagram"
    assert identify_platform("https://linkedin.com/in/john-doe") == "LinkedIn"
    assert identify_platform("https://reddit.com/r/technology/comments/xyz") == "Reddit"


def test_face_engine():
    img_path = "samples/test_face.jpg"
    assert os.path.exists(img_path)
    engine = FaceEngine(detector="retinaface", model="ArcFace")
    res = engine.analyze(img_path)
    assert res.face_detected is True
    assert len(res.embedding) == 512
    assert len(res.face_hash) == 64
    assert os.path.exists(res.cropped_image_path)


def test_search_engine_demo():
    img_path = "samples/test_face.jpg"
    search = SearchEngine()
    match = search.search(img_path, demo_mode=True)
    assert match.is_social is True
    assert len(match.match_hash) == 64
    assert match.platform == "Twitter/X"


def test_blockchain_lifecycle_and_tamper_evidence():
    engine = BlockchainEngine()
    engine.ensure_connected(auto_start=True)
    addr = engine.deploy_contract()
    assert addr.startswith("0x")
    assert len(addr) == 42

    face_hash = "12ed18e6c1f9e6c7c6c6e56b8b3fd10aa3a748fe917b8938b12e4c0e3baf82ec"
    match_hash = "24230013558d497ded74554f7e1f897def1a62f15dd793878a66243f32f61244"
    match_url = "https://twitter.com/identity_ledger/status/1789012345678901234"
    platform = "Twitter/X"
    metadata = {"model": "ArcFace", "detector": "retinaface"}

    rec = engine.record_match(face_hash, match_hash, match_url, platform, metadata)
    assert rec.record_id >= 0
    assert rec.tx_hash.startswith("0x")

    # Positive verification
    v_ok = engine.verify_record(rec.record_id, face_hash, match_hash)
    assert v_ok.is_verified is True
    assert v_ok.face_hash_matches is True
    assert v_ok.match_hash_matches is True
    assert v_ok.on_chain_url == match_url

    # Negative verification (tampered data)
    tampered_face = "0" * 64
    v_tampered = engine.verify_record(rec.record_id, tampered_face, match_hash)
    assert v_tampered.is_verified is False
    assert v_tampered.face_hash_matches is False
