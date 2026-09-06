import os
import sys
import pytest

# Ensure UTF-8 output encoding for Windows compatibility
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.utils import (
    hash_embedding,
    hash_payload,
    to_bytes32,
    identify_platform,
    select_image_file,
    compute_cosine_similarity,
    compute_merkle_root,
    assess_image_quality,
    string_similarity,
    create_resilient_session,
    extract_opengraph_metadata,
)
from src.face_engine import FaceEngine, BiometricLRUCache
from src.search_engine import SearchEngine
from src.consensus_engine import IdentityConsensusEngine
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


def test_select_image_file():
    # Verify select_image_file non-interactive behavior
    res = select_image_file(prompt_if_cancelled=False, use_gui=False)
    assert res is None


def test_cosine_and_merkle_roots():
    # Identical vectors -> Cosine similarity 1.0
    v1 = [1.0, 2.0, 3.0, 4.0]
    assert pytest.approx(compute_cosine_similarity(v1, v1), 0.001) == 1.0

    # Opposite vectors -> Cosine similarity -1.0
    v2 = [-1.0, -2.0, -3.0, -4.0]
    assert pytest.approx(compute_cosine_similarity(v1, v2), 0.001) == -1.0

    # Orthogonal vectors -> 0.0
    v3 = [1.0, 0.0]
    v4 = [0.0, 1.0]
    assert pytest.approx(compute_cosine_similarity(v3, v4), 0.001) == 0.0

    # Merkle tree root derivation
    leaves = ["aa" * 32, "bb" * 32, "cc" * 32]
    root1 = compute_merkle_root(leaves)
    root2 = compute_merkle_root(leaves)
    assert root1 == root2
    assert len(root1) == 64

    # Fuzzy string similarity
    assert string_similarity("Alex Morgan", "alex morgan") == 1.0
    assert string_similarity("Alex Morgan", "Alex M.") > 0.60


def test_image_quality_assessment():
    img_path = "samples/test_face.jpg"
    assert os.path.exists(img_path)
    q = assess_image_quality(img_path)
    assert q["is_valid"] is True
    assert q["width"] > 0
    assert q["height"] > 0
    assert "blur_variance" in q


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
    assert res.quality is not None


def test_multi_site_concurrent_search():
    img_path = "samples/test_face.jpg"
    search = SearchEngine()
    multi_res = search.search_multi_site(
        image_path=img_path,
        target_platforms=["Twitter/X", "LinkedIn", "GitHub", "Instagram", "Web"],
        demo_mode=True,
    )
    assert multi_res.total_found >= 5
    assert "Twitter/X" in multi_res.matches_by_platform
    assert "LinkedIn" in multi_res.matches_by_platform
    assert "GitHub" in multi_res.matches_by_platform


def test_identity_consensus_and_disparity_reporting():
    face_engine = FaceEngine(detector="retinaface", model="ArcFace")
    search_engine = SearchEngine()
    consensus_engine = IdentityConsensusEngine()

    face_res = face_engine.analyze("samples/test_face.jpg")
    real_hash = face_res.face_hash
    real_emb = face_res.embedding

    # 1. Test Consensus Quorum Reached (All sites agree)
    search_agreed = search_engine.search_multi_site(
        "samples/test_face.jpg",
        demo_mode=True,
        simulate_disparity=False,
    )
    result_agreed = consensus_engine.evaluate_multi_site_consensus(
        input_face_hash=real_hash,
        input_embedding=real_emb,
        multi_site_result=search_agreed,
        face_engine=face_engine,
    )
    assert result_agreed.consensus_reached is True
    assert result_agreed.disparate_situation_detected is False
    assert result_agreed.verified_site_count >= 2
    assert len(result_agreed.merkle_root) == 64

    # 2. Test Disparate Entities Detection & Reporting (Mismatched profiles)
    search_disparate = search_engine.search_multi_site(
        "samples/test_face.jpg",
        demo_mode=True,
        simulate_disparity=True,
    )
    result_disparate = consensus_engine.evaluate_multi_site_consensus(
        input_face_hash=real_hash,
        input_embedding=real_emb,
        multi_site_result=search_disparate,
        face_engine=face_engine,
    )
    assert result_disparate.disparate_situation_detected is True
    assert result_disparate.disparate_summary is not None
    assert any(e.final_vote == "DIVERGENT" for e in result_disparate.site_evidences)


def test_blockchain_consensus_lifecycle_and_tamper_defense():
    engine = BlockchainEngine()
    engine.ensure_connected(auto_start=True)
    addr = engine.deploy_contract()
    assert addr.startswith("0x")

    face_hash = "12ed18e6c1f9e6c7c6c6e56b8b3fd10aa3a748fe917b8938b12e4c0e3baf82ec"
    merkle_root = "55aa" * 16
    entity_name = "Alex Morgan"
    platforms = ["Twitter/X", "LinkedIn", "GitHub"]
    urls = [
        "https://twitter.com/alex/1",
        "https://linkedin.com/in/alex",
        "https://github.com/alex",
    ]
    score = 0.985
    verified_count = 3
    metadata = {"consensus_protocol": "v2.0", "quorums": 3}

    res = engine.record_consensus_match(
        face_hash=face_hash,
        merkle_root=merkle_root,
        entity_name=entity_name,
        platforms=platforms,
        match_urls=urls,
        consensus_score=score,
        verified_site_count=verified_count,
        metadata=metadata,
    )

    assert res.consensus_id >= 0
    assert res.tx_hash.startswith("0x")

    # Positive on-chain verification
    v_ok = engine.verify_consensus_record(res.consensus_id, face_hash, merkle_root)
    assert v_ok.is_verified is True
    assert v_ok.face_hash_matches is True
    assert v_ok.merkle_root_matches is True
    assert v_ok.entity_name == entity_name
    assert pytest.approx(v_ok.consensus_score, 0.01) == score

    # Negative verification (tampered Merkle root)
    tampered_root = "0" * 64
    v_tampered = engine.verify_consensus_record(res.consensus_id, face_hash, tampered_root)
    assert v_tampered.is_verified is False
    assert v_tampered.merkle_root_matches is False


def test_resilient_session_and_opengraph_extraction():
    sess = create_resilient_session(pool_size=10, max_retries=2)
    assert sess is not None
    assert "User-Agent" in sess.headers

    # Metadata extraction on empty or invalid URL returns gracefully
    meta_empty = extract_opengraph_metadata("", session=sess)
    assert meta_empty["og_image"] is None

    meta_invalid = extract_opengraph_metadata("invalid-url", session=sess)
    assert meta_invalid["og_image"] is None


def test_bounded_lru_cache():
    cache = BiometricLRUCache(maxsize=3)
    cache.set("k1", {"val": 1})
    cache.set("k2", {"val": 2})
    cache.set("k3", {"val": 3})
    assert len(cache) == 3
    assert "k1" in cache

    # Access k1 to make it most recently used
    _ = cache.get("k1")

    # Add k4 -> k2 should be evicted (least recently used)
    cache.set("k4", {"val": 4})
    assert len(cache) == 3
    assert "k2" not in cache
    assert "k1" in cache
    assert "k4" in cache

    cache.clear()
    assert len(cache) == 0


def test_consensus_formula_weights():
    consensus_engine = IdentityConsensusEngine()
    face_engine = FaceEngine()

    face_res = face_engine.analyze("samples/test_face.jpg")
    search_res = SearchEngine().search_multi_site("samples/test_face.jpg", demo_mode=True)

    eval_res = consensus_engine.evaluate_multi_site_consensus(
        input_face_hash=face_res.face_hash,
        input_embedding=face_res.embedding,
        multi_site_result=search_res,
        face_engine=face_engine,
    )
    # Consensus score is bounded 0.0 to 1.0 and incorporates biometric, entity, and site factors
    assert 0.0 <= eval_res.consensus_score <= 1.0
    assert eval_res.consensus_score >= 0.70
    assert eval_res.verified_site_count >= 2


def test_multi_model_warmup():
    engine = FaceEngine(model="ArcFace")
    assert engine.warm_up() is True


def test_retry_mechanism_on_disparity():
    face_engine = FaceEngine()
    face_res = face_engine.analyze("samples/test_face.jpg")
    search_engine = SearchEngine()
    consensus_engine = IdentityConsensusEngine()

    # Call with simulated disparity
    attempts_called = 0

    def mock_search_func(retry_lvl: int):
        nonlocal attempts_called
        attempts_called += 1
        return search_engine.search_multi_site(
            image_path="samples/test_face.jpg",
            demo_mode=True,
            simulate_disparity=True,
            retry_level=retry_lvl,
        )

    res = consensus_engine.run_consensus_pipeline(
        search_engine_func=mock_search_func,
        face_engine=face_engine,
        input_face_hash=face_res.face_hash,
        input_embedding=face_res.embedding,
        max_retries=2,
    )

    # Should have retried 2 times more (total 3 attempts)
    assert attempts_called == 3
    assert res.attempts_taken == 3
    assert res.disparate_situation_detected is True
    assert "Disparate entities situation detected" in res.disparate_summary


def test_authentic_social_handle_and_profile_validation():
    search_engine = SearchEngine()

    # 1. Test profile URL validation
    assert search_engine._is_valid_profile_url("https://x.com/alluarjun", "Twitter/X") is True
    assert search_engine._is_valid_profile_url("https://twitter.com/alluarjun", "Twitter/X") is True
    assert search_engine._is_valid_profile_url("https://zeenews.india.com/people/article-123", "Twitter/X") is False
    assert search_engine._is_valid_profile_url("https://x.com/explore", "Twitter/X") is False

    assert search_engine._is_valid_profile_url("https://www.instagram.com/alluarjunonline/", "Instagram") is True
    assert search_engine._is_valid_profile_url("https://www.instagram.com/p/C982349823/", "Instagram") is False
    assert search_engine._is_valid_profile_url("https://timesofindia.indiatimes.com/news", "Instagram") is False

    assert search_engine._is_valid_profile_url("https://www.linkedin.com/in/arjun-allu-b4076715b", "LinkedIn") is True
    assert search_engine._is_valid_profile_url("https://www.linkedin.com/feed/", "LinkedIn") is False

    assert search_engine._is_valid_profile_url("https://github.com/torvalds", "GitHub") is True
    assert search_engine._is_valid_profile_url("https://github.com/torvalds/linux", "GitHub") is False
    assert search_engine._is_valid_profile_url("https://github.com/topics/ai", "GitHub") is False

    # 2. Test profile URL normalization
    assert search_engine._normalize_social_url("https://x.com/alluarjun/status/123456789", "Twitter/X") == "https://x.com/alluarjun"
    assert search_engine._normalize_social_url("https://twitter.com/alluarjun/with_replies", "Twitter/X") == "https://x.com/alluarjun"
    assert search_engine._normalize_social_url("https://www.instagram.com/alluarjunonline/reels/", "Instagram") == "https://www.instagram.com/alluarjunonline"


