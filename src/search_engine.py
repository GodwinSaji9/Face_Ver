from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional
import serpapi
from src.utils import hash_payload, identify_platform


@dataclass
class SearchMatch:
    title: str
    link: str
    source: str
    platform: str
    thumbnail: Optional[str]
    is_social: bool
    match_hash: str
    timestamp_utc: str
    raw_snippet: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class SearchEngine:
    """
    Performs genuine reverse image search using Google Lens via SerpAPI.
    Discovers matching web content and isolates real social media posts.
    """

    KNOWN_SOCIAL_DOMAINS = {
        "instagram.com",
        "twitter.com",
        "x.com",
        "linkedin.com",
        "facebook.com",
        "reddit.com",
        "tiktok.com",
        "youtube.com",
        "pinterest.com",
        "github.com",
        "threads.net",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPAPI_KEY")

    def search(
        self,
        image_path: str,
        max_results: int = 20,
        prefer_social: bool = True,
        demo_mode: bool = False,
    ) -> SearchMatch:
        """
        Executes reverse image search on the provided face image crop.
        Returns the top matching social media post (or top visual match).
        """
        # If demo mode is explicitly requested or no API key is available
        if demo_mode or not self.api_key:
            if not self.api_key and not demo_mode:
                raise ValueError(
                    "SerpAPI key not found. Please set SERPAPI_KEY in your .env file, "
                    "pass --serpapi-key, or use --demo-search to simulate the search step."
                )
            return self._generate_simulated_match(image_path)

        client = serpapi.Client(api_key=self.api_key)

        # Handle local image upload vs remote URL
        is_url = image_path.startswith("http://") or image_path.startswith("https://")
        search_params: Dict[str, Any] = {"engine": "google_lens", "hl": "en"}

        if is_url:
            search_params["url"] = image_path
        else:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")
            
            # Upload local cropped face image to SerpAPI image store
            upload = client.upload_image(image_path)
            if "image_id" not in upload:
                raise RuntimeError(f"Failed to upload image to SerpAPI: {upload}")
            search_params["image_id"] = upload["image_id"]

        # Perform live Google Lens reverse search
        results = client.search(search_params)
        visual_matches = results.get("visual_matches", [])

        if not visual_matches:
            raise RuntimeError(
                f"Google Lens returned 0 visual matches for this face. "
                f"SerpAPI search metadata: {results.get('search_metadata', {})}"
            )

        # Parse and prioritize social media matches
        candidates: List[SearchMatch] = []
        for match in visual_matches[:max_results]:
            link = match.get("link", "")
            title = match.get("title", "Discovered Match")
            source = match.get("source", "")
            thumbnail = match.get("thumbnail")
            platform = identify_platform(link or source)
            
            is_social = any(domain in link.lower() for domain in self.KNOWN_SOCIAL_DOMAINS)

            now_iso = datetime.now(timezone.utc).isoformat()
            payload_for_hash = {
                "title": title,
                "link": link,
                "source": source,
                "platform": platform,
                "is_social": is_social,
            }
            m_hash = hash_payload(payload_for_hash)

            candidates.append(
                SearchMatch(
                    title=title,
                    link=link,
                    source=source,
                    platform=platform,
                    thumbnail=thumbnail,
                    is_social=is_social,
                    match_hash=m_hash,
                    timestamp_utc=now_iso,
                    raw_snippet=match.get("snippet"),
                )
            )

        # If prefer_social is True, look for first social post
        if prefer_social:
            for c in candidates:
                if c.is_social:
                    return c

        # Fallback to the highest-ranked visual match
        return candidates[0]

    def _generate_simulated_match(self, image_path: str) -> SearchMatch:
        """
        Provides a realistic simulated match when running in demo/offline mode.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        sample_url = "https://twitter.com/identity_ledger/status/1789012345678901234"
        sample_title = "Verified Profile Portrait Discovery #Web3FaceID"
        sample_source = "Twitter/X"
        sample_platform = "Twitter/X"

        payload = {
            "title": sample_title,
            "link": sample_url,
            "source": sample_source,
            "platform": sample_platform,
            "is_social": True,
            "demo_mode": True,
        }
        m_hash = hash_payload(payload)

        return SearchMatch(
            title=sample_title,
            link=sample_url,
            source=sample_source,
            platform=sample_platform,
            thumbnail="https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150",
            is_social=True,
            match_hash=m_hash,
            timestamp_utc=now_iso,
            raw_snippet="[DEMO SEARCH] Genuine SerpAPI reverse-image search structure verified.",
        )
