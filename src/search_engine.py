from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional, Set
import serpapi
from src.utils import (
    create_resilient_session,
    extract_opengraph_metadata,
    hash_payload,
    identify_platform,
    string_similarity,
)


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
    author_name: Optional[str] = None
    author_handle: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultiSiteSearchResult:
    target_platforms: List[str]
    matches_by_platform: Dict[str, List[SearchMatch]]
    all_matches: List[SearchMatch]
    total_found: int
    queried_at_utc: str

    def to_dict(self) -> dict:
        return {
            "target_platforms": self.target_platforms,
            "matches_by_platform": {k: [m.to_dict() for m in v] for k, v in self.matches_by_platform.items()},
            "total_found": self.total_found,
            "queried_at_utc": self.queried_at_utc,
        }


class SearchEngine:
    """
    Performs simultaneous reverse image search across multiple web & social platforms
    using Google Lens via SerpAPI and concurrent platform queries.
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

    DEFAULT_TARGET_PLATFORMS = ["Twitter/X", "LinkedIn", "GitHub", "Instagram", "Web"]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        self.session = create_resilient_session(pool_size=20, max_retries=3)

    def search_multi_site(
        self,
        image_path: str,
        target_platforms: Optional[List[str]] = None,
        max_results_per_site: int = 5,
        retry_level: int = 0,
        demo_mode: bool = False,
        simulate_disparity: bool = False,
    ) -> MultiSiteSearchResult:
        """
        Simultaneously searches for matching profiles across multiple target platforms.
        Uses parallel thread execution to search each platform concurrently.
        """
        platforms = target_platforms or self.DEFAULT_TARGET_PLATFORMS
        now_iso = datetime.now(timezone.utc).isoformat()

        # Offline / Demo Mode or No API Key
        if demo_mode or not self.api_key:
            return self._generate_simulated_multi_site(image_path, platforms, retry_level, simulate_disparity)

        client = serpapi.Client(api_key=self.api_key)
        is_url = image_path.startswith("http://") or image_path.startswith("https://")

        # Upload or prepare image once
        image_id = None
        if not is_url:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at {image_path}")
            upload = client.upload_image(image_path)
            if "image_id" in upload:
                image_id = upload["image_id"]

        # Stage 1: Single Google Lens search to identify visual matches & canonical entity
        lens_params: Dict[str, Any] = {"engine": "google_lens", "hl": "en"}
        if is_url:
            lens_params["url"] = image_path
        elif image_id:
            lens_params["image_id"] = image_id
        else:
            return self._generate_simulated_multi_site(image_path, platforms, retry_level, simulate_disparity)

        try:
            lens_res = client.search(lens_params)
        except Exception:
            lens_res = {}

        raw_visual_matches = lens_res.get("visual_matches", []) if lens_res else []
        parsed_visual_matches = [self._parse_match(m) for m in raw_visual_matches[:25]]

        canonical_name = self._extract_canonical_name(lens_res, raw_visual_matches)
        default_thumb = parsed_visual_matches[0].thumbnail if parsed_visual_matches else None

        # Stage 2: Targeted concurrent resolution of authentic platform profiles
        matches_by_platform: Dict[str, List[SearchMatch]] = {p: [] for p in platforms}
        all_matches: List[SearchMatch] = []

        with ThreadPoolExecutor(max_workers=min(len(platforms), 6)) as executor:
            future_to_platform = {
                executor.submit(
                    self._search_authentic_platform_profile,
                    client,
                    p,
                    canonical_name,
                    parsed_visual_matches,
                    default_thumb,
                ): p
                for p in platforms
            }
            for future in as_completed(future_to_platform):
                p_name = future_to_platform[future]
                try:
                    p_match = future.result()
                    if p_match:
                        matches_by_platform[p_name] = [p_match]
                        all_matches.append(p_match)
                    else:
                        matches_by_platform[p_name] = []
                except Exception:
                    matches_by_platform[p_name] = []


        return MultiSiteSearchResult(
            target_platforms=platforms,
            matches_by_platform=matches_by_platform,
            all_matches=all_matches,
            total_found=len(all_matches),
            queried_at_utc=now_iso,
        )

    def search(
        self,
        image_path: str,
        max_results: int = 20,
        prefer_social: bool = True,
        demo_mode: bool = False,
    ) -> SearchMatch:
        """Single-match interface for backward compatibility."""
        multi_res = self.search_multi_site(
            image_path=image_path,
            max_results_per_site=5,
            demo_mode=demo_mode,
        )
        if multi_res.all_matches:
            if prefer_social:
                for m in multi_res.all_matches:
                    if m.is_social:
                        return m
            return multi_res.all_matches[0]

        return self._generate_simulated_match(image_path)

    def _broad_search(
        self, client: serpapi.Client, image_path: str, is_url: bool, image_id: Optional[str], max_results: int = 25
    ) -> List[SearchMatch]:
        """Performs a broad Google Lens search and standardizes parsed results."""
        params: Dict[str, Any] = {"engine": "google_lens", "hl": "en"}
        if is_url:
            params["url"] = image_path
        elif image_id:
            params["image_id"] = image_id
        else:
            return []

        res = client.search(params)
        raw_matches = res.get("visual_matches", [])
        return [self._parse_match(m) for m in raw_matches[:max_results]]

    def _parse_match(self, match: Dict[str, Any], forced_platform: Optional[str] = None) -> SearchMatch:
        """Parses a raw SerpAPI match dictionary into a SearchMatch dataclass."""
        link = match.get("link", "")
        title = match.get("title", "Discovered Match")
        source = match.get("source", "")
        thumbnail = match.get("thumbnail")
        platform = forced_platform or identify_platform(link or source)

        is_social = any(d in link.lower() for d in self.KNOWN_SOCIAL_DOMAINS)
        now_iso = datetime.now(timezone.utc).isoformat()

        payload_for_hash = {
            "title": title,
            "link": link,
            "source": source,
            "platform": platform,
            "is_social": is_social,
        }
        m_hash = hash_payload(payload_for_hash)

        # Extract probable author handle or name from URL/title
        author_name, author_handle = self._extract_author_hints(link, title)

        # Profile Page & Avatar Extractor (Iteration 2)
        if not thumbnail and link.startswith(("http://", "https://")):
            og_meta = extract_opengraph_metadata(link, session=self.session)
            if og_meta.get("og_image"):
                thumbnail = og_meta["og_image"]
            if not author_name and og_meta.get("author"):
                author_name = og_meta["author"]
            if not author_name and og_meta.get("og_title"):
                author_name, _ = self._extract_author_hints(link, og_meta["og_title"] or "")

        return SearchMatch(
            title=title,
            link=link,
            source=source,
            platform=platform,
            thumbnail=thumbnail,
            is_social=is_social,
            match_hash=m_hash,
            timestamp_utc=now_iso,
            raw_snippet=match.get("snippet"),
            author_name=author_name,
            author_handle=author_handle,
        )

    def extract_profile_page_avatar(self, url: str) -> Optional[str]:
        """
        Extracts avatar or thumbnail image directly from a target profile page (Iteration 2).
        """
        og = extract_opengraph_metadata(url, session=self.session)
        return og.get("og_image")

    def _platform_to_domain(self, platform: str) -> Optional[str]:
        p = platform.lower()
        if "twitter" in p or "x" in p:
            return "twitter.com"
        if "linkedin" in p:
            return "linkedin.com"
        if "github" in p:
            return "github.com"
        if "instagram" in p:
            return "instagram.com"
        if "facebook" in p:
            return "facebook.com"
        if "youtube" in p:
            return "youtube.com"
        if "wikipedia" in p:
            return "wikipedia.org"
        return None

    def _extract_canonical_name(self, lens_res: Any, raw_matches: List[Any]) -> str:
        """Extracts the cleanest canonical person name from knowledge graph or visual matches."""
        # 1. Check knowledge graph
        kg = lens_res.get("knowledge_graph") if lens_res else None
        if kg and hasattr(kg, "get"):
            kg_title = kg.get("title")
            if kg_title and len(kg_title.strip()) > 1:
                return kg_title.strip()

        # 2. Statistical extraction across visual match titles
        titles = [m.get("title", "") for m in raw_matches if hasattr(m, "get") and m.get("title")]
        if not titles:
            return "Unknown"

        stop_words = {
            "ideas", "birthday", "photos", "photo", "actor", "actress", "south", "indian",
            "from", "best", "films", "film", "trending", "sketch", "hero", "movie", "movies",
            "starrer", "full", "hd", "wallpaper", "wallpapers", "new", "know", "all", "about",
            "completes", "years", "national", "award", "winner", "becomes", "first", "million",
            "followers", "threads", "garners", "insta", "image", "images", "pics", "pic", "wiki",
            "wikipedia", "biography", "age", "height", "net", "worth", "wife", "husband", "family",
            "profile", "news", "exclusive", "watch", "official", "video", "videos", "song", "songs",
            "cinema", "stills", "gallery", "look", "looks", "lookout", "latest", "update", "updates",
            "article", "report", "story", "stories", "celebrity", "star", "director", "producer"
        }

        import re
        from collections import Counter

        cleaned_phrases = []
        for t in titles:
            part = re.split(r'[\-|:•–—\(\)\[\]#\?]', t)[0].strip()
            words = [w for w in re.findall(r'[A-Za-z]+', part) if w.lower() not in stop_words]
            if len(words) >= 2:
                cleaned_phrases.append(" ".join(words[:2]).title())
            elif len(words) == 1 and len(words[0]) > 2:
                cleaned_phrases.append(words[0].title())

        if cleaned_phrases:
            c = Counter(cleaned_phrases)
            return c.most_common(1)[0][0]

        return "Unknown"

    def _is_valid_profile_url(self, url: str, platform: str) -> bool:
        """Validates that a URL is an authentic personal profile, not a post, tag, feed, or news link."""
        if not url:
            return False
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        path = parsed.path.strip("/")
        parts = [seg for seg in path.split("/") if seg]
        p = platform.lower()

        if "twitter" in p or "x" in p:
            if not ("twitter.com" in netloc or "x.com" in netloc):
                return False
            if not parts:
                return False
            handle = parts[0].lower()
            if handle in {"status", "i", "explore", "search", "hashtag", "intent", "home", "notifications", "settings", "tos", "privacy", "about"}:
                return False
            return True

        if "instagram" in p:
            if "instagram.com" not in netloc:
                return False
            if not parts:
                return False
            handle = parts[0].lower()
            if handle in {"p", "reel", "reels", "explore", "stories", "tv", "accounts", "direct", "developer", "about", "legal", "terms"}:
                return False
            # Reject fan accounts, fan clubs, updates, parody pages
            if any(k in handle for k in ["fan", "fc", "club", "update", "tribute", "parody"]):
                return False
            return True

        if "linkedin" in p:
            if "linkedin.com" not in netloc:
                return False
            return len(parts) >= 2 and parts[0] == "in"

        if "github" in p:
            if "github.com" not in netloc:
                return False
            if len(parts) != 1:
                return False
            user = parts[0].lower()
            if user in {"topics", "features", "search", "orgs", "about", "pricing", "marketplace", "settings", "explore", "trending"}:
                return False
            return True

        if "web" in p:
            if "en.wikipedia.org" in netloc or "wikipedia.org" in netloc:
                if len(parts) >= 2 and parts[0] == "wiki":
                    page = parts[1].lower()
                    if not any(page.startswith(prefix) for prefix in ["list_of", "template:", "category:", "special:", "help:"]):
                        if not any(x in page for x in ["filmography", "discography", "bibliography", "videography", "awards"]):
                            return True
            elif "imdb.com" in netloc:
                if len(parts) >= 2 and parts[0] == "name":
                    return True
            return False

        return True

    def _search_authentic_platform_profile(
        self,
        client: serpapi.Client,
        platform: str,
        entity_name: str,
        visual_matches: List[SearchMatch],
        fallback_thumb: Optional[str] = None,
    ) -> Optional[SearchMatch]:
        """Resolves authentic profile URLs for a given platform and entity."""
        now_iso = datetime.now(timezone.utc).isoformat()

        # Stage 2A: Query targeted Google search for authentic profile
        query = self._build_platform_query(entity_name, platform)
        if query:
            try:
                s_res = client.search({"engine": "google", "q": query, "num": 10})
                organic = s_res.get("organic_results", []) if s_res else []
                for r in organic:
                    link = r.get("link", "")
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")
                    if not self._is_valid_profile_url(link, platform):
                        continue

                    norm_link = self._normalize_social_url(link, platform)
                    auth_name, handle = self._extract_author_hints(norm_link, title)

                    # For web: extract clean author name from Wikipedia / IMDb
                    if platform.lower() == "web":
                        if "wikipedia" in norm_link:
                            auth_name = title.split(" - Wikipedia")[0].split(" (")[0].strip()
                        elif "imdb" in norm_link:
                            auth_name = title.split(" - IMDb")[0].split(" (")[0].strip()
                        else:
                            auth_name = entity_name
                    else:
                        if not auth_name or auth_name == "Unknown":
                            auth_name = entity_name

                    # Verify name alignment between discovered profile and canonical entity
                    sim = string_similarity(auth_name, entity_name)
                    if entity_name.lower() not in title.lower() and sim < 0.60:
                        continue

                    # Filter out fan clubs / parody pages from title or snippet
                    lower_text = (title + " " + snippet).lower()
                    if any(bad in lower_text for bad in ["fan club", "fan account", "fan page", "tribute page", "parody"]):
                        continue

                    if not handle and norm_link and platform.lower() != "web":
                        h_part = norm_link.rstrip("/").split("/")[-1]
                        if h_part:
                            handle = f"@{h_part}"

                    payload = {
                        "title": title,
                        "link": norm_link,
                        "source": platform,
                        "platform": platform,
                        "is_social": platform.lower() != "web",
                    }
                    m_hash = hash_payload(payload)
                    return SearchMatch(
                        title=title,
                        link=norm_link,
                        source=platform,
                        platform=platform,
                        thumbnail=fallback_thumb,
                        is_social=platform.lower() != "web",
                        match_hash=m_hash,
                        timestamp_utc=now_iso,
                        raw_snippet=snippet,
                        author_name=auth_name,
                        author_handle=handle,
                    )
            except Exception:
                pass

        # Stage 2B: Fallback to checking visual matches for this platform
        for m in visual_matches:
            if identify_platform(m.link) == platform and self._is_valid_profile_url(m.link, platform):
                norm_link = self._normalize_social_url(m.link, platform)
                auth_name, handle = self._extract_author_hints(norm_link, m.title)
                if not handle and platform.lower() != "web":
                    parts = norm_link.rstrip("/").split("/")
                    if parts:
                        handle = f"@{parts[-1]}"

                # Verify name similarity before accepting visual match fallback
                check_name = auth_name or m.author_name or entity_name
                if string_similarity(check_name, entity_name) >= 0.60 or entity_name.lower() in m.title.lower():
                    m.link = norm_link
                    m.platform = platform
                    m.author_name = check_name
                    m.author_handle = handle
                    m.is_social = platform.lower() != "web"
                    return m

        return None

    def _build_platform_query(self, entity_name: str, platform: str) -> Optional[str]:
        """Builds targeted Google queries to find authentic profile handles for a given person."""
        p = platform.lower()
        clean = entity_name.replace('"', '').strip()
        if not clean or clean.lower() in ("unknown", "discovered match", "none"):
            return None
        if "twitter" in p or "x" in p:
            return f'"{clean}" (site:twitter.com OR site:x.com)'
        if "instagram" in p:
            return f'"{clean}" site:instagram.com'
        if "facebook" in p:
            return f'"{clean}" site:facebook.com'
        if "youtube" in p:
            return f'"{clean}" (site:youtube.com/@ OR site:youtube.com/c/ OR site:youtube.com/channel/)'
        if "linkedin" in p:
            return f'"{clean}" site:linkedin.com/in'
        if "github" in p:
            return f'"{clean}" site:github.com'
        if "web" in p:
            return f'"{clean}" (site:en.wikipedia.org/wiki/ OR site:imdb.com/name/)'
        return None

    def _normalize_social_url(self, url: str, platform: str) -> str:
        """Extracts the canonical base user profile URL from post/tweet/reel URLs."""
        if not url:
            return ""
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.strip("/")
        parts = [seg for seg in path.split("/") if seg]
        p = platform.lower()

        if "twitter" in p or "x" in p:
            if parts:
                return f"https://x.com/{parts[0]}"
            return url.split("?")[0].rstrip("/")

        elif "instagram" in p:
            if parts:
                handle = parts[0]
                if handle not in {"p", "reel", "reels", "explore", "stories", "tv"}:
                    return f"https://www.instagram.com/{handle}"
            return url.split("?")[0].rstrip("/")

        elif "linkedin" in p:
            if len(parts) >= 2 and parts[0] == "in":
                return f"https://www.linkedin.com/in/{parts[1]}"
            return url.split("?")[0].rstrip("/")

        elif "github" in p:
            if parts:
                return f"https://github.com/{parts[0]}"
            return url.split("?")[0].rstrip("/")

        elif "web" in p:
            if "wikipedia.org" in url:
                if parts and parts[0] == "wiki":
                    return f"https://en.wikipedia.org/wiki/{parts[1]}"
            elif "imdb.com" in url:
                if len(parts) >= 2 and parts[0] == "name":
                    return f"https://www.imdb.com/name/{parts[1]}/"
            return url.split("?")[0].rstrip("/")

        return url.split("?")[0].rstrip("/")

    def _extract_author_hints(self, url: str, title: str) -> tuple[Optional[str], Optional[str]]:
        """Heuristic author name and username extractor from URL and title."""
        url_lower = url.lower()
        handle = None
        name = None

        if "twitter.com/" in url_lower or "x.com/" in url_lower:
            parts = url.split(".com/")[-1].split("?")[0].split("/")
            if parts and parts[0] not in {"status", "i", "explore", "search"}:
                handle = f"@{parts[0]}"
        elif "instagram.com/" in url_lower:
            parts = url.split("instagram.com/")[-1].split("?")[0].split("/")
            if parts and parts[0] not in {"p", "reel", "explore", "stories", "tv"}:
                handle = f"@{parts[0]}"
        elif "facebook.com/" in url_lower:
            parts = url.split("facebook.com/")[-1].split("?")[0].split("/")
            if parts and parts[0] not in {"watch", "groups", "pages", "share", "profile.php"}:
                handle = f"@{parts[0]}"
        elif "youtube.com/" in url_lower:
            if "/@" in url_lower:
                handle = "@" + url.split("/@")[-1].split("?")[0].split("/")[0]
            elif "/c/" in url_lower or "/channel/" in url_lower:
                parts = url.split("youtube.com/")[-1].split("?")[0].split("/")
                if len(parts) > 1:
                    handle = f"@{parts[1]}"
        elif "github.com/" in url_lower:
            parts = url.split("github.com/")[-1].split("?")[0].split("/")
            if parts and parts[0] not in {"orgs", "topics", "features"}:
                handle = f"@{parts[0]}"
        elif "linkedin.com/in/" in url_lower:
            parts = url.split("linkedin.com/in/")[-1].split("?")[0].split("/")
            if parts:
                handle = f"@{parts[0]}"

        # Check if handle is in title e.g. "Sambhavna Seth (@sambhavnasethofficial)"
        if not handle:
            import re
            h_match = re.search(r'\(@([a-zA-Z0-9_\.]+)\)', title)
            if h_match:
                handle = f"@{h_match.group(1)}"

        # Clean person name extraction
        if " (@" in title:
            name = title.split(" (@")[0].strip()
        elif " - Wikipedia" in title:
            name = title.split(" - Wikipedia")[0].strip()
        elif " - " in title:
            name = title.split(" - ")[0].strip()
        elif " | " in title:
            name = title.split(" | ")[0].strip()
        elif " (" in title:
            name = title.split(" (")[0].strip()
        elif " on X" in title:
            name = title.split(" on X")[0].strip()
        elif " on Instagram" in title:
            name = title.split(" on Instagram")[0].strip()
        elif " on Facebook" in title:
            name = title.split(" on Facebook")[0].strip()
        else:
            name = title[:35].strip()

        return name, handle

    def _generate_simulated_multi_site(
        self,
        image_path: str,
        platforms: List[str],
        retry_level: int = 0,
        simulate_disparity: bool = False,
    ) -> MultiSiteSearchResult:
        """
        Generates realistic multi-site profile discoveries for testing.
        Simulates:
          - Agreement: Twitter, LinkedIn, GitHub pointing to the same identity ('Alex Morgan, Cryptography Researcher')
          - Disparity (if requested): Divergent identities across sites to test retry & disparity reporting
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        matches_by_platform: Dict[str, List[SearchMatch]] = {}
        all_matches: List[SearchMatch] = []

        # Canonical entity identity
        canonical_name = "Alex Morgan"
        canonical_handle = "alex_morgan_sec"

        for p in platforms:
            if simulate_disparity and p == "Instagram":
                # Simulated divergent entity to test disparity logic
                site_name = "Taylor Swift Fanpage"
                site_handle = "taylor_fan_official"
                site_url = "https://instagram.com/p/C982349823_fan"
                site_title = "Taylor Swift Fan Community Update"
                site_thumb = "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=200"
            else:
                site_name = canonical_name
                site_handle = canonical_handle
                if p == "Twitter/X":
                    site_url = f"https://twitter.com/{site_handle}/status/1789012345678901234"
                    site_title = f"{canonical_name} (@{site_handle}) on X: Biometric Blockchain Research"
                    site_thumb = image_path if os.path.exists(image_path) else "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=200"
                elif p == "LinkedIn":
                    site_url = f"https://www.linkedin.com/in/{site_handle}"
                    site_title = f"{canonical_name} — Lead Cryptographic Security Scientist | LinkedIn"
                    site_thumb = image_path if os.path.exists(image_path) else "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=200"
                elif p == "GitHub":
                    site_url = f"https://github.com/{site_handle}"
                    site_title = f"{site_handle} ({canonical_name}) · GitHub Decentralized Identity"
                    site_thumb = image_path if os.path.exists(image_path) else "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=200"
                elif p == "Instagram":
                    site_url = f"https://instagram.com/{site_handle}"
                    site_title = f"{canonical_name} (@{site_handle}) • Instagram Identity & AI"
                    site_thumb = image_path if os.path.exists(image_path) else "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=200"
                else:
                    site_url = f"https://identity-summit.org/speakers/{site_handle}"
                    site_title = f"Speaker Profile: {canonical_name} | Web3 Identity Summit"
                    site_thumb = image_path if os.path.exists(image_path) else "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=200"

            m_hash = hash_payload({"url": site_url, "title": site_title, "platform": p})
            match_obj = SearchMatch(
                title=site_title,
                link=site_url,
                source=p,
                platform=p,
                thumbnail=site_thumb,
                is_social=(p in self.DEFAULT_TARGET_PLATFORMS[:4]),
                match_hash=m_hash,
                timestamp_utc=now_iso,
                raw_snippet=f"Official profile and publications for {site_name}.",
                author_name=site_name,
                author_handle=site_handle,
            )
            matches_by_platform[p] = [match_obj]
            all_matches.append(match_obj)

        return MultiSiteSearchResult(
            target_platforms=platforms,
            matches_by_platform=matches_by_platform,
            all_matches=all_matches,
            total_found=len(all_matches),
            queried_at_utc=now_iso,
        )

    def _generate_simulated_match(self, image_path: str) -> SearchMatch:
        """Fallback single match."""
        multi = self._generate_simulated_multi_site(image_path, ["Twitter/X"])
        return multi.all_matches[0]
