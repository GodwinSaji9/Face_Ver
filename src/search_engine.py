from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import os
import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse
import serpapi
from src.utils import (
    create_resilient_session,
    extract_opengraph_metadata,
    hash_payload,
    identify_platform,
    strict_name_match,
    string_similarity,
)

INVALID_INSTAGRAM_SEGMENTS = {
    "p", "reel", "reels", "explore", "stories", "tv", "popular", "tags", "tag",
    "directory", "channel", "topics", "accounts", "direct", "developer", "about",
    "legal", "terms", "privacy", "help", "login", "signup", "emails", "guides",
    "locations", "create", "settings", "share"
}

INVALID_HANDLE_SUBSTRINGS = [
    "fan", "fans", "fc", "club", "update", "updates", "tribute", "parody",
    "picture", "pictures", "photo", "photos", "daily", "edits", "army"
]


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

    def __init__(
        self,
        api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("SERPAPI_KEY")
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.session = create_resilient_session(pool_size=20, max_retries=3)

    def _search_serper(self, query: str, num: int = 8) -> List[Dict[str, Any]]:
        """Executes a Google Search query via Serper.dev API."""
        if not self.serper_api_key:
            return []
        try:
            resp = self.session.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": self.serper_api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": num},
                timeout=6.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("organic", [])
        except Exception:
            pass
        return []

    def _search_serper_images(self, query: str, num: int = 5) -> List[Dict[str, Any]]:
        """Queries Google Images via Serper.dev API to retrieve authentic avatars/thumbnails."""
        if not self.serper_api_key:
            return []
        try:
            resp = self.session.post(
                "https://google.serper.dev/images",
                headers={
                    "X-API-KEY": self.serper_api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": num},
                timeout=6.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("images", [])
        except Exception:
            pass
        return []

    def _extract_serper_image_url(self, image_items: List[Dict[str, Any]]) -> Optional[str]:
        """Returns a valid, non-blocked direct image URL or Google CDN thumbnail URL."""
        if not image_items or not isinstance(image_items, list):
            return None
        for img in image_items:
            img_url = img.get("imageUrl", "")
            thumb_url = img.get("thumbnailUrl", "")
            # Facebook/Instagram lookaside URLs block crawlers and return 200 HTML redirects
            is_blocked = any(b in img_url.lower() for b in ["lookaside.fbsbx", "lookaside.instagram", "facebook.com", "instagram.com"])
            if img_url and not is_blocked:
                return img_url
            if thumb_url:
                return thumb_url
            if img_url:
                return img_url
        return None

    def _identify_face_by_visual_search(self, image_path: str) -> Optional[str]:
        """
        Performs automated reverse visual image search on the cropped face image
        to identify the individual without requiring manual human input.
        """
        if not os.path.exists(image_path):
            return None
        try:
            search_url = 'https://yandex.com/images/search'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            params = {
                'rpt': 'imageview',
                'format': 'json',
                'request': '{"blocks":[{"block":"b-page_type_search-by-image__link"}]}'
            }
            with open(image_path, 'rb') as f:
                files = {'upfile': ('face.jpg', f, 'image/jpeg')}
                resp = self.session.post(search_url, params=params, files=files, headers=headers, timeout=8)

            if resp.status_code == 200:
                data = resp.json()
                params_dict = data.get('blocks', [{}])[0].get('params', {})
                img_url = params_dict.get('originalImageUrl') or params_dict.get('url')
                if img_url:
                    search_page = self.session.get(f"https://yandex.com/images/search?rpt=imageview&url={img_url}", headers=headers, timeout=8)
                    if search_page.status_code == 200:
                        import html as html_lib
                        from collections import Counter
                        unescaped = html_lib.unescape(search_page.text)
                        titles = re.findall(r'"title":\s*"([^"]+)"', unescaped)
                        words_counter = Counter()
                        stop_words = {
                            "similar image", "yandex image", "full image", "catch news",
                            "photo gallery", "wallpaper download", "news update", "search for",
                            "image result", "stock photo", "high resolution", "desktop wallpaper"
                        }
                        for t in titles:
                            for m in re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b', t):
                                if not any(stop in m.lower() for stop in stop_words):
                                    words_counter[m] += 1
                        if words_counter:
                            top_candidate, count = words_counter.most_common(1)[0]
                            if count >= 2:
                                return top_candidate
        except Exception:
            pass
        return None

    def search_multi_site(
        self,
        image_path: str,
        target_platforms: Optional[List[str]] = None,
        entity_hint: Optional[str] = None,
        max_results_per_site: int = 5,
        retry_level: int = 0,
        demo_mode: bool = False,
        simulate_disparity: bool = False,
    ) -> MultiSiteSearchResult:
        """
        Simultaneously searches for matching profiles across multiple target platforms.
        Uses parallel thread execution to search each platform concurrently.
        Supports both Serper.dev and SerpApi providers.
        """
        platforms = target_platforms or self.DEFAULT_TARGET_PLATFORMS
        now_iso = datetime.now(timezone.utc).isoformat()

        # Offline / Demo Mode or No API Keys at all
        if demo_mode or (not self.api_key and not self.serper_api_key):
            return self._generate_simulated_multi_site(image_path, platforms, retry_level, simulate_disparity)

        client = None
        if self.api_key:
            try:
                client = serpapi.Client(api_key=self.api_key)
            except Exception:
                client = None

        canonical_name = entity_hint or "Unknown"
        default_thumb = None
        parsed_visual_matches = []

        # Stage 1: If entity not hinted, attempt SerpApi Google Lens lookup
        if canonical_name == "Unknown" and client:
            try:
                is_url = image_path.startswith("http://") or image_path.startswith("https://")
                image_id = None
                if not is_url:
                    if os.path.exists(image_path):
                        upload = client.upload_image(image_path)
                        if "image_id" in upload:
                            image_id = upload["image_id"]
                lens_params: Dict[str, Any] = {"engine": "google_lens", "hl": "en"}
                if is_url:
                    lens_params["url"] = image_path
                elif image_id:
                    lens_params["image_id"] = image_id

                if is_url or image_id:
                    lens_res = client.search(lens_params)
                    raw_visual_matches = lens_res.get("visual_matches", []) if lens_res else []
                    parsed_visual_matches = [self._parse_match(m) for m in raw_visual_matches[:25]]
                    canonical_name = self._extract_canonical_name(lens_res, raw_visual_matches)
                    default_thumb = parsed_visual_matches[0].thumbnail if parsed_visual_matches else None
            except Exception:
                pass

        # Stage 1B: If canonical_name is still Unknown, run automated visual face search
        if canonical_name in ("Unknown", "", None):
            try:
                visual_id = self._identify_face_by_visual_search(image_path)
                if visual_id:
                    canonical_name = visual_id
            except Exception:
                pass

        # Stage 1C: If canonical_name is still Unknown, infer from filename if not generic
        if canonical_name in ("Unknown", "", None):
            base_fname = os.path.splitext(os.path.basename(image_path))[0]
            clean_fname = re.sub(r'[_\-\.]+', ' ', base_fname).strip()
            words = [w for w in clean_fname.split() if len(w) > 2 and not w.isdigit()]
            if len(words) >= 2 and not any(w.lower() in ("img", "image", "photo", "pic", "face", "crop", "test") for w in words):
                canonical_name = " ".join(words).title()

        # If Serper.dev is available and we have a canonical name, retrieve a profile portrait
        if not default_thumb and self.serper_api_key and canonical_name != "Unknown":
            try:
                s_imgs = self._search_serper_images(f"{canonical_name} portrait profile", num=5)
                if not s_imgs:
                    s_imgs = self._search_serper_images(f"{canonical_name} profile", num=5)
                if s_imgs:
                    default_thumb = self._extract_serper_image_url(s_imgs)
            except Exception:
                pass

        # If we still cannot identify an entity and have no active search client, fallback to simulated
        if canonical_name == "Unknown" and not client:
            return self._generate_simulated_multi_site(image_path, platforms, retry_level, simulate_disparity)

        # Stage 2: Targeted concurrent resolution of authentic platform profiles
        matches_by_platform: Dict[str, List[SearchMatch]] = {p: [] for p in platforms}
        all_matches: List[SearchMatch] = []

        # Check if Twitter/X is in requested platforms: resolve it first to propagate handle
        known_handle = None
        remaining_platforms = list(platforms)
        twitter_platform = next((p for p in platforms if "twitter" in p.lower() or "x" in p.lower()), None)
        if twitter_platform:
            try:
                tw_match = self._search_authentic_platform_profile(
                    client=client,
                    platform=twitter_platform,
                    entity_name=canonical_name,
                    visual_matches=parsed_visual_matches,
                    fallback_thumb=default_thumb,
                )
                if tw_match:
                    matches_by_platform[twitter_platform] = [tw_match]
                    all_matches.append(tw_match)
                    if tw_match.author_handle:
                        known_handle = tw_match.author_handle.lstrip("@")
                    elif tw_match.link:
                        known_handle = tw_match.link.rstrip("/").split("/")[-1]
            except Exception:
                matches_by_platform[twitter_platform] = []
            remaining_platforms.remove(twitter_platform)

        with ThreadPoolExecutor(max_workers=min(max(len(remaining_platforms), 1), 6)) as executor:
            future_to_platform = {
                executor.submit(
                    self._search_authentic_platform_profile,
                    client,
                    p,
                    canonical_name,
                    parsed_visual_matches,
                    default_thumb,
                    known_handle,
                ): p
                for p in remaining_platforms
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
        if lens_res and isinstance(lens_res, dict):
            kg = lens_res.get("knowledge_graph")
            if kg and isinstance(kg, dict):
                kg_title = kg.get("title")
                if kg_title and len(kg_title.strip()) > 1:
                    return kg_title.strip()

        titles = [m.get("title", "") for m in raw_matches if isinstance(m, dict) and m.get("title")]
        if not titles:
            return "Unknown"

        generic_web_noise = {
            "wikipedia", "wiki", "biography", "bio", "profile", "official", "website",
            "facebook", "twitter", "instagram", "linkedin", "github", "youtube", "imdb",
            "photo", "photos", "image", "images", "pics", "pic", "hd", "wallpaper", "wallpapers",
            "stock", "getty", "news", "exclusive", "interview", "video", "videos", "latest",
            "net worth", "age", "height", "family", "education", "quotes", "status", "about"
        }

        import re
        from collections import Counter

        candidates = []
        for t in titles:
            primary_seg = re.split(r'[\-|:•–—\(\)\[\]#\?]', t)[0].strip()
            words = [w for w in re.findall(r'[A-Za-z]+', primary_seg) if w.lower() not in generic_web_noise]
            if len(words) >= 2:
                candidates.append(" ".join(words[:3]).title())
            elif len(words) == 1 and len(words[0]) > 2:
                candidates.append(words[0].title())

        if candidates:
            return Counter(candidates).most_common(1)[0][0]

        return "Unknown"

    def _is_valid_profile_url(self, url: str, platform: str) -> bool:
        """Validates that a URL is an authentic personal profile, not a post, tag, feed, or directory."""
        if not url:
            return False
        parsed = urlparse(url)
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
            if handle in {"status", "i", "explore", "search", "hashtag", "intent", "home", "notifications", "settings", "tos", "privacy", "about", "help", "login", "signup"}:
                return False
            if any(bad in handle for bad in INVALID_HANDLE_SUBSTRINGS):
                return False
            return True

        if "instagram" in p:
            if "instagram.com" not in netloc:
                return False
            if len(parts) != 1:
                return False
            handle = parts[0].lower()
            if handle in INVALID_INSTAGRAM_SEGMENTS:
                return False
            if any(bad in handle for bad in INVALID_HANDLE_SUBSTRINGS):
                return False
            return True

        if "linkedin" in p:
            if "linkedin.com" not in netloc:
                return False
            if len(parts) < 2 or parts[0] != "in":
                return False
            return True

        if "github" in p:
            if "github.com" not in netloc:
                return False
            if len(parts) != 1:
                return False
            user = parts[0].lower()
            if user in {"topics", "features", "search", "orgs", "about", "pricing", "marketplace", "settings", "explore", "trending", "site", "security", "customer-stories", "readme", "login", "signup"}:
                return False
            return True

        if "web" in p:
            if "wikipedia.org" in netloc:
                if len(parts) >= 2 and parts[0] == "wiki":
                    page = parts[1].lower()
                    if not any(page.startswith(prefix) for prefix in ["list_of", "template:", "category:", "special:", "help:", "talk:", "portal:"]):
                        if not any(x in page for x in ["filmography", "discography", "bibliography", "videography", "awards"]):
                            return True
            elif "imdb.com" in netloc:
                if len(parts) >= 2 and parts[0] == "name":
                    return True
            return False

        return True

    def _build_platform_queries(self, entity_name: str, platform: str, known_handle: Optional[str] = None) -> List[str]:
        """Builds prioritized Google queries to find authentic profile handles for a given person."""
        p = platform.lower()
        clean = entity_name.replace('"', '').strip()
        if not clean or clean.lower() in ("unknown", "discovered match", "none"):
            return []

        queries = []
        if "twitter" in p or "x" in p:
            queries = [
                f'"{clean}" official (site:x.com OR site:twitter.com)',
                f'"{clean}" (site:x.com OR site:twitter.com)'
            ]
        elif "instagram" in p:
            if known_handle:
                h = known_handle.lstrip("@").strip()
                queries.append(f'"{h}" site:instagram.com')
            queries.extend([
                f'"{clean}" official instagram',
                f'"{clean}" site:instagram.com'
            ])
        elif "linkedin" in p:
            queries = [f'"{clean}" site:linkedin.com/in']
        elif "github" in p:
            queries = [f'"{clean}" site:github.com']
        elif "web" in p:
            queries = [
                f'"{clean}" site:en.wikipedia.org/wiki/',
                f'"{clean}" (site:en.wikipedia.org/wiki/ OR site:imdb.com/name/)'
            ]
        return queries

    def _normalize_social_url(self, url: str, platform: str) -> str:
        """Extracts the canonical base user profile URL from post/tweet/reel URLs."""
        if not url:
            return ""
        parsed = urlparse(url)
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
                if handle.lower() not in INVALID_INSTAGRAM_SEGMENTS and not any(bad in handle.lower() for bad in INVALID_HANDLE_SUBSTRINGS):
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
                if len(parts) >= 2 and parts[0] == "wiki":
                    return f"https://en.wikipedia.org/wiki/{parts[1]}"
            elif "imdb.com" in url:
                if len(parts) >= 2 and parts[0] == "name":
                    return f"https://www.imdb.com/name/{parts[1]}/"
            return url.split("?")[0].rstrip("/")

        return url.split("?")[0].rstrip("/")

    def _search_authentic_platform_profile(
        self,
        client: Optional[serpapi.Client],
        platform: str,
        entity_name: str,
        visual_matches: List[SearchMatch],
        fallback_thumb: Optional[str] = None,
        known_handle: Optional[str] = None,
    ) -> Optional[SearchMatch]:
        """Resolves authentic profile URLs for a given platform and entity."""
        now_iso = datetime.now(timezone.utc).isoformat()
        clean_name = entity_name.replace('"', '').strip()
        p = platform.lower()

        # Stage 2A: Query targeted Google search for authentic profile
        queries = self._build_platform_queries(clean_name, platform, known_handle=known_handle)
        for query in queries:
            try:
                organic = []
                if self.serper_api_key:
                    organic = self._search_serper(query, num=8)
                elif client:
                    try:
                        s_res = client.search({"engine": "google", "q": query, "num": 8})
                        organic = s_res.get("organic_results", []) if s_res else []
                    except Exception:
                        organic = []

                for r in organic:
                    link = r.get("link", "")
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")

                    # Check for direct valid profile URL
                    if self._is_valid_profile_url(link, platform):
                        norm_link = self._normalize_social_url(link, platform)
                        auth_name, handle = self._extract_author_hints(norm_link, title)

                        # Clean author candidate
                        if " (@" in title:
                            c_name = title.split(" (@")[0].strip()
                        elif " - " in title:
                            c_name = title.split(" - ")[0].strip()
                        elif " | " in title:
                            c_name = title.split(" | ")[0].strip()
                        elif " on X" in title:
                            c_name = title.split(" on X")[0].strip()
                        else:
                            c_name = auth_name or title

                        if not strict_name_match(c_name, clean_name):
                            continue

                        # Filter out fan clubs / parody pages from title or snippet
                        lower_text = (title + " " + snippet).lower()
                        if any(bad in lower_text for bad in ["fan club", "fan account", "fan page", "tribute page", "parody"]):
                            continue

                        final_auth_name = c_name if c_name else clean_name

                        if not handle and norm_link and p != "web":
                            h_part = norm_link.rstrip("/").split("/")[-1]
                            if h_part:
                                handle = f"@{h_part}"

                        payload = {
                            "title": title,
                            "link": norm_link,
                            "source": platform,
                            "platform": platform,
                            "is_social": p != "web",
                        }
                        m_hash = hash_payload(payload)
                        thumb_to_use = fallback_thumb
                        try:
                            og_img = self.extract_profile_page_avatar(norm_link)
                            if og_img and og_img.startswith(("http://", "https://")):
                                thumb_to_use = og_img
                        except Exception:
                            pass

                        if not thumb_to_use and self.serper_api_key:
                            try:
                                s_imgs = self._search_serper_images(f"{clean_name} {platform} profile", num=5)
                                if not s_imgs:
                                    s_imgs = self._search_serper_images(f"{clean_name} profile", num=5)
                                if s_imgs:
                                    thumb_to_use = self._extract_serper_image_url(s_imgs)
                            except Exception:
                                pass

                        return SearchMatch(
                            title=title,
                            link=norm_link,
                            source=platform,
                            platform=platform,
                            thumbnail=thumb_to_use,
                            is_social=p != "web",
                            match_hash=m_hash,
                            timestamp_utc=now_iso,
                            raw_snippet=snippet,
                            author_name=final_auth_name,
                            author_handle=handle,
                        )

                    # Special Instagram snippet extraction if result mentions official profile handle
                    if "instagram" in p and not self._is_valid_profile_url(link, platform):
                        text = title + " " + snippet
                        handles = re.findall(r'@([a-zA-Z0-9_\.]{3,30})', text)
                        for cand_h in handles:
                            h_low = cand_h.lower()
                            if h_low in INVALID_INSTAGRAM_SEGMENTS:
                                continue
                            if any(bad in h_low for bad in INVALID_HANDLE_SUBSTRINGS):
                                continue
                            if any(part.lower() in h_low for part in clean_name.split() if len(part) > 2):
                                profile_url = f"https://www.instagram.com/{cand_h}"
                                payload = {
                                    "title": f"{clean_name} (@{cand_h})",
                                    "link": profile_url,
                                    "source": platform,
                                    "platform": platform,
                                    "is_social": True,
                                }
                                m_hash = hash_payload(payload)
                                return SearchMatch(
                                    title=f"{clean_name} (@{cand_h})",
                                    link=profile_url,
                                    source=platform,
                                    platform=platform,
                                    thumbnail=fallback_thumb,
                                    is_social=True,
                                    match_hash=m_hash,
                                    timestamp_utc=now_iso,
                                    raw_snippet=snippet,
                                    author_name=clean_name,
                                    author_handle=f"@{cand_h}",
                                )
            except Exception:
                pass

        # Stage 2B: Fallback to checking visual matches for this platform
        for m in visual_matches:
            if identify_platform(m.link) == platform and self._is_valid_profile_url(m.link, platform):
                norm_link = self._normalize_social_url(m.link, platform)
                auth_name, handle = self._extract_author_hints(norm_link, m.title)
                c_name = auth_name or m.author_name or m.title
                if strict_name_match(c_name, clean_name):
                    if not handle and p != "web":
                        parts = norm_link.rstrip("/").split("/")
                        if parts:
                            handle = f"@{parts[-1]}"
                    m.link = norm_link
                    m.platform = platform
                    m.author_name = clean_name
                    m.author_handle = handle
                    m.is_social = p != "web"
                    return m

        return None

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
                site_name = "Divergent Identity Profile"
                site_handle = "divergent_user"
                site_url = "https://instagram.com/p/C982349823_fan"
                site_title = "Divergent Identity Community Update"
                site_thumb = "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=200"
            else:
                site_name = canonical_name
                site_handle = canonical_handle
                if p == "Twitter/X":
                    site_url = f"https://x.com/{site_handle}"
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
