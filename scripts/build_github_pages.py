"""Build a public-only Pages artifact without changing the domain-root sources."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "https://riemenschneider2015-svg.github.io/linguoai-de/"
LOCALES = ("de", "en", "ru", "uk", "tr", "es", "it", "fr", "pt", "pl", "nl")
PRODUCT_PAGES = ("index.html", "colab.html", "anleitung.html")
PUBLIC_PAGES = (
    *PRODUCT_PAGES,
    "web.html",
    "404.html",
    "impressum.html",
    "datenschutz.html",
    "drive-datenschutz.html",
)
PUBLIC_ASSETS = (
    "colab-guide.css",
    "editorial-guides.css",
    "editorial.css",
    "favicon.svg",
    "funnel.css",
    "guide.css",
    "locale.js",
    "logo-mark.svg",
    "simple.css",
    "site.js",
    "styles.css",
    "web-app.css",
    "web-app.js",
    "fonts/Manrope-OFL.txt",
    "fonts/Manrope-Variable.woff2",
)
DOWNLOAD_NAME = re.compile(
    r"(?:LinguoAI-\d+\.\d+\.\d+-source\.zip|"
    r"linguoai-\d+\.\d+\.\d+-py3-none-any\.whl|"
    r"SHA256SUMS-\d+\.\d+\.\d+\.txt|LinguoAI_Colab\.ipynb)\Z"
)
OLD_ORIGIN = re.compile(r"https?://(?:www\.)?linguoai\.de(?=[/?#\s\"'<>`]|$)")
JS_PATH = re.compile(r"(?P<quote>[\"'`])(?P<path>/(?!/)[^\"'`\r\n]*)(?P=quote)")
CSS_URL = re.compile(r"url\(\s*(?P<quote>[\"']?)(?P<url>[^)\s\"']+)(?P=quote)\s*\)")


@dataclass(frozen=True)
class SiteURLs:
    base_url: str

    def __post_init__(self) -> None:
        parts = urlsplit(self.base_url)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
            or re.search(r"[\s\"'<>`\\]", self.base_url)
            or any(part in {".", ".."} for part in parts.path.split("/"))
            or "//" in parts.path
        ):
            raise ValueError("--base-url must be an HTTPS site URL without query or fragment.")
        path = parts.path.rstrip("/") + "/"
        object.__setattr__(self, "base_url", urlunsplit(("https", parts.netloc, path, "", "")))

    @property
    def base_path(self) -> str:
        return urlsplit(self.base_url).path

    def url(self, value: str) -> str:
        """Rebase this site's URLs; preserve external URLs, fragments and relative paths."""
        parts = urlsplit(value)
        if parts.netloc and parts.hostname in {"linguoai.de", "www.linguoai.de"}:
            target = urlsplit(self.base_url)
            return urlunsplit(
                (
                    target.scheme,
                    target.netloc,
                    self.base_path + parts.path.lstrip("/"),
                    parts.query,
                    parts.fragment,
                )
            )
        if value.startswith("/") and not value.startswith("//"):
            return self.base_path + value.lstrip("/")
        return value

    def absolute_links(self, source: str) -> str:
        return OLD_ORIGIN.sub(self.base_url.rstrip("/"), source)

    def script(self, source: str) -> str:
        source = self.absolute_links(source)
        route_roots = {*LOCALES, *PUBLIC_PAGES, "assets", "downloads", "api", "site.webmanifest"}

        def rewrite(match: re.Match[str]) -> str:
            path = match["path"]
            first = re.split(r"[/#?]", path[1:], maxsplit=1)[0]
            # Only site routes, including the language template /${selected}/... .
            # Unrelated JavaScript strings and regular expressions stay unchanged.
            if not first or first in route_roots or first.startswith("${"):
                return match["quote"] + self.url(path) + match["quote"]
            return match[0]

        return JS_PATH.sub(rewrite, source)

    def css(self, source: str) -> str:
        return CSS_URL.sub(
            lambda match: f"url({match['quote']}{self.url(match['url'])}{match['quote']})",
            source,
        )

    def html(self, source: str) -> str:
        document = BeautifulSoup(source, "html.parser")
        for tag in document.find_all(True):
            for attribute in ("href", "src", "action", "poster", "xlink:href"):
                if isinstance(tag.get(attribute), str):
                    tag[attribute] = self.url(tag[attribute])
            if tag.name == "meta" and isinstance(tag.get("content"), str):
                tag["content"] = self.url(tag["content"])
            if tag.get("style"):
                tag["style"] = self.css(tag["style"])
            if tag.get("srcset") and not tag["srcset"].lstrip().startswith("data:"):
                candidates = []
                for item in tag["srcset"].split(","):
                    url, *descriptor = item.strip().split()
                    candidates.append(" ".join([self.url(url), *descriptor]))
                tag["srcset"] = ", ".join(candidates)
            if tag.name == "script" and not tag.get("src") and tag.string:
                tag.string.replace_with(self.script(str(tag.string)))
            if tag.name == "style" and tag.string:
                tag.string.replace_with(self.css(str(tag.string)))
        return str(document)

    def json_value(self, value: object) -> object:
        if isinstance(value, str):
            return self.url(value)
        if isinstance(value, list):
            return [self.json_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self.json_value(item) for key, item in value.items()}
        return value


def public_files(source: Path) -> list[Path]:
    """Explicit allowlist: never publish the checkout, tests, settings or credentials."""
    names = [*PUBLIC_PAGES, "site.webmanifest", "robots.txt", "sitemap.xml"]
    names.extend(f"{locale}/{page}" for locale in LOCALES[1:] for page in PRODUCT_PAGES)
    names.extend(f"assets/{name}" for name in PUBLIC_ASSETS)
    names.extend(f"assets/locales/{locale}.json" for locale in LOCALES)
    names.extend(
        f"downloads/{path.name}"
        for path in sorted((source / "downloads").iterdir())
        if path.is_file() and DOWNLOAD_NAME.fullmatch(path.name)
    )
    files = [source / name for name in names]
    for path in files:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError(f"Missing or unsafe public file: {path}")
    return files


def build_site(source: Path, output: Path, base_url: str = DEFAULT_BASE_URL) -> list[Path]:
    source = source.resolve()
    output = output.resolve()
    urls = SiteURLs(base_url)
    if source == output or source.is_relative_to(output):
        raise ValueError("The output directory must not contain the source directory.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new or empty output directory; existing files are never deleted.")
    files = public_files(source)
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for path in files:
        relative = path.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if relative.parts[0] == "downloads":
            # Keep archive contents, notebooks and SHA-256 manifests byte-identical.
            shutil.copyfile(path, destination)
        elif path.suffix in {".html", ".css", ".js", ".xml", ".webmanifest", ".txt"}:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".html":
                text = urls.html(text)
            elif path.suffix == ".css":
                text = urls.css(text)
            elif path.suffix == ".js":
                text = urls.script(text)
            elif path.suffix == ".webmanifest":
                text = (
                    json.dumps(urls.json_value(json.loads(text)), ensure_ascii=False, indent=2)
                    + "\n"
                )
            else:
                text = urls.absolute_links(text)
            destination.write_text(text, encoding="utf-8", newline="\n")
        else:
            shutil.copyfile(path, destination)
        written.append(destination)
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_source = ROOT / "website" if (ROOT / "website" / "index.html").is_file() else ROOT
    parser.add_argument("--source-dir", type=Path, default=default_source)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "_site")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    args = parser.parse_args()
    files = build_site(args.source_dir, args.output_dir, args.base_url)
    print(f"Built {len(files)} public files for {SiteURLs(args.base_url).base_url}")


if __name__ == "__main__":
    main()
