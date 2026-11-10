import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from bs4 import BeautifulSoup

from scripts.build_github_pages import (
    DEFAULT_BASE_URL,
    LOCALES,
    PRODUCT_PAGES,
    SiteURLs,
    build_site,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "website" if (ROOT / "website" / "index.html").is_file() else ROOT
NODE = shutil.which("node")
COLAB_URL = (
    "https://colab.research.google.com/github/riemenschneider2015-svg/linguoai-de/"
    "blob/main/notebooks/LinguoAI_Colab.ipynb"
)


@unittest.skipUnless((SOURCE / "index.html").is_file(), "Public site is absent from source archive")
class GitHubPagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="linguoai-pages-test-")
        cls.output = Path(cls.temporary.name) / "site"
        cls.source_hash = hashlib.sha256((SOURCE / "index.html").read_bytes()).digest()
        build_site(SOURCE, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def assert_local_url_exists(self, value, page_url):
        parsed = urlsplit(urljoin(page_url, value))
        target = urlsplit(DEFAULT_BASE_URL)
        if parsed.scheme not in {"http", "https"} or parsed.netloc != target.netloc:
            return
        self.assertTrue(parsed.path.startswith(target.path), value)
        relative = unquote(parsed.path[len(target.path) :])
        path = self.output / relative
        if parsed.path.endswith("/"):
            path /= "index.html"
        self.assertTrue(path.is_file(), f"{page_url}: {value} resolves to missing {path}")

    def test_all_languages_keep_working_links_assets_and_canonicals(self):
        for locale in LOCALES:
            for name in PRODUCT_PAGES:
                relative = Path(name) if locale == "de" else Path(locale) / name
                with self.subTest(page=relative.as_posix()):
                    page_url = urljoin(DEFAULT_BASE_URL, relative.as_posix())
                    document = BeautifulSoup(
                        (self.output / relative).read_text("utf-8"), "html.parser"
                    )
                    for tag in document.find_all(True):
                        for attribute in ("href", "src", "poster"):
                            if isinstance(tag.get(attribute), str):
                                self.assert_local_url_exists(tag[attribute], page_url)
                    canonical = document.select_one('link[rel="canonical"]')["href"]
                    expected = page_url.removesuffix("index.html")
                    self.assertEqual(canonical, expected)
                    self.assertEqual(len(document.select("[data-locale-link]")), len(LOCALES))
                    for link in document.select("[data-colab-start]"):
                        self.assertEqual(link["href"], COLAB_URL)

    def test_styles_manifest_sitemap_and_downloads_use_the_published_base(self):
        css = (self.output / "assets/editorial.css").read_text("utf-8")
        self.assertIn("/linguoai-de/assets/fonts/Manrope-Variable.woff2", css)
        for path in (self.output / "assets").glob("*.css"):
            for match in re.finditer(r"url\(['\"]?([^)'\"]+)", path.read_text("utf-8")):
                self.assert_local_url_exists(
                    match[1], urljoin(DEFAULT_BASE_URL, "assets/" + path.name)
                )
        manifest = json.loads((self.output / "site.webmanifest").read_text("utf-8"))
        self.assertEqual(urljoin(DEFAULT_BASE_URL, manifest["start_url"]), DEFAULT_BASE_URL)
        for icon in manifest["icons"]:
            self.assert_local_url_exists(icon["src"], DEFAULT_BASE_URL + "site.webmanifest")
        self.assertIn(
            DEFAULT_BASE_URL + "sitemap.xml", (self.output / "robots.txt").read_text("utf-8")
        )
        sitemap = (self.output / "sitemap.xml").read_text("utf-8")
        self.assertNotIn("https://linguoai.de", sitemap)
        for value in re.findall(r"https://[^\s<>\"]+", sitemap):
            if value.startswith(DEFAULT_BASE_URL):
                self.assert_local_url_exists(value, DEFAULT_BASE_URL)
        for path in (self.output / "downloads").iterdir():
            self.assertEqual(path.read_bytes(), (SOURCE / "downloads" / path.name).read_bytes())

    def test_output_contains_only_public_files_and_source_is_unchanged(self):
        self.assertTrue((self.output / ".nojekyll").is_file())
        for name in (
            "CNAME",
            ".git",
            ".github",
            ".env",
            "scripts",
            "tests",
            "linguoai",
            "deploy",
            "pyproject.toml",
            "README.md",
        ):
            self.assertFalse((self.output / name).exists(), name)
        self.assertEqual(
            hashlib.sha256((SOURCE / "index.html").read_bytes()).digest(), self.source_hash
        )
        for path in self.output.rglob("*"):
            if path.is_file() and "downloads" not in path.relative_to(self.output).parts:
                self.assertNotEqual(path.suffix, ".py")
                if path.suffix in {".html", ".js", ".css", ".xml", ".webmanifest"}:
                    self.assertNotIn("https://linguoai.de/", path.read_text("utf-8"), str(path))

    def test_custom_domain_build_and_existing_output_protection(self):
        custom = Path(self.temporary.name) / "custom"
        build_site(SOURCE, custom, "https://example.org")
        document = BeautifulSoup((custom / "index.html").read_text("utf-8"), "html.parser")
        self.assertEqual(
            document.select_one('link[rel="canonical"]')["href"], "https://example.org/"
        )
        self.assertEqual(
            document.select_one('script[src*="site.js"]')["src"].split("?")[0], "/assets/site.js"
        )
        with self.assertRaisesRegex(ValueError, "new or empty"):
            build_site(SOURCE, self.output)
        with self.assertRaises(ValueError):
            build_site(SOURCE, SOURCE)

    @unittest.skipUnless(NODE, "Node.js is needed to execute the generated browser scripts")
    def test_browser_redirects_explicit_language_and_studio_api_resolve_under_base(self):
        redirects = {}
        for name in PRODUCT_PAGES:
            document = BeautifulSoup((self.output / name).read_text("utf-8"), "html.parser")
            redirects[name] = next(
                str(tag.string)
                for tag in document.find_all("script")
                if not tag.get("src") and tag.string and "localeDetect" in tag.string
            )
        payload = {
            "base": DEFAULT_BASE_URL,
            "redirects": redirects,
            "locale": (self.output / "assets/locale.js").read_text("utf-8"),
            "site": (self.output / "assets/site.js").read_text("utf-8"),
            "studio": (self.output / "assets/web-app.js").read_text("utf-8"),
        }
        program = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
for (const script of [input.locale, input.site, input.studio, ...Object.values(input.redirects)]) {
  new vm.Script(script);
}
for (const [page, script] of Object.entries(input.redirects)) {
  for (const language of ['en', 'ru', 'uk', 'tr', 'es', 'it', 'fr', 'pt', 'pl', 'nl']) {
    let redirected;
    const context = {
      document: {documentElement: {
        dataset: {localeDetect: 'true'}, classList: {replace() {}}, setAttribute() {},
      }},
      window: {
        localStorage: {getItem() {return null;}},
        location: {search: '', hash: '#details', replace(url) {redirected = url;}},
      },
      navigator: {languages: [language]}, URLSearchParams,
    };
    vm.runInNewContext(script, context);
    const suffix = page === 'index.html' ? '' : page;
    assert.equal(
      new URL(redirected, input.base).href, `${input.base}${language}/${suffix}#details`,
    );
  }
}
const rootLink = {href: `${input.base}anleitung.html`, getAttribute() {return this.href;}};
const languageLink = {
  href: `${input.base}ru/colab.html`, dataset: {localeLink: 'ru'}, addEventListener() {},
};
const context = {
  document: {
    documentElement: {dataset: {localePage: 'de'}}, addEventListener() {},
    querySelectorAll(selector) {
      if (selector.startsWith('a[href]')) return [rootLink];
      return selector === '[data-locale-link]' ? [languageLink] : [];
    },
  },
  window: {
    location: {
      href: `${input.base}colab.html?lang=de`, origin: new URL(input.base).origin,
      search: '?lang=de', hash: '#help-title',
    },
    localStorage: {setItem() {}}, addEventListener() {},
  },
  URL, URLSearchParams,
};
vm.runInNewContext(input.locale, context);
assert.equal(new URL(rootLink.href).searchParams.get('lang'), 'de');
assert.equal(languageLink.href, `${input.base}ru/colab.html?lang=ru#help-title`);
let apiURL;
const stop = {};
class ProbeURL extends URL {
  constructor(value, base) {
    super(value, base);
    if (value === 'api/') {apiURL = this.href; throw stop;}
  }
}
try {
  vm.runInNewContext(input.studio, {
    document: {}, window: {location: {href: `${input.base}web.html`}}, URL: ProbeURL,
  });
}
catch (error) {if (error !== stop) throw error;}
assert.equal(apiURL, `${input.base}api/`);
console.log('Browser language routes, fragment preservation and Studio API base passed');
"""
        result = subprocess.run(
            [NODE, "-e", program],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class SiteURLsTests(unittest.TestCase):
    def test_url_queries_external_links_and_template_routes(self):
        urls = SiteURLs(DEFAULT_BASE_URL)
        self.assertEqual(
            urls.url("/colab.html?lang=de#help-title"), "/linguoai-de/colab.html?lang=de#help-title"
        )
        self.assertEqual(
            urls.url("https://linguoai.de/en/#versionen"), DEFAULT_BASE_URL + "en/#versionen"
        )
        self.assertEqual(
            urls.url("https://studio.linguoai.de/api/"), "https://studio.linguoai.de/api/"
        )
        self.assertEqual(urls.url("//example.org/image.png"), "//example.org/image.png")
        self.assertEqual(urls.url("#details"), "#details")
        self.assertEqual(
            urls.script("replace(`/${selected}/anleitung.html${window.location.hash}`)"),
            "replace(`/linguoai-de/${selected}/anleitung.html${window.location.hash}`)",
        )

    def test_rejects_unsafe_or_ambiguous_base_urls(self):
        for url in (
            "http://example.org",
            "https://name:secret@example.org/",
            "https://example.org/?q=1",
            "https://example.org/#x",
            "https://example.org/a/../",
            "https://example.org/a//b",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                SiteURLs(url)


if __name__ == "__main__":
    unittest.main()
