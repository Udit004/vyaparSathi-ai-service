"""Temporary Firecrawl smoke test for the VyaparSathi AI service."""
import asyncio
import importlib.util
import os
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Load app/lib/firecrawl.py in isolation so the smoke test does not depend on
# the rest of the package __init__ chain (motor/pymongo).
app_pkg = types.ModuleType("app")
app_pkg.__path__ = [str(ROOT / "app")]
config_pkg = types.ModuleType("app.config")
config_pkg.__path__ = [str(ROOT / "app" / "config")]
lib_pkg = types.ModuleType("app.lib")
lib_pkg.__path__ = [str(ROOT / "app" / "lib")]
sys.modules["app"] = app_pkg
sys.modules["app.config"] = config_pkg
sys.modules["app.lib"] = lib_pkg

settings_spec = importlib.util.spec_from_file_location(
    "app.config.settings", ROOT / "app" / "config" / "settings.py"
)
settings_mod = importlib.util.module_from_spec(settings_spec)
sys.modules["app.config.settings"] = settings_mod
settings_spec.loader.exec_module(settings_mod)
config_pkg.settings = settings_mod

spec = importlib.util.spec_from_file_location("app.lib.firecrawl", ROOT / "app" / "lib" / "firecrawl.py")
firecrawl_mod = importlib.util.module_from_spec(spec)
sys.modules["app.lib.firecrawl"] = firecrawl_mod
spec.loader.exec_module(firecrawl_mod)


async def main() -> None:
    print("configured:", firecrawl_mod.is_firecrawl_configured())
    print("api_url:", os.getenv("FIRECRAWL_API_URL", "https://api.firecrawl.dev"))

    search = await firecrawl_mod.search_web("VyaparSathi inventory management", limit=2)
    print("search success:", search.get("success"), "| results:", len(search.get("results") or []))
    for item in (search.get("results") or [])[:2]:
        print("  -", item.get("title"), "|", item.get("url"))

    doc = await firecrawl_mod.scrape_url("https://example.com")
    markdown = doc.get("markdown") or ""
    print("scrape success:", doc.get("success"), "| markdown chars:", len(markdown))
    print("markdown head:", markdown[:120].replace("\n", " "))


asyncio.run(main())
