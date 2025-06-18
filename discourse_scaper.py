import os, json, asyncio, requests
from datetime import datetime
from io import BytesIO
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
from playwright.async_api import async_playwright

# === CONFIG ===
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_PATH = "/c/courses/tds-kb/34"
CATEGORY_JSON = f"{BASE_URL}{CATEGORY_PATH}.json"
AUTH_STATE_FILE = "auth.json"
DATE_FROM = datetime(2025, 1, 1)
DATE_TO = datetime(2025, 4, 14)
MAX_CONCURRENT_TASKS = 4

# If tesseract is not in PATH, set it explicitly (adjust if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def parse_date(date_str):
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unknown date format: {date_str}")

def ocr_from_url(url):
    try:
        resp = requests.get(url, timeout=10)
        img = Image.open(BytesIO(resp.content))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"🛑 OCR failed on {url}: {e}")
        return ""

async def fetch_and_parse_topic(playwright, topic):
    result = {
        "topic_id": topic["id"],
        "topic_title": topic.get("title"),
        "main_excerpt": None,
        "replies": []
    }

    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context(storage_state=AUTH_STATE_FILE)
    page = await context.new_page()

    try:
        url = f"{BASE_URL}/t/{topic['slug']}/{topic['id']}.json"
        await page.goto(url, timeout=10000)
        content = await page.inner_text("pre")
        data = json.loads(content)
        posts = data.get("post_stream", {}).get("posts", [])

        for post in posts:
            created = parse_date(post["created_at"])
            if not (DATE_FROM <= created <= DATE_TO):
                continue

            soup = BeautifulSoup(post["cooked"], "html.parser")
            content_text = soup.get_text()
            image_urls = [img["src"] for img in soup.find_all("img")]
            ocr_texts = [ocr_from_url(u) for u in image_urls]

            entry = {
                "post_id": post["id"],
                "author": post["username"],
                "created_at": post["created_at"],
                "content": content_text,
                "images": image_urls,
                "ocr_texts": ocr_texts,
                "post_number": post["post_number"],
                "topic_id": topic["id"],
                "slug": topic["slug"]
            }


            if post["post_number"] == 1:
                result["main_excerpt"] = entry
            else:
                result["replies"].append(entry)

    except Exception as e:
        result["error"] = str(e)
    finally:
        await page.close()
        await browser.close()

    return result

async def scrape_all():
    async with async_playwright() as p:
        # Login flow (only once)
        if not os.path.exists(AUTH_STATE_FILE):
            print("🔐 First-time login: Please login manually in browser...")
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(f"{BASE_URL}/login")
            await page.pause()
            await context.storage_state(path=AUTH_STATE_FILE)
            await browser.close()

        # Validate login
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=AUTH_STATE_FILE)
        page = await context.new_page()
        try:
            await page.goto(CATEGORY_JSON, timeout=10000)
            json.loads(await page.inner_text("pre"))
            print("✅ Authenticated")
        except:
            print("⚠️ Authentication invalid. Please rerun and login.")
            return
        await browser.close()

        # Get all topic metadata
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(storage_state=AUTH_STATE_FILE)
        page = await ctx.new_page()

        topics, page_num = [], 0
        while True:
            try:
                await page.goto(f"{CATEGORY_JSON}?page={page_num}", timeout=10000)
                data = json.loads(await page.inner_text("pre"))
                batch = data.get("topic_list", {}).get("topics", [])
                if not batch:
                    break
                topics.extend(batch)
                page_num += 1
            except Exception as e:
                print(f"⚠️ Failed to load page {page_num}: {e}")
                break
        await browser.close()

        filtered = [t for t in topics if DATE_FROM.date() <= parse_date(t["created_at"]).date() <= DATE_TO.date()]
        print(f"📄 {len(filtered)} topics to process...")

        # Limit concurrency
        sem = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

        async def safe_fetch(topic):
            async with sem:
                return await fetch_and_parse_topic(p, topic)

        results = await asyncio.gather(*[safe_fetch(t) for t in filtered])

        with open("tds_posts_with_ocr.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(results)} entries with OCR to 'tds_posts_with_ocr.json'")

if __name__ == "__main__":
    asyncio.run(scrape_all())
