# main.py
import time
from crawler.sitemap_loader import load_sitemap
from crawler.selenium_loader import SeleniumLoader
from qa.chunker import chunk_text
from qa.prompt import build_prompt
from llm.ollama_llm import OllamaLLM


def run():
    print("=" * 70)
    print("Website QA Bot — Startup")
    print("=" * 70)

    sitemap_url = "https://www.maxfitai.com/sitemap.xml"

    # --------------------------------------------------
    # STEP 1: Load sitemap
    # --------------------------------------------------
    print("\n[STEP 1] Loading sitemap...")
    start = time.time()
    urls = load_sitemap(sitemap_url)
    print(f"[OK] Sitemap loaded — {len(urls)} URLs found "
          f"({time.time() - start:.2f}s)")

    # LIMIT FOR TESTING (IMPORTANT)
    MAX_PAGES = 20
    urls = urls[:MAX_PAGES]
    print(f"[INFO] Crawling limited to first {MAX_PAGES} pages\n")

    # --------------------------------------------------
    # STEP 2: Initialize Selenium (ONCE)
    # --------------------------------------------------
    print("[STEP 2] Initializing Selenium driver...")
    loader = SeleniumLoader(page_load_timeout=25)
    print("[OK] Selenium driver started")

    all_chunks = []

    # --------------------------------------------------
    # STEP 3: Crawl pages
    # --------------------------------------------------
    print("\n[STEP 3] Crawling website pages...\n")

    for idx, url in enumerate(urls, start=1):
        print(f"→ ({idx}/{len(urls)}) Fetching: {url}")

        try:
            text = loader.load_page_text(url)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)

            print(f"  ✓ Success — {len(chunks)} chunks")

        except Exception as e:
            print(f"  ✗ Skipped — {e}")
            continue

    # --------------------------------------------------
    # STEP 4: Shutdown Selenium
    # --------------------------------------------------
    print("\n[STEP 4] Closing Selenium driver...")
    loader.close()
    print("[OK] Selenium closed cleanly")

    print(f"\n[INFO] Total text chunks collected: {len(all_chunks)}")

    if not all_chunks:
        print("[FATAL] No content collected. Exiting.")
        return

    # --------------------------------------------------
    # STEP 5: Prepare LLM context
    # --------------------------------------------------
    print("\n[STEP 5] Preparing context for LLM...")

    # Use more chunks but keep reasonable size
    MAX_CHUNKS = 5  # Increased from 2
    context = "\n\n".join(all_chunks[:MAX_CHUNKS])

    word_count = len(context.split())
    print(f"[OK] Context prepared ({word_count} words from {min(MAX_CHUNKS, len(all_chunks))} chunks)")

    if word_count > 3000:
        print(f"[WARNING] Large context ({word_count} words). LLM may be slow.")

    # --------------------------------------------------
    # STEP 6: User question
    # --------------------------------------------------
    print("\n[STEP 6] Awaiting user question...")
    question = input("Ask a question: ").strip()

    if not question:
        print("[ERROR] No question provided. Exiting.")
        return

    prompt = build_prompt(context, question)
    print("[OK] Prompt built")

    # --------------------------------------------------
    # STEP 7: LLM inference (Ollama)
    # --------------------------------------------------
    print("\n[STEP 7] Querying LLM (Ollama)...")
    print("[TIP] This may take 30-120 seconds. Press Ctrl+C to cancel if needed.\n")

    llm = OllamaLLM(model="qwen2.5:3b")

    start = time.time()
    try:
        answer = llm.generate(prompt, timeout=120)  # Increased timeout
        elapsed = time.time() - start
        print(f"\n[OK] LLM responded in {elapsed:.2f}s")
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] User interrupted LLM generation.")
        return
    except Exception as e:
        print(f"\n[ERROR] LLM generation failed: {e}")
        return

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(answer)
    print("=" * 70)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Program interrupted by user.")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback

        traceback.print_exc()