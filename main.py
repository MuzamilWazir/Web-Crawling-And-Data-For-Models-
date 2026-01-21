# main.py
import time
import sys
from crawler.sitemap_loader import load_sitemap
from crawler.selenium_loader import SeleniumLoader
from qa.chunker import chunk_text
from qa.prompt import build_prompt


def select_llm():
    """Let user choose which LLM to use"""
    print("\n" + "=" * 70)
    print("SELECT LLM PROVIDER")
    print("=" * 70)
    print("1. Ollama (Local - Free)")
    print("2. OpenAI (GPT-4o-mini - Paid)")
    print("3. Claude (Anthropic - Paid)")
    print("4. Gemini (Google - Paid)")
    print("=" * 70)

    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        from llm.ollama_llm import OllamaLLM
        return OllamaLLM(model="qwen2.5:3b")

    elif choice == "2":
        from llm.openai_llm import OpenAILLM
        api_key = input("Enter OpenAI API Key (or press Enter to use env var): ").strip()
        return OpenAILLM(model="gpt-4o-mini", api_key=api_key or None)

    elif choice == "3":
        from llm.claude_llm import ClaudeLLM
        api_key = input("Enter Anthropic API Key (or press Enter to use env var): ").strip()
        return ClaudeLLM(model="claude-3-5-sonnet-20241022", api_key=api_key or None)

    elif choice == "4":
        from llm.gemini_llm import GeminiLLM
        api_key = input("Enter Google API Key (or press Enter to use env var): ").strip()
        return GeminiLLM(model="gemini-1.5-flash", api_key=api_key or None)

    else:
        print("[ERROR] Invalid choice. Exiting.")
        sys.exit(1)


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

    # LIMIT FOR TESTING
    MAX_PAGES = 20
    urls = urls[:MAX_PAGES]
    print(f"[INFO] Crawling limited to first {MAX_PAGES} pages\n")

    # --------------------------------------------------
    # STEP 2: Initialize Selenium
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
    # STEP 5: Prepare context
    # --------------------------------------------------
    print("\n[STEP 5] Preparing context for LLM...")

    MAX_CHUNKS = 5
    context = "\n\n".join(all_chunks[:MAX_CHUNKS])

    word_count = len(context.split())
    print(f"[OK] Context prepared ({word_count} words from {min(MAX_CHUNKS, len(all_chunks))} chunks)")

    # --------------------------------------------------
    # STEP 6: Select LLM
    # --------------------------------------------------
    try:
        llm = select_llm()
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM: {e}")
        return

    # --------------------------------------------------
    # STEP 7: User question
    # --------------------------------------------------
    print("\n[STEP 7] Awaiting user question...")
    question = input("Ask a question: ").strip()

    if not question:
        print("[ERROR] No question provided. Exiting.")
        return

    prompt = build_prompt(context, question)
    print("[OK] Prompt built")

    # --------------------------------------------------
    # STEP 8: LLM inference
    # --------------------------------------------------
    print("\n[STEP 8] Querying LLM...\n")

    start = time.time()
    try:
        answer = llm.generate(prompt)
        elapsed = time.time() - start
        print(f"\n[OK] Response received in {elapsed:.2f}s")
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] User interrupted.")
        return
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        import traceback
        traceback.print_exc()
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