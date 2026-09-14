import os
import shutil
import subprocess
import json
import requests
import time

# ── Word-limit presets ────────────────────────────────────────────────────────
WORD_LIMITS = {
    "short":  (50,  100),
    "medium": (150, 200),
    "long":   (250, 300),
}

# ── Suggested genres ──────────────────────────────────────────────────────────
SUGGESTED_GENRES = [
    "Fantasy",
    "Mystery",
    "Adventure",
    "Romance",
    "Horror",
    "Science Fiction",
    "Fairy Tale",
    "Comedy",
]


def ensure_ollama_running() -> bool:
    """
    Ensure Ollama backend / tray app is running.
    Checks port 11434 (default), and if inactive:
    1. Locates ollama executable (using which / common Windows installation directories).
    2. Runs 'ollama app.exe' (tray app) on Windows, or 'ollama serve' in background.
    3. Waits for the port to become responsive before returning.
    """
    url = "http://localhost:11434/api/tags"
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            print("[OK] Ollama is already running on port 11434.", flush=True)
            return True
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        pass

    print("[*] Ollama is not running on port 11434. Attempting to start Ollama...", flush=True)
    
    # Locate executable
    ollama_path = shutil.which("ollama")
    ollama_app_path = None
    
    if ollama_path:
        # Check if ollama app.exe is in the same directory (Windows Tray app)
        parent_dir = os.path.dirname(ollama_path)
        app_path = os.path.join(parent_dir, "ollama app.exe")
        if os.path.exists(app_path):
            ollama_app_path = app_path
            
    # Hardcoded default locations on Windows
    if os.name == 'nt':
        local_app_data = os.environ.get("LOCALAPPDATA")
        default_paths = []
        if local_app_data:
            default_paths.append(os.path.join(local_app_data, "Programs", "Ollama", "ollama app.exe"))
            default_paths.append(os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe"))
        default_paths.append(r"C:\Users\Fairy Tyagi\AppData\Local\Programs\Ollama\ollama app.exe")
        default_paths.append(r"C:\Users\Fairy Tyagi\AppData\Local\Programs\Ollama\ollama.exe")
        
        for dp in default_paths:
            if os.path.exists(dp):
                if "ollama app.exe" in dp and not ollama_app_path:
                    ollama_app_path = dp
                elif "ollama.exe" in dp and not ollama_path:
                    ollama_path = dp

    started = False
    try:
        if ollama_app_path:
            print(f"[*] Starting Ollama Tray App: {ollama_app_path}", flush=True)
            if os.name == 'nt':
                os.startfile(ollama_app_path)
                started = True
            else:
                subprocess.Popen([ollama_app_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                started = True
    except Exception as e:
        print(f"[!] Failed to startfile tray app: {e}", flush=True)

    if not started:
        try:
            cmd = ["ollama", "serve"]
            if ollama_path:
                cmd[0] = ollama_path
            
            print(f"[*] Starting Ollama CLI service: {' '.join(cmd)}", flush=True)
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
            subprocess.Popen(
                cmd,
                startupinfo=startupinfo,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True
            )
            started = True
        except Exception as e:
            print(f"[!] Failed to run ollama serve: {e}", flush=True)

    if started:
        print("[*] Waiting for Ollama (up to 20 seconds) to respond...", flush=True)
        for i in range(20):
            time.sleep(1)
            try:
                res = requests.get(url, timeout=1)
                if res.status_code == 200:
                    print(f"[OK] Ollama is active and responsive after {i+1}s.", flush=True)
                    return True
            except:
                pass
        print("[!] Ollama did not respond within timeout.", flush=True)
        return False
    
    print("[!] Ollama executable not found, cannot start automatically.", flush=True)
    return False


def clean_json_response(raw: str) -> str:
    """Strip markdown code blocks and whitespace from LLM response."""
    raw = raw.strip()
    # Remove markdown backticks if present
    if raw.startswith("```"):
        # Remove starting ```json or ```
        raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
        # Remove ending ```
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def generate_story_ollama(
    word1: str, 
    word2: str, 
    word3: str, 
    genre: str = "Fantasy",
    word_limit: str = "medium"
) -> dict:
    """
    Generate a story from three words using local Ollama (llama3:latest).
    Supports word limits and genre customization while maintaining 
    connection robustness.

    Returns a dictionary: {"story": "...", "vocabulary": {"word": "definition", ...}}
    """
    min_words, max_words = WORD_LIMITS.get(word_limit.lower(), (150, 200))

    prompt = (
        f"You are a creative fiction writer specialising in {genre} stories for children. "
        f"Write a {genre} story that is between {min_words} and {max_words} words long. "
        f"The story must naturally include the words: {word1}, {word2}, and {word3}. "
        f"It should have a clear beginning, middle, and end, and finish with a moral or lesson. "
        f"\n\nIMPORTANT: Respond ONLY with a valid JSON object in this format:\n"
        f"{{\n"
        f"  \"story\": \"full story text here\",\n"
        f"  \"vocabulary\": {{\n"
        f"    \"word1\": \"definition 1\",\n"
        f"    \"word2\": \"definition 2\"\n"
        f"  }}\n"
        f"}}\n"
        f"Identify 3 to 5 'interesting' words from the story that a child might want to learn. "
        f"Exclude simple function words like 'in', 'on', 'up', 'that', 'the', etc. "
        f"Provide simple, child-friendly definitions for these words."
    )

    try:
        url = "http://localhost:11434/api/generate"
        try_count = 0
        max_tries = 2
        
        while try_count < max_tries:
            try_count += 1
            try:
                print(f"[*] Sending prompt to Ollama (llama3:latest) (attempt {try_count}/{max_tries})...", flush=True)
                response = requests.post(
                    url,
                    json={
                        "model": "llama3:latest",
                        "prompt": prompt,
                        "stream": False,
                        "format": "json"  # Ensure JSON output if supported by weights
                    },
                    timeout=300 # Five-minute timeout for robust local generation
                )
                print(f"[*] Ollama response status: {response.status_code}", flush=True)
                response.raise_for_status()
                data = response.json()
                
                raw_response = data.get("response", "").strip()
                
                # Parse the JSON from the LLM, cleaning it first
                json_str = clean_json_response(raw_response)
                story_data = json.loads(json_str)
                
                print(f"[OK] Story generated: {len(story_data.get('story','').split())} words, {len(story_data.get('vocabulary', {}))} vocabulary items.", flush=True)
                return story_data
            except requests.exceptions.ConnectionError as ce:
                if try_count < max_tries:
                    print(f"[!] Connection to Ollama failed. Attempting to start Ollama and retry...", flush=True)
                    if ensure_ollama_running():
                        continue
                raise ce


            
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract just the story text
        print(f"[!] JSON parse failed, falling back to plain text extraction.", flush=True)
        if 'raw_response' in locals() and raw_response:
            return {"story": raw_response, "vocabulary": {}}
        return {
            "story": "Oh no! The magic book is a bit stuck. Please try again!",
            "vocabulary": {}
        }
    except requests.exceptions.ConnectTimeout:
        return {
            "story": "AI Error: Connection to Ollama timed out. Is the llama3:latest model still loading?",
            "vocabulary": {}
        }
    except requests.exceptions.ConnectionError:
        return {
            "story": "AI Error: Could not reach Ollama. Please make sure the Ollama application is running on your computer.",
            "vocabulary": {}
        }
    except requests.exceptions.HTTPError as e:
        return {
            "story": f"AI Error: Ollama service error ({e.response.status_code}).",
            "vocabulary": {}
        }
    except Exception as e:
        return {
            "story": f"AI Error: {str(e)}. Please check your Ollama console.",
            "vocabulary": {}
        }

if __name__ == "__main__":
    # Test with sample words
    w1, w2, w3 = "cat", "umbrella", "star"
    print(f"Testing Robust Ollama (llama3:latest) with: {w1}, {w2}, {w3}...", flush=True)
    
    story = generate_story_ollama(w1, w2, w3, genre="Sci-Fi", word_limit="short")
    print("\n--- Story Result ---")
    print(story)