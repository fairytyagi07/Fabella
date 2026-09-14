"""
app.py
Flask backend for Sketch-to-Story.
Serves the frontend and provides prediction + story generation APIs.
"""

import os
import io
import json
import base64
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from Story_A import generate_story_ollama, ensure_ollama_running
from Audio import create_tts_audio

# ── Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sketch_model.keras")
CATEGORIES_PATH = os.path.join(BASE_DIR, "model", "categories.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ── Flask App ───────────────────────────────────────────────────────
app = Flask(__name__, static_folder=STATIC_DIR)

# ── Load model and categories ──────────────────────────────────────
model = None
categories_raw = None  # Original dict with sets
categories_flat = None # Flattened list for prediction
curated_indices = []   # Indices of 'reliable' categories
CURATED_PATH = os.path.join(BASE_DIR, "model", "categories_curated.json")


def load_model():
    """Load the trained model and category list."""
    global model, categories_raw, categories_flat, curated_indices
    if not os.path.exists(MODEL_PATH):
        # We allow the server to start even if model is missing, 
        # so the user can see the UI while training.
        print(f"[!] Warning: Model not found at {MODEL_PATH}")
    else:
        # Patch for BatchNormalization axis list error ([3] -> 3)
        # This is a known issue when loading models across certain Keras 3 versions
        OriginalBN = tf.keras.layers.BatchNormalization
        original_from_config = OriginalBN.from_config

        @classmethod
        def patched_from_config(cls, config):
            if 'axis' in config and isinstance(config['axis'], list):
                config['axis'] = config['axis'][0]
            return original_from_config(config)

        # Apply the monkey-patch for BatchNormalization
        tf.keras.layers.BatchNormalization.from_config = patched_from_config

        # PATCH FOR DENSE (quantization_config mismatch)
        OriginalDense = tf.keras.layers.Dense
        original_dense_from_config = OriginalDense.from_config

        @classmethod
        def patched_dense_from_config(cls, config):
            # Remove quantization_config if it's there
            if 'quantization_config' in config:
                config.pop('quantization_config')
            return original_dense_from_config(config)

        # Apply the monkey-patch for Dense
        tf.keras.layers.Dense.from_config = patched_dense_from_config

        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[OK] Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"[!] Error loading model: {e}")
            # Final fallback
            try:
                model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
                print(f"[OK] Model loaded from {MODEL_PATH} (safe_mode=False)")
            except Exception as e2:
                print(f"[CRITICAL] Failed to load model: {e2}")

    if not os.path.exists(CATEGORIES_PATH):
        print(f"[!] Warning: Categories not found at {CATEGORIES_PATH}")
    else:
        with open(CATEGORIES_PATH, "r") as f:
            categories_raw = json.load(f)
        
        # Try loading the flat prediction-order list first
        original_345_path = os.path.join(BASE_DIR, "model", "trained_336.json")
        if os.path.exists(original_345_path):
            with open(original_345_path, "r") as f:
                categories_flat = json.load(f)
            print(f"[OK] 336 Model Prediction Categories loaded.")
        else:
            # Fallback: flatten from the pack-based dict
            if isinstance(categories_raw, dict):
                categories_flat = []
                for pack in ["1", "2", "3"]:
                    if pack in categories_raw:
                        categories_flat.extend(categories_raw[pack])
            elif isinstance(categories_raw, list):
                # categories.json is already a flat list
                categories_flat = categories_raw
        
        total = len(categories_flat) if categories_flat else 0
        print(f"[OK] Categories loaded: {total} total")

        # Load curated list and find their indices for accuracy filtering
        if os.path.exists(CURATED_PATH) and categories_flat:
            try:
                with open(CURATED_PATH, "r") as f:
                    curated_list = json.load(f)
                
                # Normalize and find indices
                cat_lower = [c.lower() for c in categories_flat]
                for item in curated_list:
                    item_lower = item.lower()
                    if item_lower in cat_lower:
                        idx = cat_lower.index(item_lower)
                        curated_indices.append(idx)
                
                print(f"[OK] {len(curated_indices)} curated category indices identified for accuracy.")
            except Exception as e:
                print(f"[!] Error loading curated categories: {e}")


def preprocess_image(image_data_base64: str) -> np.ndarray:
    """
    Convert a base64-encoded image to a 28x28 grayscale numpy array.
    Robustly crops, centers, and post-processes the sketch to match
    the Quick, Draw! training data format.
    
    Pipeline:
    1. Decode base64 -> PIL Image
    2. Convert to grayscale
    3. Crop to bounding box + center in square with padding
    4. Resize to 28x28
    5. Adaptive post-processing: thin strokes get dilated, all get mild smoothing
    6. Normalize to [0, 1] and reshape
    """
    from scipy.ndimage import maximum_filter, gaussian_filter
    
    # 1. Decode base64
    if "," in image_data_base64:
        image_data_base64 = image_data_base64.split(",")[1]
    image_bytes = base64.b64decode(image_data_base64)
    image = Image.open(io.BytesIO(image_bytes))

    # 2. Convert to grayscale
    image = image.convert("L")

    # 3. Find bounding box of the sketch (non-black pixels)
    # The canvas sends inverted images: white (255) strokes on black (0) bg.
    bbox = image.getbbox()
    
    if bbox:
        # Crop to the bounding box
        image = image.crop(bbox)
        
        # Make it square by adding padding
        w, h = image.size
        max_dim = max(w, h)
        
        # Use 20% padding for better alignment with training data
        padding = int(max_dim * 0.2)
        new_size = max_dim + 2 * padding
        
        square_image = Image.new("L", (new_size, new_size), 0)
        paste_x = (new_size - w) // 2
        paste_y = (new_size - h) // 2
        square_image.paste(image, (paste_x, paste_y))
        image = square_image

    # 4. Resize to 28x28
    # Using LANCZOS filter for high-quality downsampling
    image = image.resize((28, 28), Image.LANCZOS)

    # 5. Adaptive post-processing
    img_array = np.array(image).astype("float32")
    
    # Check stroke density (ratio of non-zero pixels to total)
    # Thin/pencil strokes have low density; thick marker strokes have high density
    non_zero_ratio = np.count_nonzero(img_array) / img_array.size
    
    if non_zero_ratio < 0.15:
        # Sparse/thin strokes — apply dilation to make them visible
        img_array = maximum_filter(img_array, size=2)
    
    # Mild Gaussian blur to smooth jagged downsampled edges
    img_array = gaussian_filter(img_array, sigma=0.4)
    
    # Normalize brightness: scale so max pixel = 255
    pmax = img_array.max()
    if pmax > 0:
        img_array = img_array * (255.0 / pmax)

    # 6. Normalize to [0, 1] and reshape for model input: (1, 28, 28, 1)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main page."""
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/categories")
def get_categories():
    """Return the categorized category list."""
    if not categories_raw:
        return jsonify({"error": "Categories not loaded"}), 500
    # If categories_raw is a flat list, partition into 3 packs for API compat
    if isinstance(categories_raw, list):
        l1 = len(categories_raw) // 3
        l2 = l1 * 2
        categories_dict = {
            "1": categories_raw[:l1],
            "2": categories_raw[l1:l2],
            "3": categories_raw[l2:]
        }
        return jsonify(categories_dict)
    return jsonify(categories_raw)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept three base64-encoded sketch images and return predictions.
    
    Expected JSON body:
    {
        "sketches": ["base64...", "base64...", "base64..."]
    }
    """
    data = request.get_json()

    if not data or "sketches" not in data:
        return jsonify({"error": "Missing 'sketches' in request body"}), 400

    sketches = data["sketches"]
    print(f"[*] Received /predict request with {len(sketches)} sketches.", flush=True)

    if len(sketches) != 3:
        return jsonify({"error": "Exactly 3 sketches required"}), 400

    predictions = []
    for i, sketch_b64 in enumerate(sketches):
        # Skip null sketches (for real-time individual updates)
        if not sketch_b64:
            predictions.append(None)
            continue
            
        try:
            print(f"[*] Preprocessing sketch {i+1}...", flush=True)
            img_array = preprocess_image(sketch_b64)
            print(f"[*] Predicting sketch {i+1}...", flush=True)
            full_pred = model.predict(img_array, verbose=0)[0]
            
            # Focus ONLY on curated categories if available for much higher accuracy
            if curated_indices:
                filtered_pred = np.zeros_like(full_pred)
                for idx in curated_indices:
                    filtered_pred[idx] = full_pred[idx]
                pred = filtered_pred
            else:
                pred = full_pred

            # Get top 3 predictions
            top3_indices = np.argsort(pred)[-3:][::-1]
            top3 = [
                {
                    "label": categories_flat[int(idx)],
                    "confidence": float(pred[int(idx)])
                }
                for idx in top3_indices if pred[idx] > 0
            ]
            
            # Handle case where no curated categories match well
            if not top3:
                top_idx = np.argmax(full_pred)
                top3 = [{"label": categories_flat[int(top_idx)], "confidence": float(full_pred[int(top_idx)])}]
            
            print(f"\n[AI] Sketch {i+1} Top Predictions:", flush=True)
            for p in top3:
                print(f"  > {p['label']}: {p['confidence']*100:.1f}%", flush=True)
            print("-" * 30, flush=True)

            predictions.append({
                "label": top3[0]["label"],
                "confidence": round(top3[0]["confidence"], 4),
                "top3": top3,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            predictions.append({
                "label": "unknown",
                "confidence": 0.0,
                "error": str(e),
            })

    print(f"[OK] Returned {len(predictions)} predictions.", flush=True)
    return jsonify({"predictions": predictions})


@app.route("/generate-story", methods=["POST"])
def generate_story_route():
    """
    Generate a story from three recognized objects.
    
    Expected JSON body:
    {
        "objects": ["cat", "house", "tree"],
        "genre": "adventure"  // optional
    }
    """
    data = request.get_json()

    if not data or "objects" not in data:
        return jsonify({"error": "Missing 'objects' in request body"}), 400

    objects = data["objects"]
    if len(objects) != 3:
        return jsonify({"error": "Exactly 3 objects required"}), 400

    word_limit = data.get("word_limit", "medium")
    genre = data.get("genre", "Fantasy")

    print(f"[*] Generating story for objects: {objects}, genre: {genre}, length: {word_limit}", flush=True)
    try:
        story_data = generate_story_ollama(
            objects[0], objects[1], objects[2],
            word_limit=word_limit,
            genre=genre
        )
        print(f"[OK] Story generation completed.", flush=True)
        return jsonify({
            "story": story_data.get("story", ""),
            "vocabulary": story_data.get("vocabulary", {}),
            "genre": genre,
            "is_ai": True
        })
    except Exception as e:
        print(f"[ERROR] Story generation failed: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

@app.route("/speak", methods=["POST"])
def speak_route():
    """
    Generate an offline voice buffer using system TTS.
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    audio_url = create_tts_audio(data["text"], output_dir=os.path.join(STATIC_DIR, "audio"))
    if audio_url:
        return jsonify({"audio_url": audio_url})
    
    return jsonify({"error": "Failed to generate audio"}), 500




# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    
    # Start Ollama service in a background daemon thread
    import threading
    ollama_thread = threading.Thread(target=ensure_ollama_running)
    ollama_thread.daemon = True
    ollama_thread.start()
    
    # Check for Server-Only mode (e.g. for mobile testing or remote access)
    if os.environ.get("FLASK_SERVER_ONLY"):
        print("\n[*] Sketch-to-Story booting in Server-Only mode (port 5001)...\n")
        app.run(debug=False, host="0.0.0.0", port=5001)
    else:
        try:
            import webview
            print("\n[*] Sketch-to-Story booting Native Desktop App...\n")
            webview.create_window('Sketch-to-Story', app, width=1280, height=800)
            webview.start(private_mode=False)
        except (ImportError, Exception) as e:
            print(f"\n[!] Webview launch failed: {e}")
            print("[*] Falling back to Browser Server mode (port 5001)...\n")
            app.run(debug=False, host="0.0.0.0", port=5001)