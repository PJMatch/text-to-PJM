import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ANIMATIONS_DIR = BASE_DIR.parent / "UEProject_5_5" / "Content" / "PjmAnimations_O"
OUTPUT_FILE = BASE_DIR / "available_animations.json"

EXTRA_ANIMATIONS = {
    "ONA",
    "ONO",
    "KOLEZANKA",
    "OK",
    "DZISIAJ",
}

def main():
    animations = {
        file.stem.upper()
        for file in ANIMATIONS_DIR.glob("*.uasset")
    }

    animations.update(EXTRA_ANIMATIONS)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(sorted(animations), f, ensure_ascii=False, indent=2)

    print(f"Saved {len(animations)} animations to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()