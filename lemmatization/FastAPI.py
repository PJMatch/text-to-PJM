from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import unicodedata
from pathlib import Path

# importing the NLP engine function
from nlp_engine import process_polish_text

BASE_DIR = Path(__file__).resolve().parent
ANIMATIONS_DIR = BASE_DIR.parent / "UEProject_5_5" / "Content" / "PjmAnimations_O"

app = FastAPI(title="PJM Translator API")

def load_available_animations() -> set[str]:
    """ Loads available animation names from .uasset files in the animation directory. """
    if not ANIMATIONS_DIR.exists():
        return set()

    return {
        file.stem.upper()
        for file in ANIMATIONS_DIR.glob("*.uasset")
    }

AVAILABLE_ANIMATIONS = load_available_animations()

AVAILABLE_ANIMATIONS.add("ONA")
AVAILABLE_ANIMATIONS.add("ONO")
AVAILABLE_ANIMATIONS.add("KOLEZANKA")
AVAILABLE_ANIMATIONS.add("OK")
AVAILABLE_ANIMATIONS.add("DZISIAJ")

class GlossItem(BaseModel):
    type: str
    gloss: str
    letters: Optional[List[str]] = None
    tense: Optional[str] = None
    is_negated: Optional[bool] = None
    is_plural: Optional[bool] = None

class Clause(BaseModel):
    sentence_type: str
    pjm_sequence: List[GlossItem]

class GlossRequest(BaseModel):
    text: Optional[str] = None
    clauses: List[Clause] = Field(default_factory=list)

# Function which maps gloss items to animation items
class AnimationItem(BaseModel):
    gloss: str
    animation: str
    type: str
    sentence_type: Optional[str] = None
    tense: Optional[str] = None
    is_negated: Optional[bool] = None
    is_plural: Optional[bool] = None

class AnimationResponse(BaseModel):
    animations: List[AnimationItem]

# Input text from user
class TextRequest(BaseModel):
    text: str

def remove_polish_chars(text: str) -> str:
    """Removes Polish diacritical marks from text to match animation file names."""
    text = unicodedata.normalize('NFKD', text)
    # Remove diacritical marks (accents) to get base characters (normalize returns the base character and its accent)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # 'Ł' is a separate Unicode character (not a base + accent), so normalize manually
    text = text.replace("Ł", "L")
    return text

PJM_DIGRAPHS = {"CH", "CZ", "RZ", "SZ", "DZ"}

PJM_DIACRITICS = {
    "Ą": "AA",
    "Ć": "CC",
    "Ę": "EE",
    "Ł": "LL",
    "Ń": "NN",
    "Ó": "OO",
    "Ś": "SS",
    "Ż": "ZZ",
    "Ź": "ZZZ"
}

def get_fingerspell_tokens(word: str) -> List[str]:
    """Splits a word into PJM fingerspelling tokens (handles digraphs and diacritics)."""
    word = word.upper()
    tokens = []
    i = 0
    while i < len(word):
        # Check for 2-letter digraphs first
        if i + 1 < len(word) and word[i:i+2] in PJM_DIGRAPHS:
            tokens.append(word[i:i+2])
            i += 2
        else:
            char = word[i]
            # Convert diacritics (e.g., 'Ż' -> 'ZZ'), or keep the char if not in dict
            tokens.append(PJM_DIACRITICS.get(char, char))
            i += 1
    return tokens

def split_pjm_numbers(gloss: str) -> List[str]:
    """Splits numbers 21-59 into tens and units (e.g., '34' -> ['30', '4'])."""
    if gloss.isdigit():
        val = int(gloss)
        # Check if the number is between 21 and 59, and is not a round ten (30, 40, 50)
        if 20 < val < 60 and val % 10 != 0:
            return [str((val // 10) * 10), str(val % 10)]
    return [gloss]

def map_gloss_to_animation(gloss_item: GlossItem, sentence_type: Optional[str] = None) -> AnimationItem:
    """Maps a gloss item to an animation name by uppercasing it, removing Polish characters, and keeping its metadata."""
    gloss = gloss_item.gloss.upper()
    animation_name = remove_polish_chars(gloss)

    return AnimationItem(
        gloss=gloss,
        animation=animation_name,
        type=gloss_item.type,
        sentence_type=sentence_type,
        tense=gloss_item.tense,
        is_negated=gloss_item.is_negated,
        is_plural=gloss_item.is_plural
    )

@app.get("/")
def root():
    """Checks if the PJM Translator API is running."""
    return {"message": "PJM Translator API works"}

# Main endpoint for translating text to animations
@app.post("/translate", response_model=List[AnimationItem])
def translate_text_to_animations(request: TextRequest):
    """Translates Polish text into PJM animation items, including number splitting and fingerspelling fallback."""
    try:
        # Polish text to structured gloss data
        raw_gloss_data = process_polish_text(request.text)
        data = GlossRequest(**raw_gloss_data)

        # Mapping gloss items to animations
        animations: List[AnimationItem] = []
        for clause in data.clauses:
            for gloss_item in clause.pjm_sequence:
                
                # Special handling for fingerspelling
                if gloss_item.type == "fingerspell":
                    tokens = get_fingerspell_tokens(gloss_item.gloss)
                    for token in tokens:
                        animations.append(AnimationItem(
                            gloss=token,
                            animation=remove_polish_chars(token),
                            type="fingerspell",
                            sentence_type=clause.sentence_type
                        ))
                else:
                    animation_item = map_gloss_to_animation(gloss_item, clause.sentence_type)
                    
                    sub_glosses = split_pjm_numbers(gloss_item.gloss)
                    
                    for sg in sub_glosses:
                        # Create a temporary item for the split gloss
                        temp_gloss_item = GlossItem(
                            type=gloss_item.type,
                            gloss=sg,
                            tense=gloss_item.tense,
                            is_negated=gloss_item.is_negated,
                            is_plural=gloss_item.is_plural
                        )
                        
                        animation_item = map_gloss_to_animation(temp_gloss_item, clause.sentence_type)
                        
                        # If the animation is not available, fallback to fingerspelling each letter of the gloss
                        if animation_item.animation.upper() not in AVAILABLE_ANIMATIONS:
                            tokens = get_fingerspell_tokens(temp_gloss_item.gloss)
                            for token in tokens:
                                animations.append(AnimationItem(
                                    gloss=token,
                                    animation=remove_polish_chars(token),
                                    type="fingerspell",
                                    sentence_type=clause.sentence_type
                                ))
                        else:
                            animations.append(animation_item)
        return animations

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd tłumaczenia: {str(e)}")