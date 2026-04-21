from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import unicodedata

# importing the NLP engine function
from nlp_engine import process_polish_text

app = FastAPI(title="PJM Translator API")

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
    text = unicodedata.normalize('NFKD', text)
    # Remove diacritical marks (accents) to get base characters (normalize returns the base character and its accent)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # 'Ł' is a separate Unicode character (not a base + accent), so normalize manually
    text = text.replace("Ł", "L")
    return text

def map_gloss_to_animation(gloss_item: GlossItem, sentence_type: Optional[str] = None) -> AnimationItem:
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
    return {"message": "PJM Translator API works"}

# Main endpoint for translating text to animations
@app.post("/translate", response_model=List[AnimationItem])
def translate_text_to_animations(request: TextRequest):
    try:
        # Polish text to structured gloss data
        raw_gloss_data = process_polish_text(request.text)
        data = GlossRequest(**raw_gloss_data)

        # Mapping gloss items to animations
        animations: List[AnimationItem] = []
        for clause in data.clauses:
            for gloss_item in clause.pjm_sequence:
                
                # Special handling for fingerspelling
                if gloss_item.type == "fingerspell" and gloss_item.letters:
                    for letter in gloss_item.letters:
                        letter_upper = letter.upper()
                        # Map each letter to its corresponding animation
                        animations.append(AnimationItem(
                            gloss=letter_upper,
                            animation=remove_polish_chars(letter_upper),
                            type="fingerspell",
                            sentence_type=clause.sentence_type
                        ))
                else:
                    animations.append(map_gloss_to_animation(gloss_item, clause.sentence_type))

        return animations

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd tłumaczenia: {str(e)}")