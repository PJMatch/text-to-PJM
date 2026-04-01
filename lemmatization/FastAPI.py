from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

# importing the NLP engine function
from nlp_engine import process_polish_text

app = FastAPI(title="PJM Translator API")

class GlossItem(BaseModel):
    type: str
    gloss: str
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
    tense: Optional[str] = None
    is_negated: Optional[bool] = None
    is_plural: Optional[bool] = None

class AnimationResponse(BaseModel):
    animations: List[AnimationItem]

# Input text from user
class TextRequest(BaseModel):
    text: str

# NA RAZIE ZAMIENIAMY RECZNIE TO JEST DO ZMIANY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ANIMATION_MAP = {
    "DRZWI": "DRZWI",
    "BATERIA": "BATERIA",
    "JA": "JA",
    "TY": "TY",
    "MÓWIĆ": "MOWIC",
    "ROZUMIEĆ": "ROZUMIEC",
    "NIE": "NIE"
}

def map_gloss_to_animation(gloss_item: GlossItem) -> AnimationItem:
    gloss = gloss_item.gloss.upper()
    animation_name = ANIMATION_MAP.get(gloss, gloss)

    return AnimationItem(
        gloss=gloss,
        animation=animation_name,
        type=gloss_item.type,
        tense=gloss_item.tense,
        is_negated=gloss_item.is_negated,
        is_plural=gloss_item.is_plural
    )

@app.get("/")
def root():
    return {"message": "PJM Translator API works"}

# Main endpoint for translating text to animations
@app.post("/translate", response_model=AnimationResponse)
def translate_text_to_animations(request: TextRequest):
    try:
        # Polish text to structured gloss data
        raw_gloss_data = process_polish_text(request.text)
        data = GlossRequest(**raw_gloss_data)
        
        # Mapping gloss items to animations
        animations: List[AnimationItem] = []
        for clause in data.clauses:
            for gloss_item in clause.pjm_sequence:
                animations.append(map_gloss_to_animation(gloss_item))

        return AnimationResponse(animations=animations)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Błąd tłumaczenia: {str(e)}")