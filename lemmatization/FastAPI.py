from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import json

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

# Endpoint to convert glosses to animations
# DO ZMIANY TO JEST ROWNIEŻ - aktualnie działa na podstawie danych przesłanych w request (POST), nie jest jeszce połączony z kodem NLP czyli main.py i nie odpala sie automatycznie
@app.post("/glosses-to-animations", response_model=AnimationResponse)
def glosses_to_animations(data: GlossRequest):
    animations: List[AnimationItem] = []

    for clause in data.clauses:
        for gloss_item in clause.pjm_sequence:
            animations.append(map_gloss_to_animation(gloss_item))

    return AnimationResponse(animations=animations)

# Endpoint to load glosses from a JSON file and return corresponding animations
# This is a temporary endpoint for testing purposes
@app.post("/load-json-file", response_model=AnimationResponse)
def load_json_file():
    try:
        with open("results_glosses.json", "r", encoding="utf-8") as f:
            raw = json.load(f)

        data = GlossRequest(**raw)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Nie znaleziono pliku results_glosses.json")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Błąd wczytywania JSON: {str(e)}")

    animations: List[AnimationItem] = []

    for clause in data.clauses:
        for gloss_item in clause.pjm_sequence:
            animations.append(map_gloss_to_animation(gloss_item))

    return AnimationResponse(animations=animations)