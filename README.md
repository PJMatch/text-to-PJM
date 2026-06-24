---

# Text To PJM

This repository contains the Text-to-PJM module of the project PJMatch. 
It includes a rule-based translator that converts Polish sentences into a sequence of PJM glosses, as well as an application that visualizes the generated signs using a 3D avatar.

## Lemmatization

A Natural Language Processing (NLP) engine and API based on **FastAPI**, designed to translate Polish text into a sequence of Polish Sign Language (PJM) glosses. This module was created as a backend for a 3D avatar system in Unreal Engine 5, communicating with it via the HTTP protocol.

### Main Engine Features

* **Advanced syntactic analysis:** Utilizing the `spacy_stanza` library for lemmatization and dependency parsing.
* **PJM grammar adaptation:**
    * Omitting default 3rd-person pronouns for impersonal verbs.
    * Removing relative pronouns (e.g., "kiedy", "który") in affirmative sentences.
* **Smart Fingerspelling:**
    * Automatic fingerspelling for unknown words (missing from the UE5 animation database).
    * Support for digraphs (SZ, CZ, RZ) and Polish diacritics (e.g., Ż -> ZZ).
    * Number decomposition into tens and units (e.g., "34" becomes a sequence of `30` -> `4`).
* **Offline Support:** Ability to run the server without internet access.

### Technologies

* **Python 3**
* **FastAPI** – a fast and asynchronous framework for building APIs.
* **Uvicorn** – an ASGI server.
* **spaCy-stanza** – deep linguistic analysis of the Polish language.

### Configuration
The main configuration for the model's behavior is located at the very top of the `nlp_engine.py` file:
* `EXCEPTIONS` – words that are not subject to lemmatization and fingerspelling (e.g., WARSZAWA).
  `MULTI_WORD_TO_SAFE` & `SAFE_TO_GLOSS` – dictionaries handling multi-word signs/fusions (e.g., "Dzień dobry").
* `FORCED_CLAUSE_ROOTS` – words that always form a separate, independent clause (e.g., "Do widzenia").
* `NEGATED_VERBS_MAP` – mapping for verbs that have a distinct sign when negated (e.g., ROZUMIEĆ -> NIE_ROZUMIEĆ).

## Unreal Engine 5 application

The front end of the text-to-PJM project, responsible for communicating with the lemmatization module and rendering animations on the avatars (Omar and Vivien) in the Unreal Engine environment.

---

# Text To PJM

To repozytorium zawiera moduł Text-to-PJM z projektu PJMatch.
Obejmuje on oparty na regułach translator, który konwertuje polskie zdania na sekwencję glosów PJM, a także aplikację, która wizualizuje wygenerowane znaki za pomocą awatara 3D.

## Lematyzacja

Silnik przetwarzania języka naturalnego (NLP) oraz API oparte na **FastAPI**, służące do tłumaczenia tekstu w języku polskim na sekwencję glosów Polskiego Języka Migowego (PJM). Moduł został stworzony jako backend dla systemu awatarów 3D w Unreal Engine 5, z którym komunikuje się poprzez protokół HTTP.

### Główne funkcje silnika

* **Zaawansowana analiza składniowa:** Wykorzystanie biblioteki `spacy_stanza` do lematyzacji i budowania drzew zależności (dependency parsing).
* **Adaptacja gramatyki PJM:**
    * Pomijanie domyślnych zaimków w 3. osobie dla czasowników bezosobowych.
    * Usuwanie zaimków względnych (np. "kiedy", "który") w zdaniach twierdzących.
* **Inteligentny Fingerspelling:**
    * Automatyczne literowanie słów nieznanych (brakujących w bazie animacji UE5)
    * Obsługa dwuznaków (SZ, CZ, RZ) oraz polskich znaków diakrytycznych (np. Ż -> ZZ).
    * Dekompozycja Liczb na liczbę dziesiątek i jedności (np. "34" staje się sekwencją `30` -> `4`).
* **Wsparcie Offline:** Możliwość uruchomienia serwera bez dostępu do internetu.

### Technologie

* **Python 3**
* **FastAPI** – szybki i asynchroniczny framework do budowy API.
* **Uvicorn** – serwer ASGI.
* **spaCy-stanza** – głęboka analiza lingwistyczna języka polskiego.

### Konfiguracja
Główna konfiguracja zachowań modelu znajduje się na samej górze pliku nlp_engine.py:
* `EXCEPTIONS` – słowa, które nie podlegają lematyzacji i literowaniu (np. WARSZAWA).
  `MULTI_WORD_TO_SAFE` & `SAFE_TO_GLOSS` – słowniki do obsługi zrostów (np. "Dzień dobry").
* `FORCED_CLAUSE_ROOTS` – słowa, które zawsze stanowią osobną, niezależną klauzulę (np. "Do widzenia").
* `NEGATED_VERBS_MAP` – mapowanie czasowników, które w negacji mają osobny znak (np. ROZUMIEĆ -> NIE_ROZUMIEĆ).

## Aplikacjia w Unreal Engine 5

Front end projektu text-to-PJM obsługujący aplikację odpowiedzialną za komunikację z modułem lematyzacji oraz wyświetlanie animacji realizowanej na awatarach (Omar i Vivien) w środowisku Unreal Engine.

---
## Installation

The latest release of the application can be downloaded using the links below.

### Windows

**Download installer:**  
[Download for Windows](https://drive.google.com/file/d/1Al-_PQf6m-j9aPttNQWy3rcC8bqMlgHS/view?usp=sharing)

### Linux

**Download AppImage:**  
[Download for Linux](https://drive.google.com/drive/folders/1DkOJ5ywBLLBAPPKxBfNYxOwf3fHJuwm0?usp=sharing)

### Installation steps

#### Windows
1. Download the installer.
2. Run the `.exe` file.
3. Follow the installation instructions.
4. Launch the application.

#### Linux
1. Download the `.AppImage` file.
2. Make the file executable.
3. Launch the application.