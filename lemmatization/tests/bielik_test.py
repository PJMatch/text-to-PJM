from llama_cpp import Llama
import time

def translate_to_pjm(text):
    start_load = time.perf_counter()

    try:
        llm = Llama(
            model_path="Bielik-11B-v3.0-Instruct-GGUF\Bielik-11B-v3.0-Instruct.Q6_K.gguf", 
            n_gpu_layers=-1,
            n_ctx=32768,
            verbose=False
        )
    except Exception as e:
         return f"Failed to load GGUF model: {e}"
    
    load_time = time.perf_counter() - start_load

    prompt = f"""<|im_start|>system
    Twoim zadaniem jest przekonwertować poniższe polskie zdanie na ciąg słów gotowych do animacji w Polskim Języku Migowym (PJM).
    Oto zasady:
    1. Każde słowo z oryginalnego zdania, które ma sensowny znak w PJM, zapisz w jego formie podstawowej (bezokolicznik dla czasowników, mianownik liczby pojedynczej dla rzeczowników).
    2. Pomijaj CAŁKOWICIE: przyimki (do, w, na, przez), spójniki (i, a, ale) oraz znaki interpunkcyjne.
    3. Odpowiadaj TYLKO przetłumaczonym ciągiem słów, oddzielonych spacją. Żadnych wstępów.
    Przykład zdania: Wczoraj poszedłem do sklepu i kupiłem duże jabłka, żeby upiec ciasto.
    PJM: wczoraj pójść sklep kupić duży jabłko upiec ciasto
    <|im_end|>
    <|im_start|>user
    Zdanie: {text}<|im_end|>
    <|im_start|>assistant
    """

    print("Translating sentence...")
    
    start_infer = time.perf_counter()

    output = llm(
        prompt,
        max_tokens=100,
        stop=["<|im_end|>"],
        echo=False
    )
    
    infer_time = time.perf_counter() - start_infer
    
    print(f"Load time: {load_time:.2f} seconds")
    print(f"Translation time: {infer_time:.2f} seconds")

    return output['choices'][0]['text'].strip()

if __name__ == "__main__":
    test_sentence = "Liżę loda, ale nie chcę go zjeść, bo jest za zimny."
    
    print("Original:", test_sentence)
    result = translate_to_pjm(test_sentence)
    
    print("Result (PJM words):", result)