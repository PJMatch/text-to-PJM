import morfeusz2
import os

filepath="output.txt"
report_path="report.md"

# Check if input file exists
if not os.path.exists(filepath):
    print(f"Error: File '{filepath}' not found.")

# Initialize Morfeusz with the default Polish dictionary
morf = morfeusz2.Morfeusz()

total_words = 0
correct_words = 0
incorrect_lemmas = set()

# Read and process the input file
with open(filepath, "r", encoding="utf-8") as file:
    for line in file:
        words = line.strip().split()
        
        for word in words:
            total_words += 1
            lemma_lower = word.lower()
            analysis = morf.analyse(lemma_lower)
            
            is_valid = False
            
            for token in analysis:
                # Skip unrecognized tokens
                if 'ign' in str(token):
                    continue
                    
                raw_lemma = ""
                
                # Extract lemma
                if hasattr(token, 'lemma'):
                    raw_lemma = token.lemma
                elif isinstance(token, tuple) and len(token) >= 3:
                    if isinstance(token[2], tuple) and len(token[2]) >= 2:
                        raw_lemma = token[2][1]
                
                if not raw_lemma:
                    continue

                # Remove dictionary tags
                morfeusz_base_lemma = str(raw_lemma).split(':')[0].lower()
                print(f"Checking word '{lemma_lower}' -> lemma '{morfeusz_base_lemma}'")
                
                # Validate if the word is a proper base form
                if lemma_lower == morfeusz_base_lemma:
                    is_valid = True
                    break
                    
            if is_valid:
                correct_words += 1
            else:
                incorrect_lemmas.add(word)

# Generate Markdown report
with open(report_path, "w", encoding="utf-8") as report_file:
    report_file.write("# Lemmatization Accuracy Report\n\n")
    report_file.write(f"**Total analyzed words:** {total_words}\n\n")
    
    if total_words > 0:
        accuracy = (correct_words / total_words) * 100
        report_file.write(f"**Accuracy:** {accuracy:.2f}%\n\n")
    else:
        report_file.write("**Accuracy:** No data (empty file)\n\n")

    report_file.write("## Incorrect forms:\n")
    
    if incorrect_lemmas:
        for bad_word in sorted(incorrect_lemmas):
            report_file.write(f"- {bad_word}\n")
    else:
        report_file.write("*All words are correct.*\n")

print(f"\nreport generated and saved to: {report_path}")