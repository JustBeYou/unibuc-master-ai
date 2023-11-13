import os
import random

# Function to generate a paragraph with a specified number of words
def generate_paragraph(num_words):
    words = [
                "și", "în", "este", "la", "care", "de", "a", "cu", "pe", "o", "sau",
                "un", "pentru", "mai", "ca", "sunt", "acest", "se", "pot", "aceasta",
                "din", "fie", "dar", "acestui", "fost", "prin", "aceea", "poate", "ce",
                "aceste", "când", "doar", "cum", "dintre", "fi", "avea", "mult", "fără",
                "bun", "timp", "unde", "chiar", "acestei", "meu", "alte", "al", "dacă",
                "toate", "ni", "așa", "făcut", "zi", "acestor", "multe", "între", "prea",
                "ori", "mereu", "unele", "acei", "apoi", "aceștia", "viața", "tot",
                "oricare", "lucru", "uneori", "despre", "oricum", "fostul", "două",
                "lui", "face", "acela", "tău", "alt", "aceia", "mod", "cel", "lume",
                "acelea", "acum", "făcând", "an", "către", "așadar", "acel", "mare",
                "acele", "ai", "oare", "parte", "acești", "această", "fel", "asta",
                "fiecare", "cât", "acelor", "noi", "nou", "aceeași", "cine", "îmi",
                "nostru", "dintr", "același", "fără", "aceasta", "acestora", "fiind",
                "vor", "oricând", "vei", "spune", "acelui", "asta", "acestea", "nici",
                "acestia", "lucruri", "totuși", "acești", "această", "acolo", "deci",
                "altceva", "aceluia", "dintr-un", "nouă", "aici", "făcea", "câte",
                "după", "același", "mea", "nostră", "acestui", "acel", "vreo", "aceiași",
                "tăi", "deja", "înainte", "ceva", "aceștia", "acelora", "face", "această"
            ]

    return ' '.join(random.choice(words) for _ in range(num_words))

# Function to generate a document with a specified number of paragraphs
def generate_document(num_paragraphs, paragraph_length, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for _ in range(num_paragraphs):
            paragraph = generate_paragraph(paragraph_length)
            file.write(paragraph + "\n\n")

# Generate 1000 documents, each with 100 paragraphs of 200 words
def generate_documents(num_documents, num_paragraphs, paragraph_length, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(1, num_documents + 1):
        file_path = os.path.join(directory, f'document_{i}.txt')
        generate_document(num_paragraphs, paragraph_length, file_path)
        if i % 100 == 0:  # Print a message every 100 documents
            print(f"{i} documents generated...")

# Parameters
NUM_DOCUMENTS = 100
NUM_PARAGRAPHS = 50
PARAGRAPH_LENGTH = 200
DIRECTORY = '.'

# Generate the documents
generate_documents(NUM_DOCUMENTS, NUM_PARAGRAPHS, PARAGRAPH_LENGTH, DIRECTORY)

# Indicate completion
print("Document generation complete.")