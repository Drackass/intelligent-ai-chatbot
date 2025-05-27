from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('wordnet')

print(word_tokenize("Bonjour ! Ceci est un test."))
