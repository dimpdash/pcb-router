from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(filters='', num_words=3)

tokenizer.fit_on_texts(['hey'])
print(tokenizer.sequences_to_texts([[1]]) )

tokenizer.fit_on_texts(['hello'])
print(tokenizer.sequences_to_texts([[2]]) )