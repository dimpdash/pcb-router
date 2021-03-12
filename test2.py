from pad_autoencoder import MyTokenizer

tokenizer = MyTokenizer(filters='', num_words=3)

tokenizer.fit_on_texts(['hey'])
print(tokenizer.sequences_to_texts([[1]]) )

tokenizer.fit_on_texts(['hello'])
print(tokenizer.sequences_to_texts([[2]]) )

print(tokenizer.texts_to_sequences(['hey']))
print(tokenizer.texts_to_sequences(['UNCOMMON']))