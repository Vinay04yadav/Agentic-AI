import tiktoken

# Make an encoder
enc = tiktoken.encoding_for_model("gpt-4o")

# Provide the text
text = "Hey there! My name is Vinay"
tokens = enc.encode(text) # tokens are stored in list

print(f"Tokens are: {tokens}")
# Tokens are: [25216, 1354, 0, 3673, 1308, 382, 23967, 356]

tokensList = [25216, 1354, 0, 3673, 1308, 382, 23967, 356]
DecodedText = enc.decode(tokensList) # OR enc.decode([25216, 1354, 0, 3673, 1308, 382, 23967, 356])
print(f"Decoded text for tokens is: {DecodedText}")