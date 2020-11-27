import pickle

with open("projectLinks954 - Copy.data", "rb") as filehandle:
    a = pickle.load(filehandle)
    print(len(a))
