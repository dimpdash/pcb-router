import pickle

with open("ids tmp -8981.data", "rb") as filehandle:
    a = pickle.load(filehandle)
    print(len(a))


