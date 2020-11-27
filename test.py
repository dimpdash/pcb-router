import pickle

with open("ids 271.data", "rb") as filehandle:
    a = pickle.load(filehandle)
    print(len(a))


with open("ids 23000.data", "rb") as filehandle:
    a = pickle.load(filehandle)
    print(len(a))

