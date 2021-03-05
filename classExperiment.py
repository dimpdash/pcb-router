from pad_autoencoder import DataPipeline


class B():
    def __init__(self, a = 1):
        self.a = a
        print("hey")

class A(B):
    pass
    # def __init__(self):
    #     pass

A_inst = A()
print(A_inst.a)

class Sequence():
    pass

class loadable():
    def loadData(self):
        pass

class dataGenerator(Sequence, loadable):
    def load_data(self, ids):
        pass

valData = dataGenerator()
trainData = dataGenerator()

class dataPipeline(loadable):
    def __init__(self):
        self.trainData = dataGenerator()
    def preprocessData(self):
        self.loadData(ids)