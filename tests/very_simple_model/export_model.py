import VerySimpleModel

if __name__ == '__main__':
    model = VerySimpleModel.VerySimpleModel()
    model.summary()
    model.save("./very_simple_model.h5")
