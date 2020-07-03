import numpy as np

from getModel import *


def predictNews():
    """
    main function
    :return: None
    """
    # Get padded titles, texts, and labels for training and testing
    [[trX, trY], [tsX, tsY]] = prepareData()
    predictor = initModel()
    predictor.summary()

    predictor.fit(x=trX, y=trY, epochs=15, validation_data=[tsX, tsY], verbose=2)
    print("Done")


if __name__ == "__main__":
    predictNews()