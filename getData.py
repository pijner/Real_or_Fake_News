import numpy as np
import pandas as pd


def getReal(directory="../FakeRealNews/Data"):
    """
    function to read real news data from csv
    :param directory: directory containing True.csv
    :return: data frame of true news
    """
    return pd.read_csv(directory + "/True.csv")


def getFake(directory="../FakeRealNews/Data"):
    """
    function to read fake news data from csv
    :param directory: directory containing Fake.csv
    :return: data frame of true news
    """
    return pd.read_csv(directory + "/Fake.csv")


def splitData(groupList, trainSize):
    """
    function to split data to training and testing parts
    :param groupList: list of dataframes from each group
    :param trainSize: fraction of data set to use for training (between 0-1)
    :return: list containing training and testing information
                each element in list is a list consisting of data and labels
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    groupList[0]['text'] = cleanRealTexts(list(groupList[0]['text']))

    classLabels = np.array([])
    for i, group in enumerate(groupList):
        classLabels = np.append(classLabels, np.repeat(i, len(group)))

    classData = pd.concat(groupList).reset_index(drop=True)

    splits = list(StratifiedShuffleSplit(n_splits=i,
                                         test_size=1-trainSize,
                                         train_size=trainSize,
                                         random_state=0).split(X=classData, y=classLabels))[0]
    trainIdx, testIdx = splits

    trainData = classData.iloc[trainIdx]
    testData = classData.iloc[testIdx]
    trainLabels = classLabels[trainIdx]
    testLabels = classLabels[testIdx]

    return [[trainData, trainLabels], [testData, testLabels]]


def getData(trainSize):
    """
    function to get training and testing data based on training size
    :param trainSize: fraction of data set to use for training (between 0-1)
    :return: list containing training and testing information
                each element in list is a list consisting of data and labels
    """
    return splitData([getReal(), getFake()], trainSize=trainSize)


def dateToInt(listOfDates):
    """
    function to convert dates in data frame to an integer format
    Supported formats: [Month*dd*yyyy, Mon*dd*yyyy, dd-mm-yy] (* represents any non digit character)
    :param listOfDates: list consisting of dates
    :return: list consisting of list of years, months, and dates
    """
    import re, logging
    months = {"0": 0,
              "Jan": 1,
              "Feb": 2,
              "Mar": 3,
              "Apr": 4,
              "May": 5,
              "Jun": 6,
              "Jul": 7,
              "Aug": 8,
              "Sep": 9,
              "Oct": 10,
              "Nov": 11,
              "Dec": 12}
    monthList = []
    dateList = []
    yearList = []

    for date in listOfDates:
        thisYear = "0"
        thisDate = "0"
        thisMonth = "0"
        try:
            thisMonth, thisDate, thisYear = re.findall("([A-Za-z]{3}).+?([0-9]+), (.+)", date)[0]

        except IndexError:
            try:
                thisData, thisMonth, thisYear = re.split('-', date)
                thisYear = "20" + thisYear
            except:
                logging.info("Error in date format for index" + str(len(yearList)) + "FOUND: " + date)

        finally:
            monthList.append(months[thisMonth])
            dateList.append(int(thisDate))
            yearList.append(int(thisYear))

    return np.array([np.array(yearList), np.array(monthList), np.array(dateList)])


def cleanRealTexts(realTexts):
    """
    function to remove source from real news texts
    :param realTexts: list of real news texts
    :return: list of cleaned real news texts
    """
    import re
    cleanTexts = []
    for text in realTexts:
        cleanTexts.append(re.split("\(Reuters\) - ", text)[-1])

    return cleanTexts