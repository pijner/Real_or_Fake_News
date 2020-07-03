import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from getData import getReal, getFake, cleanRealTexts, dateToInt


def dateExplore(realDates, fakeDates):
    # Check distribution of news by year
    plt.figure(figsize=(16, 9))
    plt.hist([realDates[0, realDates[0, :] != 0], fakeDates[0, fakeDates[0, :] != 0]])
    plt.xticks(list(set(np.unique(fakeDates[0])).union(np.unique(fakeDates[0])))[1:])
    plt.title("Distribution of fake and real news per year")
    plt.xlabel("Year")
    plt.ylabel("Frequency")
    plt.legend(["Real", "Fake"])
    plt.show()

    # Check distribution of news by month
    allYears = np.unique(np.concatenate((realDates[0, :], fakeDates[0, :])))
    rows = int(np.floor((len(allYears)-1) / 2))
    cols = int(np.ceil((len(allYears)-1) / rows))
    fig, ax = plt.subplots(rows, cols, figsize=(16, 9))
    for i, year in enumerate(allYears[1:]):
        r = int(np.floor(i / 2))
        c = i % 2
        ax[r, c].hist([realDates[1, realDates[0, :] == year], fakeDates[1, fakeDates[0, :] == year]],
                      # bins=list(range(0, 13)),
                      label=['Real', 'Fake'])
        ax[r, c].set_title(str(year))
        ax[r, c].set_xlabel("Month")
        ax[r, c].set_ylabel("Frequency")
        ax[r, c].set_xticks(list(range(1, 13)))

    fig.legend(["Real", "Fake"])
    fig.suptitle("Distribution of real and fake news by month per year")
    plt.subplots_adjust(hspace=0.4)

    plt.show()


def textExplore(realTexts, fakeTexts, ctxt):
    realLengths = np.array([len(t.split()) for t in realTexts])
    fakeLengths = np.array([len(t.split()) for t in fakeTexts])

    realSD = np.var(realLengths) ** 0.5
    fakeSD = np.var(fakeLengths) ** 0.5

    realMean = np.mean(realLengths)
    fakeMean = np.mean(fakeLengths)
    print("Real mean =", realMean, "\tSD =", realSD)
    print("Fake mean =", fakeMean, "\tSD =", fakeSD)

    y_max = int(max(realSD * 3 + realMean, fakeSD * 3 + fakeMean))
    y_min = int(min(realSD * 3 - realMean, fakeSD * 3 - fakeMean))


    # Boxplot for text length
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].boxplot([realLengths, fakeLengths])
    ax[0].set_xticklabels(['Real', 'Fake'])
    ax[0].set_title("Full range")

    ax[1].boxplot([realLengths, fakeLengths])
    ax[1].set_xticklabels(['Real', 'Fake'])
    ax[1].set_title("y-range = (0, " + str(y_max) + ")")
    ax[1].set_ylim([-10, y_max])

    fig.suptitle("Box-plot of length of " + ctxt)

    plt.show()

    # Histograms showing distribution of lengths of texts
    fig, ax = plt.subplots(1, 2, figsize=(16, 9))

    ax[0].hist([realLengths[realLengths < y_max], fakeLengths[fakeLengths < y_max]], bins=50)
    ax[0].set_title("Distribution of lengths below " + str(y_max))
    ax[0].set_xlabel(ctxt + " length (words)")
    ax[0].set_ylabel("Frequency")

    ax[1].hist([realLengths[realLengths > y_max], fakeLengths[fakeLengths > y_max]], bins=50)
    ax[1].set_title("Distribution of lengths above " + str(y_max))
    ax[1].set_xlabel(ctxt + " length (words)")
    ax[1].set_ylabel("Frequency")

    fig.suptitle("Histograms of length of " + ctxt)
    fig.legend(("Real", "Fake"))

    plt.show()

    realTexts = cleanRealTexts(realTexts)

    makeWordCloud(realTexts, "Real " + ctxt)
    makeWordCloud(fakeTexts, "Fake " + ctxt)

    print("Done")


def makeWordCloud(text, title):
    from wordcloud import WordCloud, STOPWORDS
    import re

    mergedText = " ".join(text)
    plt.figure(figsize=(16, 9))
    plt.imshow(WordCloud(stopwords=STOPWORDS,
                         width=7000,
                         height=4000,
                         background_color='white').generate_from_text(mergedText))
    plt.axis('off')
    plt.title(title)
    plt.show()


def subjectExplore(realSubjects, fakeSubjects):
    plt.figure(figsize=(16, 9))
    plt.hist([realSubjects, fakeSubjects])
    plt.legend(["Real", "Fake"])
    plt.title("Distribution of news articles among subjects")
    plt.xlabel("News subject")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    realData = getReal()
    fakeData = getFake()

    print(realData.head())
    print(fakeData.head())

    # Date format for index 9050 - 9085 is different from the rest of the data
    # dateExplore(dateToInt(list(realData['date'])), dateToInt(list(fakeData['date'])))
    # subjectExplore(list(realData['subject']), list(fakeData['subject']))
    textExplore(list(realData['text']), list(fakeData['text']), ctxt="text")

    print("Done")
