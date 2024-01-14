# onkelos
Programs related to the computational analysis of the Aramaic found in the Targum Onkelos

scraper pulls the text from the cal.huc.edu database and formats it into csvs, vowels removed, Latin characters and Hebrew characters.
Has the ability to format text broken up by either book, chapter, or verse. Primarily uses BeautifulSoup.

train_clf is built off of a program by @dimidd that is trained on common tractates in the Talmudim and analyzes the uncommon tractates
and those mistakenly labelled. It now still trains on the common tractates of the Talmudim (half in Jewish Palestinian Aramaic and half
in Jewish Babylonian Aramaic) but instead analyzes the Targum Onkelos to predict the dialectal origin of different parts of the text.

It utilizes the Multinomial Naive Bayes ML algorithm primarily, but also can analyze using Logistic Regression and LightGBM. For NB,
it has the functionality of displaying the most influential words for making the prediction, along with that word's weight. Alongside
the predictions, text can be displayed in Hebrew and in Latin characters. Primarily uses sklearn, pandas.

analyze_results tallies up the verses attributed to yerushalmi, taking into account confidence of the prediction.

DATE: 1/14/24
CONTRIBUTORS: scraper - Eitan Zemel, train_clf - Dimid Duchovny, Eitan Zemel, analyze_results - Eitan Zemel
