import json
import pathlib
import random
from typing import Tuple, NamedTuple, FrozenSet, Union, Dict
import heapq

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
# from skift import SeriesFtClassifier


class TrainConfig(NamedTuple):
    name: str
    train_tractates: FrozenSet[str]
    is_word_balanced: bool


LAT_TO_HEB = {
    ')': 'א',
    'b': 'ב',
    'g': 'ג',
    'd': 'ד',
    'h': 'ה',
    'w': 'ו',
    'z': 'ז',
    'x': 'ח',
    'T': 'ט',
    'y': 'י',
    'k': 'כ',
    'K': 'ך',
    'l': 'ל',
    'm': 'מ',
    'M': 'ם',
    'n': 'נ',
    'N': 'ן',
    's': 'ס',
    '(': 'ע',
    'p': 'פ',
    'P': 'ף',
    'c': 'צ',
    'C': 'ץ',
    'q': 'ק',
    'r': 'ר',
    '$': 'ש',
    '&': 'שׂ',
    't': 'ת',
}

lat_chars = ''.join(LAT_TO_HEB.keys())


def replace_with_heb_chars(w: str):
    heb = map(lambda c: LAT_TO_HEB.get(c, c), w)

    return ''.join(list(heb))


def lat_to_heb(st: str):
    words = st.split()
    heb_words = list(map(replace_with_heb_chars, words))

    return ' '.join(heb_words)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)

# csvs not being found without full location string
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    jer = pd.read_csv(r'C:/Users/eitan/OneDrive/Documents/Word Embedding in Aramaic Dialects/analysis/yerushalmi_2022_08_03.csv')
    jer.dropna(subset=['txt', 'tractate'], inplace=True)
    jer['talmud'] = 'jer'
    jer.dropna(subset=['txt', 'tractate'], inplace=True)

    bav_file = open(r'C:/Users/eitan/OneDrive/Documents/Word Embedding in Aramaic Dialects/analysis/bavli_2022_08_03.csv', encoding='utf-8', errors=None)
    bav = pd.read_csv(bav_file)
    bav['txt'] = bav.txt.str.replace(r"^'", '', regex=True)
    bav.dropna(subset=['txt', 'tractate'], inplace=True)
    bav['talmud'] = 'bav'

    talmuds = pd.concat([jer, bav])
    talmuds['n_words'] = [len(r.split()) for r in talmuds.txt.tolist()]

    targ_file = open(r'C:/Users/eitan/OneDrive/Documents/Word Embedding in Aramaic Dialects/Targum Onkelos Verses.csv', encoding='utf-8')
    targum = pd.read_csv(targ_file)
    targum.dropna(inplace=True) #what is relevance of column titles in subset?
    

    return talmuds, bav, jer, targum

def get_salient_words(nb_clf, vect, class_ind, count:bool=True):
    """taken from Dimid from StackOverflow
    
    Return salient words for given class
    
    Parameters
    ----------
    nb_clf : a Naive Bayes classifier (e.g. MultinomialNB, BernoulliNB)
    vect : CountVectorizer
    class_ind : int
    count : bool, if True returns list of (word, count), else (word, prob)

    Returns
    -------
    list: a sorted list of (word, count or prob) sorted by probability in descending order.
    """

    words = vect.get_feature_names_out()
    
    if count:
        zipped = list(zip(words, nb_clf.feature_count_[class_ind]))
    else:
        zipped = list(zip(words, np.exp(nb_clf.feature_log_prob_[class_ind]))) # np.exp() gets the actual, not log, probs


    sorted_zip = sorted(zipped, key=lambda t: t[1], reverse=True)

    return sorted_zip

# neg_salient_top_20 = get_salient_words(NB_optimal, count_vect, 0)[:20]
# pos_salient_top_20 = get_salient_words(NB_optimal, count_vect, 1)[:20]

def get_tractates(talmuds: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    babli_indices = talmuds.talmud == 'bav'
    yerushalmi_indices = talmuds.talmud == 'jer'
    babli_tractates = talmuds[babli_indices]['tractate'].unique()
    print(f"{babli_tractates=}")
    yerushalmi_tractates = talmuds[yerushalmi_indices]['tractate'].unique()
    print(f"{yerushalmi_tractates=}")
    tractates = talmuds['tractate'].unique()
    common_tractates_a = np.intersect1d(babli_tractates, yerushalmi_tractates)
    different_tractates_a = np.setdiff1d(tractates, common_tractates_a)
    assert len(common_tractates_a) == 28, "There are 28 common tractates and 20 non-common tractates"
    assert len(different_tractates_a) == 20, "There are 28 common tractates and 20 non-common tractates"

    return common_tractates_a, different_tractates_a


def create_experiments(talmuds: pd.DataFrame, targum: pd.DataFrame, full=True, model='cnt_vclf'):
    assert model in ('ft', 'cnt_vclf'), "Model must be one of: ('ft', 'cnt_vclf')"
    if model == 'ft':
        run_exp_method = run_experiment_ft
    else:
        run_exp_method = run_experiment_cnt_vclf
    common_tractates_a, different_tractates_a = get_tractates(talmuds)
    # common_conf = TrainConfig('common_non_word_balanced', frozenset(common_tractates_a), False)
    # run_exp_method(talmuds, common_conf, targum=targum)
    
    
    common_conf_word_balanced = TrainConfig('common_word_balanced', frozenset(common_tractates_a), True)
    run_exp_method(talmuds, common_conf_word_balanced, targum=targum)
    if not full:
        return
    
    ############################################

    # common_tractates_a = common_tractates_a[common_tractates_a != 'Nedarim']
    # for tractate_ind, tractate in enumerate(common_tractates_a):
    #     trainset = set(common_tractates_a)
    #     trainset.remove(tractate)
    #     conf = TrainConfig(f'common_except_{tractate}_non_word_balanced', frozenset(trainset), False)
    #     print(f"Experiment {tractate_ind + 1}/{len(common_tractates_a)}: {conf.name}")
    #     run_exp_method(talmuds, conf)
    #     conf_balanced = TrainConfig(f'common_except_{tractate}_word_balanced', frozenset(trainset), True)
    #     run_exp_method(talmuds, conf_balanced)


def run_experiment_ft(talmuds: pd.DataFrame, conf: TrainConfig):
    # ft_clf = SeriesFtClassifier()
    run_experiment(talmuds, conf, clf=ft_clf, clf_name='fasttext')


def run_experiment_cnt_vclf(talmuds: pd.DataFrame, conf: TrainConfig, targum: pd.DataFrame):
    vclf = Pipeline(
        [
            ('vect', 
             CountVectorizer(dtype=np.float32,
                             #ngram_range=(1,1),
                             token_pattern='(?<![\\w()$&])[\\w()$&][\\w()$&]+(?![\\w()$&])',
                             lowercase=False
                             )),
            ('clf', VotingClassifier(
                estimators=[
                    ('nb', MultinomialNB()),
                    ('cnb', ComplementNB()),
                    # TODO: investigate why LGB gives unequal probs, even after word balancing
                    ('lgb', lgb.LGBMClassifier(n_estimators=610, max_depth=14,   boosting_type='dart')),
                    ('lr', LogisticRegressionCV(max_iter=100))
                ],
                # weights=[1, 1, 3, 1],
                weights=[1,0,0,0], # consider only nb
                voting='soft'
            ))
        ]
    )
    run_experiment(talmuds, conf, clf=vclf, clf_name='cnt_nb_ngram_1_1', targum=targum,
                   separate_estimators=False, identify_words=True, latin_text=False)


def balance_words(talmuds: pd.DataFrame) -> pd.DataFrame:
    jer = talmuds[talmuds.talmud == 'jer']
    bav_counts = talmuds[talmuds.talmud == 'bav'].n_words.value_counts()
    bav_counts_d = bav_counts.to_dict()
    jer_counts = jer.n_words.value_counts()
    jer_counts_d = jer_counts.to_dict()
    bav_sample_inds = []
    exclude_counts = []
    for n, jer_count in sorted(jer_counts_d.items()):
        if n not in bav_counts_d:
            exclude_counts.append(n)
            continue
        bav_count = bav_counts_d[n]
        min_count = min(jer_count, bav_count)
        #print(n, bav_count, jer_count, min_count)
        inds = talmuds[(talmuds.n_words == n) & (talmuds.talmud == 'bav')].sample(min_count).index.tolist()

        bav_sample_inds += inds

    bav_balanced_sample = talmuds.loc[bav_sample_inds]
    remaining_jer = jer[~jer.n_words.isin(exclude_counts)]
    talmuds_balanced = pd.concat([bav_balanced_sample, remaining_jer], ignore_index=True)

    return talmuds_balanced.drop_duplicates()

def feature_importance_nb(clf: sklearn.pipeline.Pipeline, csv: bool) -> tuple:
    '''Identifies the importance of each feature of the NB classifier.
    
    If specified, exports info to a csv including Hebrew script for words.

    Parameters
    ----------
    clf: Pipeline, assumes that the NB classifier is the first estimator
    csv: bool, determines if csvs are generated

    Returns
    -------
    tuple: (yerushalmi_words, bavli_words)
    '''
    nb = list(clf.named_steps['clf'].named_estimators_.items())[0][1]
    vect = clf.named_steps['vect']
    bavli_words = dict(get_salient_words(nb, vect, 1))
    yerushalmi_words = dict(get_salient_words(nb, vect, 0))
    

    '''ratios dicts are {word -> (bav or jer count/total count , ratio bav count to jer count)}
    1 is added to all counts to avoid zeros
    '''
    bavli_ratios = {}
    yerushalmi_ratios = {}
    ratios = {}

    for word in vect.get_feature_names_out():
        bav_count = 1
        jer_count = 1
        try: 
            bav_count += bavli_words[word]
            hello = 19
        except: pass
        try: 
            jer_count += yerushalmi_words[word]
            hello = 5
        except: pass
        ratio = f'{bav_count}/{jer_count}'
        bavli_ratios[word] = bav_count/(bav_count+jer_count)
        yerushalmi_ratios[word] = jer_count/(bav_count+jer_count)
        ratios[word] = ratio


    if csv:
        # words_bav_df = pd.DataFrame.from_dict(bavli_words)
        # words_bav_df.columns = ['word', 'proba']
        # words_bav_df["hebrew"] = words_bav_df.word.apply(lat_to_heb)
        # words_bav_df.to_csv('actual prob nb bav words.csv', index=False)
        
        # words_jer_df = pd.DataFrame.from_dict(yerushalmi_words)
        # words_jer_df.columns=['word', 'proba']
        # words_jer_df["hebrew"] = words_jer_df.word.apply(lat_to_heb)
        # words_jer_df.to_csv("actual prob nb jer words.csv", index=False)
        
        bavli_ratios_series = pd.Series(bavli_ratios, name='Ratio')
        words_bav_df = pd.DataFrame(bavli_ratios_series).reset_index()
        # words_bav_df = pd.DataFrame.from_dict(bavli_ratios)
        words_bav_df.columns = ['word', 'bav count/total count']
        words_bav_df["hebrew"] = words_bav_df.word.apply(lat_to_heb)
        words_bav_df.to_csv('nb bav words weights.csv', index=False)
        
        yerushalmi_ratios_series = pd.Series(yerushalmi_ratios, name='Ratio')
        words_jer_df = pd.DataFrame(yerushalmi_ratios_series).reset_index()
        # words_jer_df = pd.DataFrame.from_dict(yerushalmi_words)
        words_jer_df.columns=['word', 'jer count/total count']
        words_jer_df["hebrew"] = words_jer_df.word.apply(lat_to_heb)
        words_jer_df.to_csv("nb jer words weights.csv", index=False)

        # ratios_df = pd.DataFrame.from_dict(ratios)
        ratios_series = pd.Series(ratios, name='Ratio')
        ratios_df = pd.DataFrame(ratios_series).reset_index()
        ratios_df.columns=['word', 'ratio bav/jer']
        ratios_df["hebrew"] = ratios_df.word.apply(lat_to_heb)
        ratios_df.to_csv("nb words ratios.csv", index=False)

    # return (yerushalmi_words, bavli_words)
    return (yerushalmi_ratios, bavli_ratios)


def run_experiment(talmuds: pd.DataFrame, conf: TrainConfig, clf: sklearn.pipeline.Pipeline,
                   clf_name: str, targum: pd.DataFrame, separate_estimators: bool,
                   identify_words: bool, latin_text: bool):
    '''Built off of Dimid's original method.

    Parameters
    ----------
    talmuds : pd.DataFrame containing the training text dataset, the two Talmudim
    conf : TrainConfig
    clf : sklearn.pipeline.Pipeline containing a CountVectorizer 'vect' and a Voting Classifier 'clf'
    clf_name : str
    targum : pd.DataFrame containing the testing dataset, the Targum Onkelos
    separate_estimators : bool, determines if the resultant csv includes the other algorithms
    identify_words : bool, determines if the resultant csv includes most important features/words
    latin_text : bool, determines whether the latin text is in the final csv
    '''

    # experiment_dir = f"../data/experiments/{clf_name}/{conf.name}"
    # experiment_dir = "../Targum Onkelos.csv"
    # pathlib.Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    train_val = talmuds[talmuds['tractate'].isin(conf.train_tractates)]
    if conf.is_word_balanced:
        train_val = balance_words(train_val)
    # train_val.to_csv(f"{experiment_dir}/train_set.csv")
    # test = talmuds[~talmuds['tractate'].isin(conf.train_tractates)]
    # test.to_csv(f"{experiment_dir}/test_set.csv")

    # train_val['hebrew'] = train_val['txt'].apply(lat_to_heb)
    # train_val.to_csv('Talmudim Hebrew and Latin Balanced.csv', index=False)

    clf.fit(train_val['txt'], (train_val['talmud'] == 'bav').astype(int)) # 1 = bav, 0 = jer

    # predicted_cnt_vclf = clf.predict(test['txt'])
    # predicted_proba_cnt_vclf = clf.predict_proba(test['txt'])
    # print(clf.classes_)
    
    predicted_cnt_vclf = clf.predict(targum['txt'])
    predicted_proba_cnt_vclf = clf.predict_proba(targum['txt'])
    
    bav_col = [prob[1] for prob in predicted_proba_cnt_vclf]
    jer_col = [prob[0] for prob in predicted_proba_cnt_vclf]
    diff_col = [abs(prob[1]-prob[0]) for prob in predicted_proba_cnt_vclf]
    chosen_class = ['bav' if x else 'jer' for x in predicted_cnt_vclf]
    targum['Bavli %'] = bav_col
    targum['Yerushalmi %'] = jer_col
    targum['Difference'] = diff_col
    targum['Prediction'] = chosen_class
   
    if identify_words:
        feature_importance = feature_importance_nb(clf, csv=False)
        yerushalmi_words = feature_importance[0]
        bavli_words = feature_importance[1]

        # identify and record influential words for each entry (verse)
        influential_words = [[],[],[]]
        influential_probs = [[],[],[]]
        for n, verse in enumerate(targum['txt'].values):
            words = verse.split()
            proba = dict(bavli_words if predicted_cnt_vclf[n] else yerushalmi_words)
            pqueue = []
            for w in words:
                try:
                    heapq.heappush(pqueue, (-1*proba[w], w)) # made negative to essentially sort by largest val
                except:
                    continue
            for i in  range(0,3):        
                try:
                    word = heapq.heappop(pqueue)
                    influential_words[i].append(word[1])
                    influential_probs[i].append(-1*word[0])
                except:
                    influential_words[i].append("null")
                    influential_probs[i].append(0)
        
        # add to df
        for i in range(0,len(influential_words)):
            if latin_text:
                targum[f'sig word {i+1}'] = influential_words[i]
            targum[f'sig word {i+1} hebr'] = list(map(lat_to_heb, influential_words[i]))
            targum[f'word {i+1} weight'] = influential_probs[i]
            
   
    # count = np.unique(predicted_cnt_vclf, return_counts=True)
    # count_dict = {'jer':count[1][0], 'bav':count[1][1]}
    
    if separate_estimators:
        # add other separate probas for each algo
        vectorized_txts = clf.named_steps['vect'].transform(targum['txt'])
        for estimator_name, estimator in clf.named_steps['clf'].named_estimators_.items():
            indiv_probs_cur = estimator.predict_proba(vectorized_txts)
            targum[f'prob_{estimator_name}_bav'] = indiv_probs_cur[:, 1]
            targum[f'prob_{estimator_name}_jer'] = indiv_probs_cur[:, 0]
    
    if not latin_text:
        targum.drop('txt', axis=1, inplace=True)

    targum.to_csv('Verses with Sig Words NB weights no eng.csv', index=False)
################################################################################
    
    # true_bin = (test.talmud == 'bav').values.astype(int)
    # accuracy = np.mean(predicted_cnt_vclf == true_bin)
    # auc_score = roc_auc_score(true_bin, predicted_cnt_vclf)
    # cm = confusion_matrix(true_bin, predicted_cnt_vclf)
    # heatmap_ax = sns.heatmap(cm, annot=True)
    # heatmap_ax.set(xlabel='Predicted', ylabel='True')
    # plt.savefig(f"{experiment_dir}/confusion_matrix.png")
    # plt.clf()

    # metrics = {'acc': accuracy, 'roc_auc': auc_score}
    # with open(f"{experiment_dir}/metrics.json", 'w') as metrics_fp:
    #     json.dump(metrics, metrics_fp, indent=4)

    # mistake_inds = true_bin != predicted_cnt_vclf
    # # predicted_proba_cnt_vclf = clf.predict_proba(test['txt'])
    # mistake_probs = predicted_proba_cnt_vclf[mistake_inds]
    # mistake_df = test[['cal_ind', 'txt', 'talmud', 'tractate']][mistake_inds]
    # mistake_df['prob_bav'] = mistake_probs[:, 1]
    # mistake_df['prob_jer'] = mistake_probs[:, 0]
    # mistake_df['txt_len'] = mistake_df['txt'].apply(len)
    # mistake_df['prob_diff'] = abs(mistake_df['prob_bav'] - mistake_df['prob_jer'])
    # vectorized_txts = clf.named_steps['vect'].transform(mistake_df['txt'])
    
    
    # # for estimator_name, estimator in clf.named_steps['clf'].named_estimators_.items():
    # #     mistake_probs_cur = estimator.predict_proba(vectorized_txts)
    # #     mistake_df[f'prob_{estimator_name}_bav'] = mistake_probs_cur[:, 1]
    # #     mistake_df[f'prob_{estimator_name}_jer'] = mistake_probs_cur[:, 0]
   
   
    # mistake_df.set_index('cal_ind', inplace=True)
    # mistake_df['txt_heb'] = mistake_df.txt.apply(lat_to_heb)
    # CAL_URL = "http://cal.huc.edu/get_a_chapter.php?file={}&sub={}&cset=H&row={}"
    # mistake_df['cal_url'] = mistake_df.index.map(lambda x: CAL_URL.format(x[:5], x[5:7], x))
    # mistake_df.to_csv(f'{experiment_dir}/mistakes.tsv', sep='\t')
    

def count_mistakes(clf_name: str) -> Dict[Tuple[str, str], int]:
    tractate_to_mistakes = {}
    for exp in list(pathlib.Path(f'../data/experiments/{clf_name}/').glob('common_*/')):
        tractate = str(exp.name).replace('common_except_', '')
        mistakes = pd.read_csv(f'{exp}/mistakes.tsv', sep='\t')
        for talmud in ('bav', 'jer'):
            cur_mistakes = mistakes[(mistakes.tractate == tractate) & (mistakes.talmud == talmud)]
            tractate_to_mistakes[(talmud, tractate)] = len(cur_mistakes)

    return tractate_to_mistakes


def analyze_mistakes(talmuds, clf_name: str, tractate_to_mistakes: Dict[Tuple[str, str], int]):
    res_df = pd.DataFrame(tractate_to_mistakes.items(), columns=['tractate', 'n_mistakes'])
    res_df.set_index('tractate', inplace=True)
    counts = talmuds.groupby(['talmud', 'tractate']).size()
    res_df['n_units'] = counts
    # res_df['mistakes_percent'] = round(100 * (res_df.n_mistakes / res_df.n_units), 2)
    # res_df.to_csv(f'../data/experiments/{clf_name}__mistakes_by_experiment_excluded_tractate.tsv', sep='\t')

    diff_df = pd.read_csv(f'../data/experiments/{clf_name}/common/mistakes.tsv', sep='\t')
    diff_mistake_counts = diff_df.groupby(['talmud', 'tractate']).size()
    babli_indices = talmuds.talmud == 'bav'
    yerushalmi_indices = talmuds.talmud == 'jer'
    babli_tractates = talmuds[babli_indices]['tractate'].unique()
    yerushalmi_tractates = talmuds[yerushalmi_indices]['tractate'].unique()
    tractates = talmuds['tractate'].unique()
    common_tractates_a = np.intersect1d(babli_tractates, yerushalmi_tractates)
    different_tractates_a = np.setdiff1d(tractates, common_tractates_a)
    diff_unit_counts = talmuds[talmuds.tractate.isin(different_tractates_a)].groupby(['talmud', 'tractate']).size()
    diff_res = pd.concat([diff_mistake_counts, diff_unit_counts], axis=1)
    diff_res.columns = ['n_mistakes', 'n_units']
    diff_res['mistakes_percent'] = round(100 * (diff_res.n_mistakes / diff_res.n_units), 2)
    diff_res.to_csv(f'../data/experiments/{clf_name}__different_tractates_mistakes.tsv', sep='\t')


def main(analyze_only: bool):
    talmuds, _, _,targum = load_data()
    if not analyze_only:
        set_seed()
        create_experiments(talmuds, targum, full=True)
    #for clf_name in ('fasttext', 'cnt_nb_lgb'):
    # for clf_name in ('cnt_nb_lgb',):
    #     tractate_to_mistakes = count_mistakes(clf_name)
    #     analyze_mistakes(talmuds, clf_name, tractate_to_mistakes)


if __name__ == '__main__':
    main(analyze_only=False)
