import pickle
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics import auc, roc_curve
from eval.f1 import compute_exact_match
from util.metrics import getRouge, getSentenceSimilarity
rougeEvaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def getPCC(x, y):
    rho = np.corrcoef(np.array(x), np.array(y))
    return rho[0,1]

def VisAUROC(a, b, c, d, e=''):
    pass

def get_threshold(thresholds, tpr, fpr):
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits = 4)
    return thresholdOpt

def computeAUROC(Label, Prediction, _print, file_name=''):
    fpr, tpr, thresholds = roc_curve(Label, Prediction)
    AUROC = auc(fpr, tpr)
    # thresh_EigenScore = thresholds[np.argmax(tpr - fpr)]
    thresh_EigenScore = get_threshold(thresholds, tpr, fpr)
    print(_print, AUROC)
    # print("thresh_EigenScore:", thresh_EigenScore)
    VisAUROC(tpr, fpr, AUROC, "EigenScore", file_name.split("_")[1])
    pass

def getAUROC(file_name, args):
    f = open(file_name, 'rb')
    resultDict = pickle.load(f)

    if args.use_roberta:
        SenSimModel = SentenceTransformer('nli-roberta-large')

    Label = []
    Score = []
    Perplexity = []
    Energy = []
    LexicalSimilarity = []
    SentBertScore = []
    Entropy = []
    EigenIndicator = []
    EigenIndicatorOutput = []
    EigenIndicator_All_Layers = []
    EigenIndicatorOutput_Seperate_Layers = []

    for item in resultDict:
        ansGT = item["answer"]
        generations = item["most_likely_generation"]
        # print("GT:", ansGT)
        # print("Generation:", generations)
        Perplexity.append(-item["perplexity"])
        Energy.append(-item["energy"])
        # Entropy.append(-item["entropy"])
        LexicalSimilarity.append(item["lexical_similarity"])
        SentBertScore.append(-item["sent_bertscore"])
        EigenIndicator.append(-item["eigenIndicator"])
        EigenIndicatorOutput.append(-item["eigenIndicatorOutput"])
        EigenIndicatorOutput_Seperate_Layers.append([-i for i in item["eigenIndicator_all_layer"]])
        EigenIndicator_All_Layers.append([-item["eigenIndicator_v3"]])

        if args.use_roberta:
            similarity = getSentenceSimilarity(generations, ansGT, SenSimModel)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                similarities = [getSentenceSimilarity(generations, ansGT, SenSimModel) for ansGT in additional_answers]
                similarity = max(similarity, max(similarities))
            if similarity>0.9:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(similarity)
        elif args.use_exact_match:
            similarity = compute_exact_match(generations, ansGT)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                similarities = [compute_exact_match(generations, ansGT) for ansGT in additional_answers]
                similarity = max(similarity, max(similarities))
            if similarity==1:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(similarity)
        else:
            rougeScore = getRouge(rougeEvaluator, generations, ansGT)
            if "coqa" in file_name or "TruthfulQA" in file_name:
                additional_answers = item["additional_answers"]
                rougeScores = [getRouge(rougeEvaluator, generations, ansGT) for ansGT in additional_answers]
                rougeScore = max(rougeScore, max(rougeScores))
            if rougeScore>0.5:
                Label.append(1)
            else:
                Label.append(0)
            Score.append(rougeScore)


######### 计算AUROC ###########

    computeAUROC(Label, Energy, "AUROC-Energy")

    computeAUROC(Label, Perplexity, "AUROC-Perplexity")

    computeAUROC(Label, Score, "AUROC-Similarity")

    computeAUROC(Label, LexicalSimilarity, "AUROC-LexicalSim")

    computeAUROC(Label, SentBertScore, "AUROC-SentBertScore")

    computeAUROC(Label, EigenIndicator, "AUROC-EigenIndicator")

    computeAUROC(Label, EigenIndicatorOutput, "AUROC-EigenScore-Output")

    computeAUROC(Label, EigenIndicator_All_Layers, "AUROC-EigenIndicator_All_Layers")


    row_counts = [len(arr) for arr in EigenIndicatorOutput_Seperate_Layers]

    unique_row_counts = list(set(row_counts)) # 去重并排序
    print(unique_row_counts)

    if len(unique_row_counts) == 1:
        split_arrays = {row: [] for row in range(unique_row_counts[0])}

        for arr in EigenIndicatorOutput_Seperate_Layers:
            for i in range(len(arr)):
                split_arrays[i].append(arr[i])

        for row in range(unique_row_counts[0]):
            split_arrays[row] = np.array(split_arrays[row])
            computeAUROC(Label, split_arrays[row], "AUROC-EigenIndicatorOutput_Seperate_Layers_"+str(row), file_name)


    


    # fpr, tpr, thresholds = roc_curve(Label, Entropy)
    # AUROC = auc(fpr, tpr)
    # thresh_Entropy = thresholds[np.argmax(tpr - fpr)]
    # thresh_Entropy = get_threshold(thresholds, tpr, fpr)
    # print("AUROC-Entropy:", AUROC)
    # print("thresh_Entropy:", thresh_Entropy)
    # VisAUROC(tpr, fpr, AUROC, "NormalizedEntropy")




######## 计算皮尔逊相关系数 ###############
    rho_Perplexity = getPCC(Score, Perplexity)
    # rho_Entropy = getPCC(Score, Entropy)
    rho_Energy = getPCC(Score, Energy)
    rho_LexicalSimilarity = getPCC(Score, LexicalSimilarity)
    rho_EigenIndicator = getPCC(Score, EigenIndicator)
    rho_EigenIndicatorOutput = getPCC(Score, EigenIndicatorOutput)
    print("rho_Perplexity:", rho_Perplexity)
    print("rho_Energy:", rho_Energy)
    # print("rho_Entropy:", rho_Entropy)
    print("rho_LexicalSimilarity:", rho_LexicalSimilarity)
    print("rho_EigenScore:", rho_EigenIndicator)
    print("rho_EigenScoreOutput:", rho_EigenIndicatorOutput)



# ######### 计算幻觉检测准确率(TruthfulQA)
#     if "TruthfulQA" in file_name:
#         acc = getTruthfulQAAccuracy(Label, Perplexity, thresh_Perplexity)
#         print("TruthfulQA Perplexity Accuracy:", acc)
#         acc = getTruthfulQAAccuracy(Label, Energy, thresh_Energy)
#         print("TruthfulQA Energy Accuracy:", acc)
#         acc = getTruthfulQAAccuracy(Label, Entropy, thresh_Entropy)
#         print("TruthfulQA Entropy Accuracy:", acc)
#         acc = getTruthfulQAAccuracy(Label, LexicalSimilarity, thresh_LexicalSim)
#         print("TruthfulQA LexicalSimilarity Accuracy:", acc)
#         acc = getTruthfulQAAccuracy(Label, SentBertScore, thresh_SentBertScore)
#         print("TruthfulQA SentBertScore Accuracy:", acc)
#         acc = getTruthfulQAAccuracy(Label, EigenIndicator, thresh_EigenScore)
#         print("TruthfulQA EigenIndicator Accuracy:", acc)
#         acc = getTruthfulQAAccuracy(Label, EigenIndicatorOutput, thresh_EigenScoreOutput)
#         print("TruthfulQA EigenIndicatorOutput Accuracy:", acc)

