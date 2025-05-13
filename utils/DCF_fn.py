import numpy as np

def compute_op_bayes_decisions(pi, C_fn, C_fp, llr_vec):
    num = pi * C_fn
    den = (1-pi)*C_fp
    #t = -np.log(num/den)
    t = 0
    
    pred_vec = []
    
    for llr in llr_vec:
        if llr > t:
            pred_vec.append(1)
        else:
            pred_vec.append(0)
    return pred_vec

def compute_DCF_u(pi, C_fn, C_fp, llr, L):
    pred_vec = compute_op_bayes_decisions(pi, C_fn, C_fp, llr)
    conf_matrix = np.zeros((2,2), dtype = np.int16)
    # 'k' and 'm' are respectively the row and the column where the counter should be updated
    for i in range(len(pred_vec)):
        k = pred_vec[i]
        m = L[i]
        conf_matrix[k][m] += 1
    
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[1][0]
    FN = conf_matrix[0][1]
    
    P_fn = FN/(FN + TP)
    P_fp = FP/(FP + TN)
    
    DCF_u = pi*C_fn*P_fn + (1-pi)*C_fp*P_fp
    # round DFU
    DCF_u = float(int(DCF_u*1000))/1000
    return DCF_u



# normalized DCF
def compute_den_normalized_DCF(pi, C_fn, C_fp, llr, L):
    pred_vec = compute_op_bayes_decisions(pi, C_fn, C_fp, llr)
    conf_matrix = np.zeros((2,2), dtype = np.int16)
    # 'k' and 'm' are respectively the row and the column where the counter should be updated
    for i in range(len(pred_vec)):
        k = pred_vec[i]
        m = L[i]
        conf_matrix[k][m] += 1
    
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[1][0]
    FN = conf_matrix[0][1]
    
    P_fn = FN/(FN + TP)
    P_fp = FP/(FP + TN)
    
    den_DCF = min(pi*C_fn, (1-pi)*C_fp)
    den_DCF = float(int(den_DCF*1000))/1000
    return den_DCF


def compute_DCF(pi, C_fn, C_fp, llr_vec, L, t):
    
    # COMPUTE PREDICTIONS
    
    pred_vec = []
    
    for llr in llr_vec:
        if llr > t:
            pred_vec.append(1)
        else:
            pred_vec.append(0)
    
    # COMPUTE CONFUSION MATRIX
        
    conf_matrix = compute_confusion_matrix(pred_vec, L)
    
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[1][0]
    FN = conf_matrix[0][1]
    
    P_fn = FN/(FN + TP)
    P_fp = FP/(FP + TN)
    
    DCF_u = pi*C_fn*P_fn + (1-pi)*C_fp*P_fp
    DCF_den = compute_den_normalized_DCF(pi, C_fn, C_fp, llr_vec, L)
    DCF = DCF_u/DCF_den
    DCF = float(int(DCF*1000))/1000
    return DCF



def compute_confusion_matrix(pred_vec, L):
    conf_matrix = np.zeros((2,2), dtype = np.int32)
    # 'k' and 'm' are respectively the row and the column where the counter should be updated
    for i in range(len(pred_vec)):
        k = pred_vec[i]
        m = L[i]
        conf_matrix[k][m] += 1

    return conf_matrix


def compute_actual_DCF(pi, C_fn, C_fp, llr_vec, L):
    DCF_u = compute_DCF_u(pi, C_fn, C_fp, llr_vec, L)
    DCF_den = compute_den_normalized_DCF(pi, C_fn, C_fp, llr_vec, L)
    DCF = DCF_u/DCF_den
    # round DFU
    actDCF = float(int(DCF*1000))/1000
    
    return DCF


def compute_min_DCF(pi, C_fn, C_fp, llr, L, t_vec): 
    # trying every threshold
    temp_DCF_vec = []
    for t in t_vec:
        DCF = compute_DCF(pi, C_fn, C_fp, llr, L, t)
        temp_DCF_vec.append(DCF)
    
    # getting the minimum DCF computed
    min_DCF = min(temp_DCF_vec)
    return min_DCF