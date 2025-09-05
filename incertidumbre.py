#funciones

import numpy as np
#from sklearn.metrics import f1_score as sklearn_f1_score

def f1_score(v):
    """
    Calcula el F1-Score desde un vector [TP, TN, FP, FN].
    """
    tp, tn, fp, fn = v
    if (tp + fp == 0) or (tp + fn == 0):
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall == 0):
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def mc_bayes_sampler(performance_indicator, alpha, v, M):
    """
    Genera muestras aleatorias de un indicador de rendimiento utilizando Monte Carlo.

    Args:
        performance_indicator (function): Una función que toma un vector v
                                          y calcula una métrica (ej. MCC).
        alpha (np.array): El vector con el conocimiento a priori.
        v (np.array): El vector con los valores de la matriz de confusión observada
                      (#TP, #TN, #FP, #FN).
        M (int): El número de muestras a generar.

    Returns:
        np.array: Un vector con M muestras del indicador de rendimiento.
    """
    # omega es la suma de los vectores alpha (a priori) y v (la evidencia) 
    omega = np.add(v, alpha)
    
    # El número total de observaciones en la matriz de confusión original
    N_t = np.sum(v)
    
    g = [] # Lista para almacenar los resultados
    
    for _ in range(M):
        #Muestrear θ desde la distribución de Dirichlet posterior
        theta_m = np.random.dirichlet(omega)
        
        # Muestrear una nueva matriz de confusión v_tilde desde una
        # distribución multinomial usando el theta_m muestreado
        v_tilde_m = np.random.multinomial(N_t, theta_m)
        
        # Calcular el indicador de rendimiento para la nueva matriz
        g_m = performance_indicator(v_tilde_m)
        g.append(g_m)
        
    return np.array(g)

def boot_sampler(performance_indicator, v, B):
    """
    Genera muestras aleatorias de un indicador de rendimiento usando bootstrap.

    Args:
        performance_indicator (function): Una función que toma un vector v
                                          y calcula una métrica (ej. MCC).
        v (np.array): El vector con los valores de la matriz de confusión observada
                      (#TP, #TN, #FP, #FN).
        B (int): El número de muestras de bootstrap a generar.

    Returns:
        np.array: Un vector con B muestras del indicador de rendimiento.
    """
    tp, tn, fp, fn = v
    
    # Crear el vector M con los resultados individuales
    # (TP=0, TN=1, FP=2, FN=3)
    M_vector = np.concatenate([
        np.zeros(tp),      # TPs
        np.ones(tn),       # TNs
        np.full(fp, 2),  # FPs
        np.full(fn, 3)   # FNs
    ])
    
    N_t = len(M_vector) # Número total de observaciones
    g_star = [] # Lista para almacenar los resultados
    
    for _ in range(B):
        # Muestrear con reemplazo desde M 
        # Se generan N_t índices aleatorios y se usan para crear M_star
        indices = np.random.randint(0, N_t, N_t)
        M_star = M_vector[indices]
        
        # Agregar M_star para obtener una nueva matriz de confusión V_star
        tp_star = np.sum(M_star == 0)
        tn_star = np.sum(M_star == 1)
        fp_star = np.sum(M_star == 2)
        fn_star = np.sum(M_star == 3)
        V_star = np.array([tp_star, tn_star, fp_star, fn_star])
        
        #calcular el indicador de rendimiento y almacenarlo
        g_b = performance_indicator(V_star)
        g_star.append(g_b)
        
    return np.array(g_star)