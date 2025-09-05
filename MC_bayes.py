import numpy as np

def matthews_correlation_coefficient(v):
    """
    Calcula el Coeficiente de Correlación de Matthews (MCC).
    v: un array o tupla de numpy con la forma (TP, TN, FP, FN).
    """
    #Asignacion de valores de confusion
    tp, tn, fp, fn = v
    # Evita la división por cero si alguno de los denominadores es 0
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0
    return (tp * tn - fp * fn) / denominator

def accuracy(v):
    """
    Calcula La exactitud de un clasificador.
    v: un array o tupla de numpy con la forma (TP, TN, FP, FN).
    """
    #Asignacion de valores de confusion
    tp, tn, fp, fn = v
    denominator = tp + tn + fp + fn 
    if denominator == 0:
        return 0
    return (tp + tn) / denominator

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


v_A = np.array([50, 35, 30, 30]) # [TP, TN, FP, FN]

# Parámetros del ejemplo
alpha_0 = np.array([0, 0, 0, 0]) # Prior no informativo
num_muestras = 1000000 # M, número de muestras a generar

# Generar las muestras para el clasificador A usando la función MCC
print(f"Generando {num_muestras} muestras para el clasificador")
g_bayes = mc_bayes_sampler(
    performance_indicator=matthews_correlation_coefficient,
    alpha=alpha_0,
    v=v_A,
    M=num_muestras
)

g_bootstrap= boot_sampler(
    performance_indicator= matthews_correlation_coefficient,
    v= v_A,
    B= num_muestras
)

# Imprimir estadísticas básicas de la distribución generada
print("\n--- Resultados para el Clasificador con Bayes ---")
print(f"Valor de MCC observado: {matthews_correlation_coefficient(v_A):.4f}")
print(f"Media de la distribución posterior de MCC: {np.mean(g_bayes):.4f}")
print(f"Desviación estándar de la distribución: {np.std(g_bayes):.4f}")
# Calcular el intervalo de confianza del 95% con bayes
cred_interval_bayes = np.percentile(g_bayes, [2.5, 97.5])
print(f"Intervalo de credibilidad del 95% para MCC: [{cred_interval_bayes[0]:.4f}, {cred_interval_bayes[1]:.4f}]")

#Calcular el intervalo de confianza del 95% con bootstrap
print("\n--- Resultados para el Clasificador con Bootstrap ---")
print(f"Media de la distribución posterior de MCC: {np.mean(g_bootstrap):.4f}")
print(f"Desviación estándar de la distribución: {np.std(g_bootstrap):.4f}")
cred_interval_bootstrap = np.percentile(g_bootstrap, [2.5, 97.5])
print(f"Intervalo de credibilidad del 95% para MCC: [{cred_interval_bootstrap[0]:.4f}, {cred_interval_bootstrap[1]:.4f}]")