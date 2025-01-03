'''Function to return a blend of kalman prediction and vision detection'''
# We assume probability represents the probability from vision network
def gradient(measured: tuple[float,float], predicted: tuple[float,float], probability: float) -> tuple[float,float]:

    if probability > 1 or probability < 0: 
        raise Exception(
            "Probability cannot be > 1 or < 0."
        )
    
    p_m = probability
    p_p = 1 - probability
    x, y = measured[0]*p_m + predicted[0]*p_p, measured[1]*p_m + predicted[1]*p_p
    return x, y