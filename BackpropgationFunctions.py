def MSE(actual, predicted):
    return ((actual - predicted) ** 2)

def MSE_Gradient(actual,predicted):
    return (predicted - actual)