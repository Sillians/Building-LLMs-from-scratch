from math import exp

def softmax(logits):
    # Calculate the exponent of each element in the input vector
    exponents = [exp(k) for k in logits]
    
    # Divide the exponent of each value by the sum of the exponents and 
    # round off to 3 decimal places
    sum_of_exponents = sum(exponents)
    probs = [round(i / sum_of_exponents, 3) for i in exponents]
    return probs

logits = [0.9, 0.5, 0.5, 0.8, 0.3]
print(softmax(logits))
