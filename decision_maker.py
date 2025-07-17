def make_decision(predicted_return, threshold=0.005):
    if predicted_return > threshold:
        return "BUY"
    elif predicted_return < -threshold:
        return "SELL"
    else:
        return "HOLD"
