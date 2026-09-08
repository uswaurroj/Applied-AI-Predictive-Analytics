import math
import random

def generate_multivariate_data(num_samples=1000):
    """Generates synthetic multi-variable features and continuous targets."""
    dataset = []
    for _ in range(num_samples):
        x1 = random.uniform(10, 100)
        x2 = random.uniform(1, 50)
        # Target with underlying line y = 2.5*x1 + 4.0*x2 + noise
        noise = random.gauss(0, 5)
        y = (2.5 * x1) + (4.0 * x2) + 15.0 + noise
        dataset.append({"x1": x1, "x2": x2, "y": y})
    return dataset

def train_predictive_model(data):
    """Simple Multivariate Ordinary Least Squares (OLS) closed-form regression."""
    n = len(data)
    
    # Calculate feature means
    mean_x1 = sum(d["x1"] for d in data) / n
    mean_x2 = sum(d["x2"] for d in data) / n
    mean_y = sum(d["y"] for d in data) / n
    
    # Compute coefficients
    num_x1 = sum((d["x1"] - mean_x1) * (d["y"] - mean_y) for d in data)
    den_x1 = sum((d["x1"] - mean_x1) ** 2 for d in data)
    w1 = num_x1 / den_x1
    
    num_x2 = sum((d["x2"] - mean_x2) * (d["y"] - mean_y) for d in data)
    den_x2 = sum((d["x2"] - mean_x2) ** 2 for d in data)
    w2 = num_x2 / den_x2
    
    bias = mean_y - (w1 * mean_x1 + w2 * mean_x2)
    
    # Evaluation Metrics
    predictions = [w1 * d["x1"] + w2 * d["x2"] + bias for d in data]
    actuals = [d["y"] for d in data]
    
    mse = sum((act - pred) ** 2 for act, pred in zip(actuals, predictions)) / n
    rmse = math.sqrt(mse)
    mae = sum(abs(act - pred) for act, pred in zip(actuals, predictions)) / n
    
    ss_tot = sum((act - mean_y) ** 2 for act in actuals)
    ss_res = sum((act - pred) ** 2 for act, pred in zip(actuals, predictions))
    r2_score = 1 - (ss_res / ss_tot)
    
    print("--- Applied AI Predictive Analytics Pipeline Output ---")
    print(f"Dataset Records Processed: {n}")
    print(f"Model Parameters: Weights = [{w1:.4f}, {w2:.4f}], Bias = {bias:.4f}")
    print(f"Mean Absolute Error (MAE) : {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE) : {rmse:.4f}")
    print(f"R-Squared (R2 Score)      : {r2_score:.4f}")

if __name__ == "__main__":
    records = generate_multivariate_data(num_samples=1500)
    train_predictive_model(records)