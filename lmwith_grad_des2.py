import pandas as pd

# Dosya yolunu kontrol edin
file_path = "datasets/advertising.csv"

# Veriyi yükleyin
try:
    df = pd.read_csv(file_path)
    print(df.head())
except FileNotFoundError:
    print("Dosya bulunamadı. Dosya yolunu kontrol edin.")
except pd.errors.EmptyDataError:
    print("Dosya boş.")
except Exception as e:
    print(f"Dosya okunurken bir hata oluştu: {e}")

# Cost Function
def cost_function(Y, b, w, X):
    m = len(Y)  # Number of observations
    sse = 0
    for i in range(m):
        y_hat = b + w * X.iloc[i]  # Predicted value
        y = Y.iloc[i]  # Actual value
        sse += (y_hat - y) ** 2  # Sum of squared errors
    mse = sse / m  # Mean Squared Error
    return mse

# Update Weights Function
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(m):
        y_hat = b + w * X.iloc[i]  # Predicted value
        y = Y.iloc[i]  # Actual value
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X.iloc[i]

    new_b = b - (learning_rate * b_deriv_sum / m)  # Update intercept
    new_w = w - (learning_rate * w_deriv_sum / m)  # Update weight

    return new_b, new_w

# Training Function
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                            cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):  # Iterative updates
        b, w = update_weights(Y, b, w, X, learning_rate)  # Update weights
        mse = cost_function(Y, b, w, X)  # Calculate new MSE
        cost_history.append(mse)

        if i % 100 == 0:  # Reporting every 100 iterations
            print("iter = {:d}, b = {:.2f}, w = {:.4f}, mse = {:.4f}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

# Eğer yukarıdaki dosya yükleme kısmı başarılı olduysa, eğitimi başlat
if 'df' in locals():
    # X ve Y değişkenlerini ayır
    X = df["radio"]
    Y = df["sales"]

    # Hyperparameters
    learning_rate = 0.001
    initial_b = 0.001
    initial_w = 0.001
    num_iters = 10000

    # Train the model
    cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
else:
    print("Veri yüklenemedi, eğitim başlatılamıyor.")
