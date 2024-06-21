"""import pandas as pd


################################################################
#Simple Linear Regression with Gradient Descent from Scratch
#################################################################

#cost MSE değerini minimuma getirmeye çalışıyorduk. w veya b değerlerini değiştirerek hatanın minimum olduğu noktaya gitmeye çalışıyorduk

#Cost Function

def cost_function(Y, b, w, X):
    m = len(Y)  #m gözlem sayısı
    sse = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse /m
    return mse

def update_weights(Y, b, w, X, learning_rate):  #öğrenme oranı becerisi
    m = len(Y)

    b_deriv_sum = 0
    w_deriv_sum = 0

    for i in range(0, m):
        y_hat = b + w * X[i] #birinci gözlem birimi gerçek
        y = Y[i] # birinci gözlem birimi gerçek
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat -y) * X[i]

        new_b = b-(learning_rate * 1 / m * b_deriv_sum)
        new_w = w-(learning_rate * 1 / m * w_deriv_sum)

    return  new_b, new_w

def  train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w= {1}, mse={2}".format(initial_b, initial_w,
                                                                        cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters): # iterasyonların yani grafikteki güncellmem update işlemlerinin defalarca olarak w değerinin nereye gelmesi grektiğini itaratif olarak hareket ettirecek işlem
        b, w = update_weights(Y, b, w, X, learning_rate)  # güncelliyor yeni ağırlıklar gelir
        mse = cost_function(Y, b, w, X) # yeni ağırlıklara göre hata değerini de güncelleyip yeni mse yi lsiteye ekliyor
        cost_history.append(mse)

        if i % 100 == 0: # her 100 itearasyonda bir raporla bölümü
            print("iter={:d}  b={:.2f}  w={:.4f} mse={:.4}".format(i, b, w, mse)) # raporlama yapıyo
    print("After {0} iterations b={1}, w={2}, mse={3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w

file_path = "datasets/advertising.csv"
df = pd.read_csv(file_path)
print(df)

X = df["radio"]
Y = df["sales"]

#hyperparameters
learning_rate = 0.001
initial_b= 0.001
initial_w= 0.001
num_iters = 10000
#hiperparametreslerin hangisinin iyi olduğunu  deneme yanılma yoluyla bulmaya çalışacapız

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)"""