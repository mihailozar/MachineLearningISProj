import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.linear_model import LinearRegression

datas = pd.read_csv('datasets/car_purchase.csv')

print(datas.head())

print(datas.iloc[-5:])

print(datas.describe())

fig1,axs1=plt.subplots(3,2)

axs1[0,0].scatter(datas["credit_card_debt"],datas["max_purchase_amount"],10, c='red', marker='o', alpha=0.7)
axs1[0,0].set_xlabel("credit_card_debt", color="red")
axs1[0,0].set_ylabel("max_purchase_amount", color="black")

axs1[1,0].scatter(datas["net_worth"],datas["max_purchase_amount"],10, c='blue', marker='o', alpha=0.7)
axs1[1,0].set_xlabel("net_worth", color="blue")
axs1[1,0].set_ylabel("max_purchase_amount", color="black")


axs1[2,0].scatter(datas["annual_salary"],datas["max_purchase_amount"],10, c='green', marker='o', alpha=0.7)
axs1[2,0].set_xlabel("annual_salary", color="green")
axs1[2,0].set_ylabel("max_purchase_amount", color="black")

axs1[0,1].scatter(datas["gender"],datas["max_purchase_amount"],10, c='pink', marker='o', alpha=0.7)
axs1[0,1].set_xlabel("gender", color="pink")
axs1[0,1].set_ylabel("max_purchase_amount", color="black")

axs1[1,1].scatter(datas["customer_id"],datas["max_purchase_amount"],10, c='purple', marker='o', alpha=0.7)
axs1[1,1].set_xlabel("customer_id", color="purple")
axs1[1,1].set_ylabel("max_purchase_amount", color="black")


axs1[2,1].scatter(datas["age"],datas["max_purchase_amount"],10, c='orange', marker='o', alpha=0.7)
axs1[2,1].set_xlabel("age", color="orange")
axs1[2,1].set_ylabel("max_purchase_amount", color="black")




fig, ax = plt.subplots()
twin1 = ax.twinx()
twin2 = ax.twinx()


ax.scatter( datas["credit_card_debt"]/10000,datas["max_purchase_amount"]/100000, s=23, c='red', marker='o', alpha=0.7, label='credit_card_debit')
twin1.scatter(datas["net_worth"]/500000,datas["max_purchase_amount"]/100000,s=23,c='blue', marker='o', alpha=0.7,label='net_worth')
twin2.scatter(datas["annual_salary"]/50000,datas["max_purchase_amount"]/100000,s=23,c='orange', marker='o', alpha=0.7, label='annual_salary')
# twin3.scatter(datas["max_purchase_amount"], datas["gender"], s=23, c='yellow', marker='o', alpha=0.7)
# twin3.scatter(datas["max_purchase_amount"], datas["customer_id"], s=23, c='green', marker='o', alpha=0.7)
# twin4.scatter(datas["max_purchase_amount"], datas["age"], s=23, c='black', marker='o', alpha=0.7)


class LinearRegressionGradientDescent:
    def __init__(self):
        self.coeffs = None
        self.features = None
        self.target = None
        self.mse_history = None

    def set_coefficients(self, *args):
        # Mapiramo koeficijente u niz oblika (n + 1) x 1
        self.coeff = np.array(args).reshape(-1, 1)

    def cost(self):
        predicted = self.features.dot(self.coeff)
        s = pow(predicted - self.target, 2).sum()
        return (0.5 / len(self.features)) * s

    def predict(self, features):
        features = features.copy(deep=True)
        features.insert(0, 'c0', np.ones((len(features), 1)))
        features = features.to_numpy()
        return features.dot(self.coeff).reshape(-1, 1).flatten()

    def gradient_descent_step(self, learning_rate):
        # learning_rate - korak ucenja; dimenzije ((n + 1) x 1);
        # korak ucenja je razlicit za razlicite koeficijente
        # m - broj uzoraka
        # n - broj razlicitih atributa (osobina)
        # features – dimenzije (m x (n + 1));
        # n + 1 je zbog koeficijenta c0
        # self.coeff – dimenzije ((n + 1) x 1)
        # predicted – dimenzije (m x (n + 1)) x ((n + 1) x 1) = (m x 1)
        predicted = self.features.dot(self.coeff)

        s = self.features.T.dot(predicted - self.target)
        gradient = (1. / len(self.features)) * s
        self.coeff = self.coeff - learning_rate * gradient
        return self.coeff, self.cost()

    def perform_gradient_descent(self, learning_rate, num_iterations=100):
        # Istorija Mean-square error-a kroz iteracije gradijentnog spusta.
        self.mse_history = []
        for i in range(num_iterations):
            _, curr_cost = self.gradient_descent_step(learning_rate)
            self.mse_history.append(curr_cost)
        return self.coeff, self.mse_history

    def fit(self, features, target):
        self.features = features.copy(deep=True)
        # Pocetna vrednost za koeficijente je 0.
        # self.coeff - dimenzije ((n + 1) x 1)
        coeff_shape = len(features.columns) + 1
        self.coeff = np.zeros(shape=coeff_shape).reshape(-1, 1)
        # Unosi se kolona jedinica za koeficijent c0,
        # kao da je vrednost atributa uz c0 jednaka 1.
        self.features.insert(0, 'c0', np.ones((len(features), 1)))
        # self.features - dimenzije (m x (n + 1))
        self.features = self.features.to_numpy()
        # self.target - dimenzije (m x 1)
        self.target = target.to_numpy().reshape(-1, 1)
from sklearn.model_selection import train_test_split


X = datas.loc[:, ['credit_card_debt', 'annual_salary',"net_worth"]]/10000
X.loc[:,["annual_salary"]]=X.loc[:,["annual_salary"]]/5
X.loc[:,["net_worth"]]=X.loc[:,["net_worth"]]/50

y = datas['max_purchase_amount']/100000

featuresTrainSet, featuresTestSet, targetTrainSet, targetTestSet = train_test_split(X,y, test_size=0.25, random_state=1)






spots = 200
estates = pd.DataFrame(data=[np.linspace(0, max(featuresTrainSet['credit_card_debt']), num=spots),np.linspace(0, max(featuresTrainSet['annual_salary']), num=spots), np.linspace(0, max(featuresTrainSet['net_worth']), num=spots)  ] )
# Kreiranje i obucavanje modela
estates=estates.T

lrgd = LinearRegressionGradientDescent()
lrgd.fit(featuresTrainSet, targetTrainSet)
learning_rates = np.array([[0.5], [0.2],[0.5],[0.5]])
res_coeff, mse_history = lrgd.perform_gradient_descent(learning_rates, 50)

# Vizuelizacija modela
pred=lrgd.predict(estates)
ax.plot(estates[0], lrgd.predict(estates), lw=5, c='red' )

lr_model = LinearRegression()
lr_model.fit(featuresTrainSet.values, targetTrainSet)
# Vizuelizacija modela
ax.plot(estates[0], lr_model.predict(estates), lw=2, c='blue')

plt.figure("MS Error")
plt.plot(np.arange(0,len(mse_history),1),mse_history)
plt.xlabel('Iteration', fontsize=13)
plt.ylabel("MS error value", fontsize=13)
plt.xticks(np.arange(0,len(mse_history),2))
plt.title('Mean-square error finction')
# plt.legend(['MS Error'])


max_purchase_amount=42000
net_worth=326373
credit_card_debt=5958
annual_salary=55421
example_estate=pd.DataFrame(data=[[net_worth],[credit_card_debt], [annual_salary]])
example_estate=example_estate.T

lrgd.set_coefficients(res_coeff)
print(f'For LGRD inputs  {net_worth}, {credit_card_debt}, {annual_salary} max_purchase_amount is {lrgd.predict(example_estate)[0]:.2f} ')
print(f'LGRD c0: {lrgd.coeff.flatten()[0]:.2f}, c1: {lrgd.coeff.flatten()[1]:.2f}, c2: {lrgd.coeff.flatten()[2]:.2f}, c3:  {lrgd.coeff.flatten()[3]:.2f}')
print(f'For LR inputs {net_worth}, {credit_card_debt}, {annual_salary} max_purchase_amount is {lr_model.predict(example_estate)[0]:.2f}')
print(f'LGRD c0: {lr_model.coef_[0]:.2f}, c1: {lr_model.coef_[1]:.2f}, c2: {lr_model.coef_[2]:.2f}')

lrgd.set_coefficients(res_coeff)
print(f'LRGD MSE: {lrgd.cost():f}')
c=np.concatenate((np.array([lr_model.intercept_]),lr_model.coef_))
lrgd.set_coefficients(c)
print(f'LR MSE: {lrgd.cost():f}')

lrgd.set_coefficients(res_coeff)

lr_coef_ = lr_model.coef_
lr_int_ = lr_model.intercept_
lr_model.coef_ = lrgd.coeff.flatten()[1:]
lr_model.intercept_ = lrgd.coeff.flatten()[0]
print(f'LRGD score: {lr_model.score(featuresTestSet.values, targetTestSet):.2f}')

lr_model.coef_ = lr_coef_
lr_model.intercept_ = lr_int_
print(f'LR score: {lr_model.score(featuresTestSet.values, targetTestSet):.2f}')



# plt.legend()
plt.tight_layout()
plt.show()