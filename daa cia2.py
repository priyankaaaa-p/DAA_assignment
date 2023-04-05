#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
data = pd.get_dummies(data, columns=["Education", "Family", "CCAvg", "Online", "CreditCard"])

#train test split
X = data.drop(["Personal Loan", "ID"], axis=1)
y = data["Personal Loan"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#define nn
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
net = Net(input_size=X_train.shape[1], hidden_size=10, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(100):
    optimizer.zero_grad()
    outputs = net(X_train.float())
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, 100, loss.item()))

#genetic algorithm
from geneticalgorithm import geneticalgorithm as ga

def fitness_func(params):
    net = Net(input_size=X_train.shape[1], hidden_size=params[0], num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=params[1])
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = net(X_train.float())
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    outputs = net(X_test.float())
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size()[0]
    return accuracy

varbound = [[1, 50], [0.0001, 0.1]]
model = ga(function=fitness_func, dimension=2, variable_type='real', variable_boundaries=varbound)
model.run()

#pso
from pyswarm import pso

def fitness_func(params):
    net = Net(input_size=X_train.shape[1], hidden_size=int(params[0]), num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=params[1])
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = net(X_train.float())
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    outputs = net(X_test.float())
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size()[0]
    return -accuracy

lb = [1, 0.0001]
ub = [50, 0.1]
xopt, fopt = pso(fitness_func, lb, ub)
print("Best fitness:", -fopt)
print("Best params:", xopt)

#ant colony optimization
import ant_colony

class NetAntColonyProblem(ant_colony.Problem):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self, solution):
        net = Net(input_size=self.X_train.shape[1], hidden_size=int(solution[0]), num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=solution[1])
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = net(self.X_train.float())
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
        outputs = net(self.X_test.float())
        _, predicted
        accuracy = (predicted == self.y_test).sum().item() / self.y_test.size()[0]
        return -accuracy

    def get_bounds(self):
        return [(1, 50), (0.0001, 0.1)]

problem = NetAntColonyProblem(X_train, y_train, X_test, y_test)
aco = ant_colony.ACO(problem, num_ants=10, num_iterations=50, alpha=1, beta=2, evaporation_rate=0.5, pheromone_deposit=0.5)
solution = aco.run()
print("Best fitness:", -problem.evaluate(solution))
print("Best params:", solution)

#cultural algorithms
import numpy as np
from pygmo import algorithm, problem, population, island

class NetProblem:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def fitness(self, x):
        net = Net(input_size=self.X_train.shape[1], hidden_size=int(x[0]), num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=x[1])
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = net(self.X_train.float())
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
        outputs = net(self.X_test.float())
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == self.y_test).sum().item() / self.y_test.size()[0]
        return [-accuracy]

    def get_bounds(self):
        return ([1, 0.0001], [50, 0.1])

prob = problem(NetProblem(X_train, y_train, X_test, y_test))
pop = population(prob, size=10)
algo = algorithm.cultural(island(algo=algorithm.pso(gen=100)), udi=10, cr=0.3)
pop = algo.evolve(pop)

best_fitness = pop.champion_f[0]
best_params = pop.champion_x
print("Best fitness:", -best_fitness)
print("Best params:", best_params)

