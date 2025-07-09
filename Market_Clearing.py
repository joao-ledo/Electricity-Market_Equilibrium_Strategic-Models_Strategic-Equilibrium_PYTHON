#******************************************************************************************************************************************************
#*                                                                 MARKET CLEARING MODEL                                                              *
#*************************************************************************************************-Developed by Joao Augusto Silva Ledo-***************
#* This code encompasses a simple market clearing model coded in PYTHON with focus on its coding structures such as its SETS and connections in pyomo *
#******************************************************************************************************************************************************

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 13:15:40 2025

@author: root
"""
import time
import os
from pyomo.environ import *
from pyomo.mpec import Complementarity, complements
from pyomo.opt import SolverFactory

class LoadInputData:  # method of the class accountable for creating its atributes
    def __init__(self, Cost, max_utility, p_max, d_max, o, b):
        self.Cost = Cost
        self.max_utility = max_utility
        self.p_max = p_max
        self.d_max = d_max
        self.o = o
        self.b = b
        
    def __repr__(self):  # method of the class accountable for returning all its atributes in a dynamic print
        return f"{self.__dict__}"
    
    def as_list(self):  # method of the class accountable for returning all its atributes in a dynamic list
        return list(self.__dict__.values())
    
    def __str__(self):  # method of the class accountable for returning all its atributes in a dynamic string
        return str(self.__dict__)
    
class ReturnOutputSolution:  # method of the class accountable for creating its outputs
    def __init__(self, Objective, p, d, Cleared_Price, profit, utility, SocialWelfare, Time, Solver_Name, Min_Max_Obj, Locally_or_NEOS_Server):
        self.Objective = Objective
        self.p = p
        self.d = d
        self.Cleared_Price = Cleared_Price
        self.profit = profit
        self.utility = utility
        self.SocialWelfare = SocialWelfare
        self.Computational_Time = Time
        self.Solver_Name = Solver_Name
        self.Min_Max_Obj = Min_Max_Obj
        self.Locally_or_NEOS_Server = Locally_or_NEOS_Server
        
    def __repr__(self):  # method of the class accountable for returning all its atributes in a dynamic print
        return f"{self.__dict__}"
    
    def as_list(self):  # method of the class accountable for returning all its atributes in a dynamic list
        return list(self.__dict__.values())
    
    def __str__(self):  # method of the class accountable for returning all its atributes in a dynamic string
        return str(self.__dict__)

def get_values_from_user(prompt, allowed_values):
    while True:
        try:
            value = int(input(prompt))
            if value not in allowed_values:
                print("\n PLEASE SELECT CARREFULLY! \n")
            else:
                return value
        except ValueError:
            print("\n Invalid input! {prompt}")
            
def Linear_or_Non_Linear():
        Selec_Linear_Non_Linear = get_values_from_user("\n Please select: \n 1 - Linear Solvers \n 2 - Non-linear Solver \n Type the value: ", list(range(1, 3)))
        if Selec_Linear_Non_Linear == 1:
            Select_Solver = get_values_from_user("\n Please select: \n 1 - glpk \n 2 - cbc \n 3 - highs \n 4 - cplex \n 5 - gurobi \n 6 - xpress \n 7 - scip \n Type the value: ", list(range(1, 8)))
            solver_name = LinearSolver[Select_Solver-1]
        else:
            Select_Solver = get_values_from_user("\n Please select: \n 1 - ipopt \n 2 - knitro \n 3 - conopt \n 4 - bonmin \n 5 - couenne \n 6 - baron \n 7 - scip \n Type the value: ", list(range(1, 8)))
            solver_name = Nonlinear[Select_Solver-1]
        return [Select_Solver, solver_name]

# Available Solvers
LinearSolver = ['glpk','cbc','highs','cplex','gurobi','xpress', 'scip']
Nonlinear = ['ipopt', 'knitro', 'conopt', 'bonmin', 'couenne', 'baron', 'scip']
    
# Create model
MarketClearingModel = ConcreteModel()

# Creates the Sets
MarketClearingModel.I = Set(initialize= [1, 2])  # Example with 2 producers
MarketClearingModel.U = Set(MarketClearingModel.I, initialize={
    1: [1],
    2: [2]
})
MarketClearingModel.IU = Set(dimen=2, initialize=lambda m: [(i, u) for i in m.I for u in m.U[i]])

MarketClearingModel.J = Set(initialize= [1])  # Example with 1 consumer
MarketClearingModel.C = Set(MarketClearingModel.J, initialize={
    1: [1]
})
MarketClearingModel.JC = Set(dimen=2, initialize=lambda m: [(j, c) for j in m.J for c in m.C[j]])

# Parameters
MarketClearingModel.Cost = Param(MarketClearingModel.IU, initialize={
    (1, 1): 1,
    (2, 2): 2
})  # Cost per unit for each producer-unit pair

MarketClearingModel.max_utility = Param(MarketClearingModel.IU, initialize={
    (1, 1): 3,
})  # Max utility per consumer for each consumer-demand pair

MarketClearingModel.p_max =  Param(MarketClearingModel.IU, initialize={(1, 1): 6, (2, 2):6})
                                                        
MarketClearingModel.d_max = Param(MarketClearingModel.JC, initialize={(1, 1): 10})

MarketClearingModel.o = {(i, u): MarketClearingModel.Cost[i, u] for i, u in MarketClearingModel.IU}

MarketClearingModel.b = {(j, c): MarketClearingModel.max_utility[j, c] for j, c in MarketClearingModel.JC}

# Variables
MarketClearingModel.p = Var(MarketClearingModel.IU, within=NonNegativeReals)
MarketClearingModel.d = Var(MarketClearingModel.JC, within=NonNegativeReals)

# Saving Model Input Data
ModelInput = LoadInputData({(i, u): MarketClearingModel.Cost[i, u] for i, u in MarketClearingModel.IU},
                           {(j, c): MarketClearingModel.max_utility[j, c] for j, c in MarketClearingModel.JC},
                           {(i, u): MarketClearingModel.p_max[i, u] for i, u in MarketClearingModel.IU},
                           {(j, c): MarketClearingModel.d_max[j, c] for j, c in MarketClearingModel.JC}, 
                           {(i, u): MarketClearingModel.o[i, u] for i, u in MarketClearingModel.IU},
                           {(j, c): MarketClearingModel.b[j, c] for j, c in MarketClearingModel.JC})

# Objective: Social Welfare
def Social_Welfare_objective(model):
    return sum(model.b[j, c] * model.d[j, c] for j, c in model.JC) - sum(model.o[i, u] * model.p[i, u] for i, u in model.IU)
MarketClearingModel.Objective = Objective(rule=Social_Welfare_objective, sense=maximize)

# Power Supply Balance Constraint
def Balance_Constraint(model):
    return sum(model.d[j, c] for j, c in model.JC) - sum(model.p[i, u] for i, u in model.IU) == 0
MarketClearingModel.Balance_Constraint = Constraint(rule=Balance_Constraint)

# Min power limit constraint for each unit
def min_power_constraint(model, i, u):
    return model.p[i, u] >= 0 # Min power for each unit
MarketClearingModel.min_power_constraint = Constraint(MarketClearingModel.IU, rule=min_power_constraint)

# Max power limit constraint for each unit
def max_power_constraint(model, i, u):
    return model.p[i, u] <= model.p_max[i, u] # Max power for each unit
MarketClearingModel.max_power_constraint = Constraint(MarketClearingModel.IU, rule=max_power_constraint)

# Min power limit constraint for each unit
def min_demand_constraint(model, j, c):
    return model.d[j, c] >= 0 # Min demand for each unit
MarketClearingModel.min_demand_constraint = Constraint(MarketClearingModel.JC, rule=min_demand_constraint)

# Max power limit constraint for each unit
def max_demand_constraint(model, j, c):
    return model.d[j, c] <= model.d_max[j, c] # Max demand for each unit
MarketClearingModel.max_demand_rule = Constraint(MarketClearingModel.JC, rule=max_demand_constraint)

# Add suffix for duals
MarketClearingModel.dual = Suffix(direction=Suffix.IMPORT)
    
Select_Solve_Locally_NEOS = get_values_from_user("\n Please select: \n 1 - Solve Locally \n 2 - Solve using NEOS Server \n Type the value: ", list(range(1, 3)))

if Select_Solve_Locally_NEOS == 1:
    (Select_Solver, solver_name) = Linear_or_Non_Linear()
else:
    os.environ['NEOS_EMAIL'] = input('\n Please provide your NEOS Server email: ')
    (Select_Solver, solver_name) = Linear_or_Non_Linear()

if Select_Solve_Locally_NEOS == 1:
    Locally_or_NEOS_Server = 'Locally'
    solver = SolverFactory(solver_name)
    # Start timer
    start_time_neos = time.time()
    
    # Solve model locally
    result = solver.solve(MarketClearingModel, tee=True)
    
    # End timer
    end_time_neos = time.time()
    
    # Compute elapsed time
    elapsed_time_neos = end_time_neos - start_time_neos
else:
    Locally_or_NEOS_Server = 'NEOS SERVER'
    neos_solver = SolverManagerFactory('neos')
    # Start timer
    start_time_neos = time.time()
    
    # Solve model using NEOS Server
    result = neos_solver.solve(MarketClearingModel,solver= solver_name, tee=True,load_solutions=True, suffixes=['dual'])
    
    # End timer
    end_time_neos = time.time()
    
    # Compute elapsed time
    elapsed_time_neos = end_time_neos - start_time_neos

cleared_price = MarketClearingModel.dual.get(MarketClearingModel.Balance_Constraint)
ModelOutput = ReturnOutputSolution(MarketClearingModel.Objective(),
                                   {(i, u): MarketClearingModel.p[i, u].value for i, u in MarketClearingModel.IU},
                                   {(j, c): MarketClearingModel.d[j, c].value for j, c in MarketClearingModel.JC},
                                   cleared_price,
                                   sum(cleared_price * MarketClearingModel.p[i, u].value - MarketClearingModel.Cost[i, u] * MarketClearingModel.p[i, u].value for i, u in MarketClearingModel.IU),
                                   sum(MarketClearingModel.max_utility[j, c]*MarketClearingModel.d[j, c].value - cleared_price*MarketClearingModel.d[j, c].value for j, c in MarketClearingModel.JC),
                                   sum(cleared_price * MarketClearingModel.p[i, u].value - MarketClearingModel.Cost[i, u] * MarketClearingModel.p[i, u].value for i, u in MarketClearingModel.IU) + sum(MarketClearingModel.max_utility[j, c]*MarketClearingModel.d[j, c].value - cleared_price*MarketClearingModel.d[j, c].value for j, c in MarketClearingModel.JC),
                                   elapsed_time_neos,
                                   solver_name,
                                   MarketClearingModel.Objective.sense.name,
                                   Locally_or_NEOS_Server)

print("\n", ModelInput, "\n")
print(ModelOutput, "\n")
