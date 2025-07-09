#********************************************************************************************************************************************************
#*                                                                 MPEC_z STRATEGIC CONSUMER                                                             *
#*************************************************************************************************-Developed by Joao Augusto Silva Ledo-*****************
#* This is a bilevel model (Stackelberg Game) which the upper-level embodies the strategic consumer and the lower-level the entire market clearing      *
#* that is replaced by 2 different optimality conditions:                                                                                               *
#* 1. The lower-level model is replaced by its KKT set of optimality conditions                                                                         *
#* 2. The lower-level model is replaced by its PDOC (Primal constraints, dual constraints, strong duality equality) set of optimality conditions        *
#* NOTE 1. THE KEY POINT IN THIS CODE ARE THE SUBSETS OF STRATEGIC AND NON-STRATEGIC PRODUCERS AND CONSUMERS                                            *
#********************************************************************************************************************************************************

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 16:22:53 2025

@author: root
"""

import time
import os
from pyomo.environ import *
from pyomo.mpec import Complementarity, complements
from pyomo.opt import SolverFactory

class LoadInputData:  # method of the class accountable for creating its atributes
    def __init__(self, Cost, max_utility, p_max, d_max, o, b_aux):
        self.Cost = Cost
        self.max_utility = max_utility
        self.p_max = p_max
        self.d_max = d_max
        self.o = o
        self.b_aux = b_aux
        
    def __repr__(self):  # method of the class accountable for returning all its atributes in a dynamic print
        return f"{self.__dict__}"
    
    def as_list(self):  # method of the class accountable for returning all its atributes in a dynamic list
        return list(self.__dict__.values())
    
    def __str__(self):  # method of the class accountable for returning all its atributes in a dynamic string
        return str(self.__dict__)
    
class ReturnOutputSolution:  # method of the class accountable for creating its outputs
    def __init__(self, MCP, Strategic_Consumer, Objective, p, d, Strategic_b, Cleared_Price, profit, utility, SocialWelfare, mu_p_min, mu_p_max, mu_d_min, mu_d_max, Time, Solver_Name, Min_Max_Obj, Locally_or_NEOS_Server):
        self.Lower_level_MCP = MCP
        self.Strategic_Consumer = Strategic_Consumer
        self.Objective = Objective
        self.p = p
        self.d = d
        self.Strategic_b = Strategic_b
        self.Cleared_Price = Cleared_Price
        self.profit = profit
        self.utility = utility
        self.SocialWelfare = SocialWelfare
        self.mu_p_min = mu_p_min
        self.mu_p_max = mu_p_max
        self.mu_d_min = mu_d_min
        self.mu_d_max = mu_d_max
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
            Select_Solver = get_values_from_user("\n Please select: \n 1 - ipopt \n 2 - knitro \n 3 - conopt \n 4 - bonmin \n 5 - couenne \n 6 - baron \n 7 - snopt \n Type the value: ", list(range(1, 8)))
            solver_name = Nonlinear[Select_Solver-1]
        return [Select_Solver, solver_name]

# Available Solvers
LinearSolver = ['glpk','cbc','highs','cplex','gurobi','xpress', 'scip']
Nonlinear = ['ipopt', 'knitro', 'conopt', 'bonmin', 'couenne', 'baron', 'snopt']

#['baron', 'bonmin', 'cbc', 'conopt', 'couenne', 'cplex', 'filmint', 'filter', 'ipopt', 'knitro', 'l-bfgs-b', 'lancelot', 'lgo', 'loqo', 'minlp', 'minos', 'minto', 'mosek', 'ooqp', 'path', 'raposa', 'snopt']
# Create model
BilevelStrategicConsumerModel = ConcreteModel()

# Creates the Sets
BilevelStrategicConsumerModel.I = Set(initialize= [1, 2])  # Example with 2 producers
BilevelStrategicConsumerModel.U = Set(BilevelStrategicConsumerModel.I, initialize={
    1: [1],
    2: [2]
})
BilevelStrategicConsumerModel.IU = Set(dimen=2, initialize=lambda m: [(i, u) for i in m.I for u in m.U[i]])

BilevelStrategicConsumerModel.J = Set(initialize= [1])  # Example with 1 consumer
BilevelStrategicConsumerModel.C = Set(BilevelStrategicConsumerModel.J, initialize={
    1: [1]
})
BilevelStrategicConsumerModel.JC = Set(dimen=2, initialize=lambda m: [(j, c) for j in m.J for c in m.C[j]])


Strategic_Consumer = get_values_from_user(f"\n Please select the strategic consumer in {list(BilevelStrategicConsumerModel.J)}: ", BilevelStrategicConsumerModel.J.data())

# Set of Strategic Producers
BilevelStrategicConsumerModel.Z = Set(initialize = [Strategic_Consumer])
BilevelStrategicConsumerModel.ZC = Set(dimen=2, initialize=lambda m: [(z, c) for z in m.Z for c in m.C[z]])

# Set of Non-Strategic Consumer
BilevelStrategicConsumerModel.L = Set(initialize = lambda m: [j for j in m.J if j != Strategic_Consumer])
BilevelStrategicConsumerModel.LC = Set(dimen=2, initialize=lambda m: [(l, c) for l in m.L for c in m.C[l]])

# Parameters
BilevelStrategicConsumerModel.Cost = Param(BilevelStrategicConsumerModel.IU, initialize={
    (1, 1): 1,
    (2, 2): 2
})  # Cost per unit for each producer-unit pair

BilevelStrategicConsumerModel.max_utility = Param(BilevelStrategicConsumerModel.IU, initialize={
    (1, 1): 3,
})  # Max utility per consumer for each consumer-demand pair

BilevelStrategicConsumerModel.p_max =  Param(BilevelStrategicConsumerModel.IU, 
                                   initialize={(1, 1): 6, 
                                               (2, 2): 6}
                                   )
                                                        
BilevelStrategicConsumerModel.d_max = Param(BilevelStrategicConsumerModel.JC, initialize={(1, 1): 10})

BilevelStrategicConsumerModel.o = {(i, u): BilevelStrategicConsumerModel.Cost[i, u] for i, u in BilevelStrategicConsumerModel.IU}

BilevelStrategicConsumerModel.b_aux = {(j, c): BilevelStrategicConsumerModel.max_utility[j, c] for j, c in BilevelStrategicConsumerModel.JC}

# Variables
BilevelStrategicConsumerModel.p = Var(BilevelStrategicConsumerModel.IU, within=NonNegativeReals)
BilevelStrategicConsumerModel.d = Var(BilevelStrategicConsumerModel.JC, within=NonNegativeReals)
BilevelStrategicConsumerModel.b = Var(BilevelStrategicConsumerModel.ZC, within=NonNegativeReals)
BilevelStrategicConsumerModel.Lambda = Var()
BilevelStrategicConsumerModel.mu_p_min = Var(BilevelStrategicConsumerModel.IU, within=NonNegativeReals)
BilevelStrategicConsumerModel.mu_p_max = Var(BilevelStrategicConsumerModel.IU, within=NonNegativeReals)
BilevelStrategicConsumerModel.mu_d_min = Var(BilevelStrategicConsumerModel.JC, within=NonNegativeReals)
BilevelStrategicConsumerModel.mu_d_max = Var(BilevelStrategicConsumerModel.JC, within=NonNegativeReals)

# Starting value
#BilevelStrategicConsumerModel.Lambda.value = 2

# Saving Model Input Data
ModelInput = LoadInputData({(i, u): BilevelStrategicConsumerModel.Cost[i, u] for i, u in BilevelStrategicConsumerModel.IU},
                           {(j, c): BilevelStrategicConsumerModel.max_utility[j, c] for j, c in BilevelStrategicConsumerModel.JC},
                           {(i, u): BilevelStrategicConsumerModel.p_max[i, u] for i, u in BilevelStrategicConsumerModel.IU},
                           {(j, c): BilevelStrategicConsumerModel.d_max[j, c] for j, c in BilevelStrategicConsumerModel.JC}, 
                           {(i, u): BilevelStrategicConsumerModel.o[i, u] for i, u in BilevelStrategicConsumerModel.IU},
                           {(l, c): BilevelStrategicConsumerModel.b_aux[l, c] for l, c in BilevelStrategicConsumerModel.LC})


# Objective: Social Welfare
def Utility_objective(model):
    return sum(model.max_utility[z, c]*model.d[z, c] - model.d[z, c]*model.Lambda for z, c in model.ZC)
BilevelStrategicConsumerModel.Objective = Objective(rule=Utility_objective, sense=maximize)

# Upper-level constraints *
def Strategic_Bidding_Boundary_constraint(model, z, c):
    return model.b[z, c] <= model.max_utility[z, c]
BilevelStrategicConsumerModel.Strategic_Bidding_Boundary_constraint = Constraint(BilevelStrategicConsumerModel.ZC, rule=Strategic_Bidding_Boundary_constraint)

# Power Supply Balance Constraint
def Balance_Constraint(model):
    return sum(model.d[j, c] for j, c in model.JC) - sum(model.p[i, u] for i, u in model.IU) == 0
BilevelStrategicConsumerModel.Balance_Constraint = Constraint(rule=Balance_Constraint)

# Min power limit constraint for each unit
def min_power_constraint(model, i, u):
    return model.p[i, u] >= 0 # Min power for each unit
BilevelStrategicConsumerModel.min_power_constraint = Constraint(BilevelStrategicConsumerModel.IU, rule=min_power_constraint)

# Max power limit constraint for each unit
def max_power_constraint(model, i, u):
    return model.p[i, u] <= model.p_max[i, u] # Max power for each unit
BilevelStrategicConsumerModel.max_power_constraint = Constraint(BilevelStrategicConsumerModel.IU, rule=max_power_constraint)

# Min power limit constraint for each unit
def min_demand_constraint(model, j, c):
    return model.d[j, c] >= 0 # Min demand for each unit
BilevelStrategicConsumerModel.min_demand_constraint = Constraint(BilevelStrategicConsumerModel.JC, rule=min_demand_constraint)

# Max power limit constraint for each unit
def max_demand_constraint(model, j, c):
    return model.d[j, c] <= model.d_max[j, c] # Max demand for each unit
BilevelStrategicConsumerModel.max_demand_constraint = Constraint(BilevelStrategicConsumerModel.JC, rule=max_demand_constraint)

def deriv_p_NS_constraint(model, i, u):
    return model.o[i, u] - model.Lambda - model.mu_p_min[i, u] + model.mu_p_max[i, u] == 0
BilevelStrategicConsumerModel.deriv_p_NS_constraint = Constraint(BilevelStrategicConsumerModel.IU, rule=deriv_p_NS_constraint)

def deriv_d_S_constraint(model, z, c):
    return - model.b[z, c] + model.Lambda - model.mu_d_min[z, c] + model.mu_d_max[z, c] == 0
BilevelStrategicConsumerModel.deriv_d_S_constraint = Constraint(BilevelStrategicConsumerModel.ZC, rule=deriv_d_S_constraint)

def deriv_d_NS_constraint(model, l, c):
    return - model.b_aux[l, c] + model.Lambda - model.mu_d_min[l, c] + model.mu_d_max[l, c] == 0
BilevelStrategicConsumerModel.deriv_d_NS_constraint = Constraint(BilevelStrategicConsumerModel.LC, rule=deriv_d_NS_constraint)


def positivity_p_min_constraint(model, i, u):
    return model.mu_p_min[i, u] >= 0
BilevelStrategicConsumerModel.positivity_p_min_constraint = Constraint(BilevelStrategicConsumerModel.IU, rule=positivity_p_min_constraint)

def positivity_p_max_constraint(model, i, u):
    return model.mu_p_max[i, u] >= 0
BilevelStrategicConsumerModel.positivity_p_max_constraint = Constraint(BilevelStrategicConsumerModel.IU, rule=positivity_p_max_constraint)

def positivity_d_min_constraint(model, j, c):
    return model.mu_d_min[j, c] >= 0
BilevelStrategicConsumerModel.positivity_d_min_constraint = Constraint(BilevelStrategicConsumerModel.JC, rule=positivity_d_min_constraint)

def positivity_d_max_constraint(model, j, c):
    return model.mu_d_max[j, c] >= 0
BilevelStrategicConsumerModel.positivity_d_max_constraint = Constraint(BilevelStrategicConsumerModel.JC, rule=positivity_d_max_constraint)

MCP_KKT_or_PDOC = get_values_from_user("\n Please select: \n 1 - KKT MCP Market Clearing model \n 2 - PDOC MCP Market Clearing model \n Type the value: ", list(range(1, 3)))

if MCP_KKT_or_PDOC == 1:
    MCP = 'KKT MCP'
    #Complementarities
    def comp_p_min_constraint(model, i, u):
        return model.p[i, u]*model.mu_p_min[i, u] == 0
    BilevelStrategicConsumerModel.comp_p_min_constraint = Constraint(BilevelStrategicConsumerModel.IU, rule=comp_p_min_constraint)
    
    def comp_p_max_constraint(model, i, u):
        return model.p_max[i, u]*model.mu_p_max[i, u] - model.p[i, u]*model.mu_p_max[i, u] == 0
    BilevelStrategicConsumerModel.comp_p_max_constraint = Constraint(BilevelStrategicConsumerModel.IU, rule=comp_p_max_constraint)
    
    def comp_d_min_constraint(model, j, c):
        return model.d[j, c]*model.mu_d_min[j, c] == 0
    BilevelStrategicConsumerModel.comp_d_min_constraint = Constraint(BilevelStrategicConsumerModel.JC, rule=comp_d_min_constraint)
    
    def comp_d_max_constraint(model, j, c):
        return model.d_max[j, c]*model.mu_d_max[j, c] - model.d[j, c]*model.mu_d_max[j, c] == 0
    BilevelStrategicConsumerModel.comp_d_max_constraint = Constraint(BilevelStrategicConsumerModel.JC, rule=comp_d_max_constraint)
else:
    MCP = 'PDOC MCP'
    def SDE_constraint(model):
        return sum(model.o[i, u]*model.p[i, u] for i, u in model.IU) -  sum(model.b[z, c]*model.d[z, c] for z, c in model.ZC) -  sum(model.b_aux[l, c]*model.d[l, c] for l, c in model.LC) == - sum(model.mu_p_max[i, u]*model.p_max[i, u] for i, u in model.IU) - sum(model.mu_d_max[j, c]*model.d_max[j, c] for j, c in model.JC)
    BilevelStrategicConsumerModel.SDE_constraint = Constraint(rule=SDE_constraint)
    
# Add suffix for duals
BilevelStrategicConsumerModel.dual = Suffix(direction=Suffix.IMPORT)
    
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
    result = solver.solve(BilevelStrategicConsumerModel, tee=True)
    
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
    result = neos_solver.solve(BilevelStrategicConsumerModel,solver= solver_name, tee=True,load_solutions=True, suffixes=['dual'])
    
    # End timer
    end_time_neos = time.time()
    
    # Compute elapsed time
    elapsed_time_neos = end_time_neos - start_time_neos

cleared_price = BilevelStrategicConsumerModel.Lambda.value
ModelOutput = ReturnOutputSolution(MCP,
                                   Strategic_Consumer,
                                   BilevelStrategicConsumerModel.Objective(),
                                   {(i, u): BilevelStrategicConsumerModel.p[i, u].value for i, u in BilevelStrategicConsumerModel.IU},
                                   {(j, c): BilevelStrategicConsumerModel.d[j, c].value for j, c in BilevelStrategicConsumerModel.JC},
                                   {(z, c): BilevelStrategicConsumerModel.b[z, c].value for z, c in BilevelStrategicConsumerModel.ZC},
                                   cleared_price,
                                   sum(cleared_price * BilevelStrategicConsumerModel.p[i, u].value - BilevelStrategicConsumerModel.Cost[i, u] * BilevelStrategicConsumerModel.p[i, u].value for i, u in BilevelStrategicConsumerModel.IU),
                                   sum(BilevelStrategicConsumerModel.max_utility[j, c]*BilevelStrategicConsumerModel.d[j, c].value - cleared_price*BilevelStrategicConsumerModel.d[j, c].value for j, c in BilevelStrategicConsumerModel.JC),
                                   sum(cleared_price * BilevelStrategicConsumerModel.p[i, u].value - BilevelStrategicConsumerModel.Cost[i, u] * BilevelStrategicConsumerModel.p[i, u].value for i, u in BilevelStrategicConsumerModel.IU) + sum(BilevelStrategicConsumerModel.max_utility[j, c]*BilevelStrategicConsumerModel.d[j, c].value - cleared_price*BilevelStrategicConsumerModel.d[j, c].value for j, c in BilevelStrategicConsumerModel.JC),
                                   {(i, u): BilevelStrategicConsumerModel.mu_p_min[i, u].value for i, u in BilevelStrategicConsumerModel.IU}, 
                                   {(i, u): BilevelStrategicConsumerModel.mu_p_max[i, u].value for i, u in BilevelStrategicConsumerModel.IU}, 
                                   {(j, c): BilevelStrategicConsumerModel.mu_d_min[j, c].value for j, c in BilevelStrategicConsumerModel.JC}, 
                                   {(j, c): BilevelStrategicConsumerModel.mu_d_max[j, c].value for j, c in BilevelStrategicConsumerModel.JC}, 
                                   elapsed_time_neos,
                                   solver_name,
                                   BilevelStrategicConsumerModel.Objective.sense.name,
                                   Locally_or_NEOS_Server)

print("\n", ModelInput, "\n")
print(ModelOutput, "\n")