#********************************************************************************************************************************************************
#*                                                                  EPEC PDOC DIAGONALIZATION                                                           *
#*************************************************************************************************-Developed by Joao Augusto Silva Ledo-*****************
#* This is an EPEC model achieved by diagonalization of multiples strategic bilevel models in equilibrium where the upper-level embodies the            *
#* strategic producer or strategic consumer and theirs respetive lower-level embodies the entire market clearing model, therefore creating an           *
#* Multi-leader-single-follower equilibrium. Each of the respetively lower-level models are replaced by its PDOC optimality conditions and solved by    *
#* an approach called diagolalization                                                                                                                   *
#* NOTE 1. THE KEY POINT IN THIS CODE ARE THE SUBSETS OF STRATEGIC AND NON-STRATEGIC PRODUCERS AND CONSUMERS ITERATIVELY SOLVED                         *
#********************************************************************************************************************************************************

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 17:48:58 2025

@author: root
"""

import time
import os
from pyomo.environ import *
from pyomo.mpec import Complementarity, complements
from pyomo.opt import SolverFactory

##########################################################################
# MAIN
##########################################################################
def main():
    Start_Strategic_Producer_or_StrategicConsumer = get_values_from_user("\n Please select which market agent do you want to start the diagonalization method as strategic: \n 1 - Strategic Producer \n 2 - Strategic Consumer \n Type the value: ", list(range(1, 3)))
    print("\n Strategic Producer Input")
    StrategicProducer_ModelInput = LoadInputData_function()
    print("\n Strategic Consumer Input")
    StrategicConsumer_ModelInput = LoadInputData_function()
    print("\n Solver Configuration")
    Config = Solve_Config()
    
    BilevelStrategicProducerModel = Creates_BilevelStrategicProducerModel(StrategicProducer_ModelInput, 1)
    BilevelStrategicConsumerModel = Creates_BilevelStrategicConsumerModel(StrategicConsumer_ModelInput, 1)
    
    (BilevelStrategicProducerModel, StrategicProducer_ModelInput, StrategicProducer_OutputSolution, BilevelStrategicConsumerModel, StrategicConsumer_ModelInput, StrategicConsumer_OutputSolution, counting) = Diagonalization(Config, BilevelStrategicProducerModel, StrategicProducer_ModelInput, BilevelStrategicConsumerModel, StrategicConsumer_ModelInput, Start_Strategic_Producer_or_StrategicConsumer)
    
    print("\n", StrategicProducer_ModelInput, "\n")
    print(StrategicProducer_OutputSolution, "\n")
    print("\n", StrategicConsumer_ModelInput, "\n")
    print(StrategicConsumer_OutputSolution, "\n")
    print(f"Number of iterations {counting}")


##########################################################################
# LOAD INPUT DATA FUNCTION
##########################################################################
def LoadInputData_function():
    ModelInput = DataStorageClass('Input Data')
    ModelInput.Set_I = [1, 2]
    ModelInput.Set_U = {1: [1],2: [2]}
    ModelInput.Set_J = [1]
    ModelInput.Set_C = {1: [1]}
    ModelInput.Cost = {(1, 1): 1, (2, 2): 2}
    ModelInput.max_utility = {(1, 1): 3,}
    ModelInput.p_max = {(1, 1): 6,(2, 2): 6}
    ModelInput.d_max = {(1, 1): 10}
    ModelInput.o_aux = ModelInput.Cost
    ModelInput.b_aux = ModelInput.max_utility
    ModelInput.MCP_KKT_or_PDOC = get_values_from_user("\n Please select: \n 1 - KKT MCP Market Clearing model \n 2 - PDOC MCP Market Clearing model \n Type the value: ", list(range(1, 3)))
    return ModelInput

##########################################################################
# RETURN OUTPUT SOLUTION
##########################################################################
def ReturnOutputSolution_function(Model, Time, solver_name, Locally_or_NEOS_Server, Strategic_Producer_or_StrategicConsumer, Strategic_Agent):
    cleared_price = Model.Lambda.value
    OutputSolution = DataStorageClass('Output Data')
    OutputSolution.Objective = Model.Objective()
    OutputSolution.p = {(i, u): Model.p[i, u].value for i, u in Model.IU}
    OutputSolution.d = {(j, c): Model.d[j, c].value for j, c in Model.JC}
    if Strategic_Producer_or_StrategicConsumer == 1:
        OutputSolution.Strategic_o = {(y, u): Model.o[y, u].value for y, u in Model.YU}
        OutputSolution.Strategic_Producer = Strategic_Agent
    else:
        OutputSolution.Strategic_b = {(z, c): Model.b[z, c].value for z, c in Model.ZC}
        OutputSolution.Strategic_Consumer = Strategic_Agent
    OutputSolution.Cleared_Price = cleared_price
    OutputSolution.profit = sum(cleared_price * Model.p[i, u].value - Model.Cost[i, u] * Model.p[i, u].value for i, u in Model.IU)
    OutputSolution.utility = sum(Model.max_utility[j, c]*Model.d[j, c].value - cleared_price*Model.d[j, c].value for j, c in Model.JC)
    OutputSolution.SocialWelfare = sum(cleared_price * Model.p[i, u].value - Model.Cost[i, u] * Model.p[i, u].value for i, u in Model.IU) + sum(Model.max_utility[j, c]*Model.d[j, c].value - cleared_price*Model.d[j, c].value for j, c in Model.JC)
    OutputSolution.mu_p_min = {(i, u): Model.mu_p_min[i, u].value for i, u in Model.IU}
    OutputSolution.mu_p_max = {(i, u): Model.mu_p_max[i, u].value for i, u in Model.IU}
    OutputSolution.mu_d_min = {(j, c): Model.mu_d_min[j, c].value for j, c in Model.JC}
    OutputSolution.mu_d_max = {(j, c): Model.mu_d_max[j, c].value for j, c in Model.JC}
    OutputSolution.Computational_Time = Time
    OutputSolution.Solver_Name = solver_name
    OutputSolution.Min_Max_Obj = Model.Objective.sense.name
    OutputSolution.Locally_or_NEOS_Server = Locally_or_NEOS_Server
    return OutputSolution
    
##########################################################################
# CLASS DEFINITION
##########################################################################
class DataStorageClass:  # method of the class accountable for creating its atributes
    def __init__(self, Name):
        self.Name = Name
        
    def __repr__(self):  # method of the class accountable for returning all its atributes in a dynamic print
        return f"{self.__dict__}"
    
    def as_list(self):  # method of the class accountable for returning all its atributes in a dynamic list
        return list(self.__dict__.values())
    
    def __str__(self):  # method of the class accountable for returning all its atributes in a dynamic string
        return str(self.__dict__)
    
##########################################################################
# CHOOSE OPTION FUNCTIONS
########################################################################## 
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
    # Available Solvers
    #['baron', 'bonmin', 'cbc', 'conopt', 'couenne', 'cplex', 'filmint', 'filter', 'ipopt', 'knitro', 'l-bfgs-b', 'lancelot', 'lgo', 'loqo', 'minlp', 'minos', 'minto', 'mosek', 'ooqp', 'path', 'raposa', 'snopt']
    LinearSolver = ['glpk','cbc','highs','cplex','gurobi','xpress', 'scip']
    Nonlinear = ['ipopt', 'knitro', 'conopt', 'bonmin', 'couenne', 'baron', 'snopt']
    Selec_Linear_Non_Linear = get_values_from_user("\n Please select: \n 1 - Linear Solvers \n 2 - Non-linear Solver \n Type the value: ", list(range(1, 3)))
    if Selec_Linear_Non_Linear == 1:
        Select_Solver = get_values_from_user("\n Please select: \n 1 - glpk \n 2 - cbc \n 3 - highs \n 4 - cplex \n 5 - gurobi \n 6 - xpress \n 7 - scip \n Type the value: ", list(range(1, 8)))
        solver_name = LinearSolver[Select_Solver-1]
    else:
        Select_Solver = get_values_from_user("\n Please select: \n 1 - ipopt \n 2 - knitro \n 3 - conopt \n 4 - bonmin \n 5 - couenne \n 6 - baron \n 7 - snopt \n Type the value: ", list(range(1, 8)))
        solver_name = Nonlinear[Select_Solver-1]
    return solver_name


##########################################################################
# CONSTRAINTS FUNCTIONS
##########################################################################        
# Objective: Strategic Procuer Profit
def Profit_objective(model):
    return sum(model.Lambda*model.p[y, u] - model.Cost[y, u]*model.p[y, u] for y, u in model.YU)

# Upper-level constraints *
def Strategic_Offering_Boundary_constraint(model, y, u):
    return model.o[y, u] >= model.Cost[y, u]

# Objective: Strategic Consumer Utility
def Utility_objective(model):
    return sum(model.max_utility[z, c]*model.d[z, c] - model.d[z, c]*model.Lambda for z, c in model.ZC)

# Upper-level constraints *
def Strategic_Bidding_Boundary_constraint(model, z, c):
    return model.b[z, c] <= model.max_utility[z, c]

# * Lower-level primal constraints
# Power Supply Balance Constraint
def Balance_Constraint(model):
    return sum(model.d[j, c] for j, c in model.JC) - sum(model.p[i, u] for i, u in model.IU) == 0

# Min power limit constraint for each unit
def min_power_constraint(model, i, u):
    return model.p[i, u] >= 0 # Min power for each unit

# Max power limit constraint for each unit
def max_power_constraint(model, i, u):
    return model.p[i, u] <= model.p_max[i, u] # Max power for each unit

# Min power limit constraint for each unit
def min_demand_constraint(model, j, c):
    return model.d[j, c] >= 0 # Min demand for each unit

# Max power limit constraint for each unit
def max_demand_constraint(model, j, c):
    return model.d[j, c] <= model.d_max[j, c] # Max demand for each unit

# * Lower-level partial derivatives (dual constraints)
def deriv_p_S_constraint(model, y, u):
    return model.o[y, u] - model.Lambda - model.mu_p_min[y, u] + model.mu_p_max[y, u] == 0

def deriv_p_NS_constraint(model, k, u):
    return model.o_aux[k, u] - model.Lambda - model.mu_p_min[k, u] + model.mu_p_max[k, u] == 0

def deriv_d_S_constraint(model, z, c):
    return - model.b[z, c] + model.Lambda - model.mu_d_min[z, c] + model.mu_d_max[z, c] == 0

def deriv_d_NS_constraint(model, j, c):
    return - model.b_aux[j, c] + model.Lambda - model.mu_d_min[j, c] + model.mu_d_max[j, c] == 0

# * Dual variables positivity
def positivity_p_min_constraint(model, i, u):
    return model.mu_p_min[i, u] >= 0

def positivity_p_max_constraint(model, i, u):
    return model.mu_p_max[i, u] >= 0

def positivity_d_min_constraint(model, j, c):
    return model.mu_d_min[j, c] >= 0

def positivity_d_max_constraint(model, j, c):
    return model.mu_d_max[j, c] >= 0

# * Lower-level complementary constraints

def comp_p_min_constraint(model, i, u):
    return model.p[i, u]*model.mu_p_min[i, u] == 0

def comp_p_max_constraint(model, i, u):
    return model.p_max[i, u]*model.mu_p_max[i, u] - model.p[i, u]*model.mu_p_max[i, u] == 0

def comp_d_min_constraint(model, j, c):
    return model.d[j, c]*model.mu_d_min[j, c] == 0

def comp_d_max_constraint(model, j, c):
    return model.d_max[j, c]*model.mu_d_max[j, c] - model.d[j, c]*model.mu_d_max[j, c] == 0

def SDE_constraint_Strategic_Producer(model):
    return sum(model.o[y, u]*model.p[y, u] for y, u in model.YU) + sum(model.o_aux[k, u]*model.p[k, u] for k, u in model.KU) -  sum(model.b_aux[j, c]*model.d[j, c] for j, c in model.JC) == - sum(model.mu_p_max[i, u]*model.p_max[i, u] for i, u in model.IU) - sum(model.mu_d_max[j, c]*model.d_max[j, c] for j, c in model.JC)

def SDE_constraint_Strategic_Consumer(model):
    return sum(model.o_aux[i, u]*model.p[i, u] for i, u in model.IU) -  sum(model.b[z, c]*model.d[z, c] for z, c in model.ZC) -  sum(model.b_aux[l, c]*model.d[l, c] for l, c in model.LC) == - sum(model.mu_p_max[i, u]*model.p_max[i, u] for i, u in model.IU) - sum(model.mu_d_max[j, c]*model.d_max[j, c] for j, c in model.JC)   

##########################################################################
# CREATES THE MODEL FUNCTIONS
########################################################################## 
def KKT_or_PDOC(model, Strategic_Producer_or_StrategicConsumer, MCP_KKT_or_PDOC):
    if MCP_KKT_or_PDOC == 1:
        MCP = 'KKT MCP'
        model.comp_p_min_constraint = Constraint(model.IU, rule=comp_p_min_constraint)    
        model.comp_p_max_constraint = Constraint(model.IU, rule=comp_p_max_constraint)    
        model.comp_d_min_constraint = Constraint(model.JC, rule=comp_d_min_constraint)    
        model.comp_d_max_constraint = Constraint(model.JC, rule=comp_d_max_constraint)
    else:
        MCP = 'PDOC MCP'
        if Strategic_Producer_or_StrategicConsumer == 1:
            model.SDE_constraint = Constraint(rule=SDE_constraint_Strategic_Producer)
        else:
            model.SDE_constraint = Constraint(rule=SDE_constraint_Strategic_Consumer)
    return (model, MCP)

def Creates_BilevelStrategicProducerModel(ModelInput, Strategic_Producer):
    BilevelStrategicProducerModel = ConcreteModel()
    
    # Creates the Sets
    # Set of all Producers
    BilevelStrategicProducerModel.I = Set(initialize= ModelInput.Set_I)  # Example with 2 producers
    BilevelStrategicProducerModel.U = Set(BilevelStrategicProducerModel.I, initialize= ModelInput.Set_U)
    BilevelStrategicProducerModel.IU = Set(dimen=2, initialize=lambda m: [(i, u) for i in m.I for u in m.U[i]])
    
    # Set of Strategic Producers
    BilevelStrategicProducerModel.Y = Set(initialize = [Strategic_Producer])
    BilevelStrategicProducerModel.YU = Set(dimen=2, initialize=lambda m: [(y, u) for y in m.Y for u in m.U[y]])
    
    # Set of Non-Strategic Producers
    BilevelStrategicProducerModel.K = Set(initialize = lambda m: [i for i in m.I if i != Strategic_Producer])
    BilevelStrategicProducerModel.KU = Set(dimen=2, initialize=lambda m: [(k, u) for k in m.K for u in m.U[k]])
    
    # Set of all Consumers
    BilevelStrategicProducerModel.J = Set(initialize= ModelInput.Set_J)  # Example with 1 consumer
    BilevelStrategicProducerModel.C = Set(BilevelStrategicProducerModel.J, initialize= ModelInput.Set_C)
    BilevelStrategicProducerModel.JC = Set(dimen=2, initialize=lambda m: [(j, c) for j in m.J for c in m.C[j]])
    
    # Parameters
    BilevelStrategicProducerModel.Cost = Param(BilevelStrategicProducerModel.IU, initialize= ModelInput.Cost)  # Cost per unit for each producer-unit pair
    BilevelStrategicProducerModel.max_utility = Param(BilevelStrategicProducerModel.JC, initialize= ModelInput.max_utility)  # Max utility per consumer for each consumer-demand pair
    BilevelStrategicProducerModel.p_max =  Param(BilevelStrategicProducerModel.IU,initialize= ModelInput.p_max)
    BilevelStrategicProducerModel.d_max = Param(BilevelStrategicProducerModel.JC, initialize= ModelInput.d_max)
    BilevelStrategicProducerModel.o_aux = {(i, u): BilevelStrategicProducerModel.Cost[i, u] for i, u in BilevelStrategicProducerModel.IU}
    BilevelStrategicProducerModel.b_aux = {(j, c): BilevelStrategicProducerModel.max_utility[j, c] for j, c in BilevelStrategicProducerModel.JC}
    
    # Variables
    BilevelStrategicProducerModel.p = Var(BilevelStrategicProducerModel.IU, within=NonNegativeReals)
    BilevelStrategicProducerModel.d = Var(BilevelStrategicProducerModel.JC, within=NonNegativeReals)
    BilevelStrategicProducerModel.o = Var(BilevelStrategicProducerModel.YU, within=NonNegativeReals)
    BilevelStrategicProducerModel.Lambda = Var()
    BilevelStrategicProducerModel.mu_p_min = Var(BilevelStrategicProducerModel.IU, within=NonNegativeReals)
    BilevelStrategicProducerModel.mu_p_max = Var(BilevelStrategicProducerModel.IU, within=NonNegativeReals)
    BilevelStrategicProducerModel.mu_d_min = Var(BilevelStrategicProducerModel.JC, within=NonNegativeReals)
    BilevelStrategicProducerModel.mu_d_max = Var(BilevelStrategicProducerModel.JC, within=NonNegativeReals)
    
    # Starting value
    #BilevelStrategicProducerModel.Lambda.value = 2
        
    # Add constraints to the model
    BilevelStrategicProducerModel.Objective = Objective(rule=Profit_objective, sense=maximize)
    BilevelStrategicProducerModel.Strategic_Offering_Boundary_constraint = Constraint(BilevelStrategicProducerModel.YU, rule=Strategic_Offering_Boundary_constraint)
    BilevelStrategicProducerModel.Balance_Constraint = Constraint(rule=Balance_Constraint)
    BilevelStrategicProducerModel.min_power_constraint = Constraint(BilevelStrategicProducerModel.IU, rule=min_power_constraint)
    BilevelStrategicProducerModel.max_power_constraint = Constraint(BilevelStrategicProducerModel.IU, rule=max_power_constraint)
    BilevelStrategicProducerModel.min_demand_constraint = Constraint(BilevelStrategicProducerModel.JC, rule=min_demand_constraint)
    BilevelStrategicProducerModel.max_demand_constraint = Constraint(BilevelStrategicProducerModel.JC, rule=max_demand_constraint)
    BilevelStrategicProducerModel.deriv_p_S_constraint = Constraint(BilevelStrategicProducerModel.YU, rule=deriv_p_S_constraint)
    BilevelStrategicProducerModel.deriv_p_NS_constraint = Constraint(BilevelStrategicProducerModel.KU, rule=deriv_p_NS_constraint)
    BilevelStrategicProducerModel.deriv_d_NS_constraint = Constraint(BilevelStrategicProducerModel.JC, rule=deriv_d_NS_constraint)
    BilevelStrategicProducerModel.positivity_p_min_constraint = Constraint(BilevelStrategicProducerModel.IU, rule=positivity_p_min_constraint)
    BilevelStrategicProducerModel.positivity_p_max_constraint = Constraint(BilevelStrategicProducerModel.IU, rule=positivity_p_max_constraint)
    BilevelStrategicProducerModel.positivity_d_min_constraint = Constraint(BilevelStrategicProducerModel.JC, rule=positivity_d_min_constraint)
    BilevelStrategicProducerModel.positivity_d_max_constraint = Constraint(BilevelStrategicProducerModel.JC, rule=positivity_d_max_constraint)
    # Choose between KKT or PDOC and add to the model the respectively KKT or PDOC constraints
    Strategic_Producer_or_StrategicConsumer = 1
    (BilevelStrategicProducerModel, ModelInput.MCP) = KKT_or_PDOC(BilevelStrategicProducerModel, Strategic_Producer_or_StrategicConsumer, ModelInput.MCP_KKT_or_PDOC)
            
    # Add suffix for duals
    BilevelStrategicProducerModel.dual = Suffix(direction=Suffix.IMPORT)

    return BilevelStrategicProducerModel

def Creates_BilevelStrategicConsumerModel(ModelInput, Strategic_Consumer):
    BilevelStrategicConsumerModel = ConcreteModel()
    
    # Creates the Sets
    # Set of all Producers
    BilevelStrategicConsumerModel.I = Set(initialize= ModelInput.Set_I)  # Example with 2 producers
    BilevelStrategicConsumerModel.U = Set(BilevelStrategicConsumerModel.I, initialize= ModelInput.Set_U)
    BilevelStrategicConsumerModel.IU = Set(dimen=2, initialize=lambda m: [(i, u) for i in m.I for u in m.U[i]])
    
    # Set of all Consumers
    BilevelStrategicConsumerModel.J = Set(initialize= ModelInput.Set_J)  # Example with 1 consumer
    BilevelStrategicConsumerModel.C = Set(BilevelStrategicConsumerModel.J, initialize= ModelInput.Set_C)
    BilevelStrategicConsumerModel.JC = Set(dimen=2, initialize=lambda m: [(j, c) for j in m.J for c in m.C[j]])
    
    # Set of Strategic Consumer
    BilevelStrategicConsumerModel.Z = Set(initialize = [Strategic_Consumer])
    BilevelStrategicConsumerModel.ZC = Set(dimen=2, initialize=lambda m: [(z, c) for z in m.Z for c in m.C[z]])

    # Set of Non-Strategic Consumer
    BilevelStrategicConsumerModel.L = Set(initialize = lambda m: [j for j in m.J if j != Strategic_Consumer])
    BilevelStrategicConsumerModel.LC = Set(dimen=2, initialize=lambda m: [(l, c) for l in m.L for c in m.C[l]])
    
    # Parameters
    BilevelStrategicConsumerModel.Cost = Param(BilevelStrategicConsumerModel.IU, initialize= ModelInput.Cost)  # Cost per unit for each producer-unit pair
    BilevelStrategicConsumerModel.max_utility = Param(BilevelStrategicConsumerModel.JC, initialize= ModelInput.max_utility)  # Max utility per consumer for each consumer-demand pair
    BilevelStrategicConsumerModel.p_max =  Param(BilevelStrategicConsumerModel.IU,initialize= ModelInput.p_max)
    BilevelStrategicConsumerModel.d_max = Param(BilevelStrategicConsumerModel.JC, initialize= ModelInput.d_max)
    BilevelStrategicConsumerModel.o_aux = {(i, u): BilevelStrategicConsumerModel.Cost[i, u] for i, u in BilevelStrategicConsumerModel.IU}
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
        
    # Add constraints to the model
    BilevelStrategicConsumerModel.Objective = Objective(rule=Utility_objective, sense=maximize)
    BilevelStrategicConsumerModel.Strategic_Bidding_Boundary_constraint = Constraint(BilevelStrategicConsumerModel.ZC, rule=Strategic_Bidding_Boundary_constraint)
    BilevelStrategicConsumerModel.Balance_Constraint = Constraint(rule=Balance_Constraint)
    BilevelStrategicConsumerModel.min_power_constraint = Constraint(BilevelStrategicConsumerModel.IU, rule=min_power_constraint)
    BilevelStrategicConsumerModel.max_power_constraint = Constraint(BilevelStrategicConsumerModel.IU, rule=max_power_constraint)
    BilevelStrategicConsumerModel.min_demand_constraint = Constraint(BilevelStrategicConsumerModel.JC, rule=min_demand_constraint)
    BilevelStrategicConsumerModel.max_demand_constraint = Constraint(BilevelStrategicConsumerModel.JC, rule=max_demand_constraint)
    BilevelStrategicConsumerModel.deriv_p_NS_constraint = Constraint(BilevelStrategicConsumerModel.IU, rule=deriv_p_NS_constraint)
    BilevelStrategicConsumerModel.deriv_d_S_constraint = Constraint(BilevelStrategicConsumerModel.ZC, rule=deriv_d_S_constraint)
    BilevelStrategicConsumerModel.deriv_d_NS_constraint = Constraint(BilevelStrategicConsumerModel.LC, rule=deriv_d_NS_constraint)
    BilevelStrategicConsumerModel.positivity_p_min_constraint = Constraint(BilevelStrategicConsumerModel.IU, rule=positivity_p_min_constraint)
    BilevelStrategicConsumerModel.positivity_p_max_constraint = Constraint(BilevelStrategicConsumerModel.IU, rule=positivity_p_max_constraint)
    BilevelStrategicConsumerModel.positivity_d_min_constraint = Constraint(BilevelStrategicConsumerModel.JC, rule=positivity_d_min_constraint)
    BilevelStrategicConsumerModel.positivity_d_max_constraint = Constraint(BilevelStrategicConsumerModel.JC, rule=positivity_d_max_constraint)
    
    
    # Choose between KKT or PDOC and add to the model the respectively KKT or PDOC constraints
    Strategic_Producer_or_StrategicConsumer = 2
    (BilevelStrategicConsumerModel, ModelInput.MCP) = KKT_or_PDOC(BilevelStrategicConsumerModel, Strategic_Producer_or_StrategicConsumer, ModelInput.MCP_KKT_or_PDOC)
            
    # Add suffix for duals
    BilevelStrategicConsumerModel.dual = Suffix(direction=Suffix.IMPORT)

    return BilevelStrategicConsumerModel

def Solve_Config():
    Config = DataStorageClass
    Config.Select_Solve_Locally_NEOS = get_values_from_user("\n Please select: \n 1 - Solve Locally \n 2 - Solve using NEOS Server \n Type the value: ", list(range(1, 3)))   
    if Config.Select_Solve_Locally_NEOS == 1:
        Config.solver_name = Linear_or_Non_Linear()
    else:
        os.environ['NEOS_EMAIL'] = input('\n Please provide your NEOS Server email: ')
        Config.solver_name = Linear_or_Non_Linear()
    return Config
    
##########################################################################
# SOLVE MODEL FUNCTION
##########################################################################
def Solve_Model(Model, Config):

    
    if Config.Select_Solve_Locally_NEOS == 1:
        Config.Locally_or_NEOS_Server = 'Locally'
        solver = SolverFactory(Config.solver_name)
        
        # When using IPOPT it increases its tolerance and make the method to converge to problems with complementary constraints, however, since the tolerance is loose, the solution might be degenerated (suboptimal)
        # IPOPT will accept a solution that might not strictly satisfy all KKT optimality or complementarity conditions to machine precision — as long as it’s within these looser tolerances.
        # To solve using KKT please prefer KINITRO
        #    solver.options['tol'] = 1e-5
        #    solver.options['acceptable_tol'] = 1e-4
        #    solver.options['compl_inf_tol'] = 1e-4
        #    solver.options['max_iter'] = 5000
        #    solver.options['bound_relax_factor'] = 1e-6
        #    solver.options['print_level'] = 5
        
        # Start timer
        start_time_neos = time.time()
        
        # Solve model locally
        result = solver.solve(Model, tee=True)
        
        # End timer
        end_time_neos = time.time()
        
        # Compute elapsed time
        elapsed_time_neos = end_time_neos - start_time_neos
    else:
        Config.Locally_or_NEOS_Server = 'NEOS SERVER'
        neos_solver = SolverManagerFactory('neos')
        # Start timer
        start_time_neos = time.time()
        
        # Solve model using NEOS Server
        result = neos_solver.solve(Model,solver= Config.solver_name, tee=True,load_solutions=True, suffixes=['dual'])
        
        # End timer
        end_time_neos = time.time()
        
        # Compute elapsed time
        elapsed_time_neos = end_time_neos - start_time_neos
        
    return [Model, elapsed_time_neos, Config.solver_name, Config.Locally_or_NEOS_Server]

def Diagonalization(Config, BilevelStrategicProducerModel, StrategicProducer_ModelInput, BilevelStrategicConsumerModel, StrategicConsumer_ModelInput, Start_Strategic_Producer_or_StrategicConsumer):
    StrategicProducer_OutputSolution = []
    StrategicConsumer_OutputSolution = []
    o_previus = {(1, 1): 0, (2, 2): 0}
    b_previus = {(1, 1): 0,}
    counting = 0
    while (sum( abs(o_previus[i, u] - StrategicProducer_ModelInput.o_aux[i, u]) for i, u in BilevelStrategicProducerModel.IU) >= 0.1) or  (sum( abs(b_previus[j, c] - StrategicConsumer_ModelInput.b_aux[j, c]) for j, c in BilevelStrategicConsumerModel.JC) >= 0.1):
        for count in list(range(1, 3)):
            if Start_Strategic_Producer_or_StrategicConsumer == 1:
                prod = 0
                for y, u in BilevelStrategicProducerModel.IU:
                    Strategic_Producer = y
                    BilevelStrategicProducerModel = Creates_BilevelStrategicProducerModel(StrategicProducer_ModelInput, Strategic_Producer)
                    (BilevelStrategicProducerModel, StrategicProducer_Time, StrategicProducer_solver_name, StrategicProducer_Locally_or_NEOS_Server) = Solve_Model(BilevelStrategicProducerModel, Config)
                    StrategicProducer_OutputSolution.append(ReturnOutputSolution_function(BilevelStrategicProducerModel, StrategicProducer_Time, StrategicProducer_solver_name, StrategicProducer_Locally_or_NEOS_Server, 1, Strategic_Producer))
                    o_previus = StrategicProducer_ModelInput.o_aux
                    StrategicProducer_ModelInput.o_aux[y, u] = StrategicProducer_OutputSolution[prod].Strategic_o.get((y, u))
                    StrategicProducer_ModelInput.b_aux = StrategicConsumer_ModelInput.b_aux
                    prod = prod + 1
                Start_Strategic_Producer_or_StrategicConsumer = 2
            else:        
                con = 0
                for z, c in BilevelStrategicConsumerModel.JC:
                    Strategic_Consumer = z
                    BilevelStrategicConsumerModel = Creates_BilevelStrategicConsumerModel(StrategicConsumer_ModelInput, Strategic_Consumer)   
                    (BilevelStrategicConsumerModel, StrategicConsumer_Time, StrategicConsumer_solver_name, StrategicConsumer_Locally_or_NEOS_Server) = Solve_Model(BilevelStrategicConsumerModel, Config)
                    StrategicConsumer_OutputSolution.append(ReturnOutputSolution_function(BilevelStrategicConsumerModel, StrategicConsumer_Time, StrategicConsumer_solver_name, StrategicConsumer_Locally_or_NEOS_Server, 2, Strategic_Consumer))
                    b_previus = StrategicConsumer_ModelInput.b_aux
                    StrategicConsumer_ModelInput.b_aux[z, c] = StrategicConsumer_OutputSolution[con].Strategic_b.get(z, c)
                    StrategicConsumer_ModelInput.o_aux = StrategicProducer_ModelInput.o_aux
                    con = con + 1
                Start_Strategic_Producer_or_StrategicConsumer = 1
        counting = counting + 1
    return [BilevelStrategicProducerModel, StrategicProducer_ModelInput, StrategicProducer_OutputSolution, BilevelStrategicConsumerModel, StrategicConsumer_ModelInput, StrategicConsumer_OutputSolution, counting]

if __name__ == "__main__":
    main()