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

##########################################################################
# MAIN
##########################################################################
def main():
    ModelInput = LoadInputData_function()
    MarketClearingModel = Creates_MarketClearingModel(ModelInput)    
    (MarketClearingModel, Time, solver_name, Locally_or_NEOS_Server) = Solve_Model(MarketClearingModel)
    OutputSolution = ReturnOutputSolution_function(MarketClearingModel, Time, solver_name, Locally_or_NEOS_Server)
    print("\n", ModelInput, "\n")
    print(OutputSolution, "\n")

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
    ModelInput.o = ModelInput.Cost
    ModelInput.b = ModelInput.max_utility
    return ModelInput

##########################################################################
# RETURN OUTPUT SOLUTION
##########################################################################
def ReturnOutputSolution_function(Model, Time, solver_name, Locally_or_NEOS_Server):
    OutputSolution = DataStorageClass('Output Data')
    cleared_price = Model.dual.get(Model.Balance_Constraint)
    OutputSolution.Objective = Model.Objective()
    OutputSolution.p = {(i, u): Model.p[i, u].value for i, u in Model.IU}
    OutputSolution.d = {(j, c): Model.d[j, c].value for j, c in Model.JC}
    OutputSolution.Cleared_Price = cleared_price
    OutputSolution.profit = sum(cleared_price * Model.p[i, u].value - Model.Cost[i, u] * Model.p[i, u].value for i, u in Model.IU)
    OutputSolution.utility = sum(Model.max_utility[j, c]*Model.d[j, c].value - cleared_price*Model.d[j, c].value for j, c in Model.JC)
    OutputSolution.SocialWelfare = sum(cleared_price * Model.p[i, u].value - Model.Cost[i, u] * Model.p[i, u].value for i, u in Model.IU) + sum(Model.max_utility[j, c]*Model.d[j, c].value - cleared_price*Model.d[j, c].value for j, c in Model.JC)
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
    return [Select_Solver, solver_name]


##########################################################################
# CONSTRAINTS FUNCTIONS
##########################################################################        
# Objective: Social Welfare
def Social_Welfare_objective(model):
    return sum(model.b[j, c] * model.d[j, c] for j, c in model.JC) - sum(model.o[i, u] * model.p[i, u] for i, u in model.IU)

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


##########################################################################
# CREATES THE MODEL FUNCTIONS
########################################################################## 

def Creates_MarketClearingModel(ModelInput):
    MarketClearingModel = ConcreteModel()
    
    # Creates the Sets
    # Set of all Producers
    MarketClearingModel.I = Set(initialize= ModelInput.Set_I)  # Example with 2 producers
    MarketClearingModel.U = Set(MarketClearingModel.I, initialize= ModelInput.Set_U)
    MarketClearingModel.IU = Set(dimen=2, initialize=lambda m: [(i, u) for i in m.I for u in m.U[i]])
        
    # Set of all Consumers
    MarketClearingModel.J = Set(initialize= ModelInput.Set_J)  # Example with 1 consumer
    MarketClearingModel.C = Set(MarketClearingModel.J, initialize= ModelInput.Set_C)
    MarketClearingModel.JC = Set(dimen=2, initialize=lambda m: [(j, c) for j in m.J for c in m.C[j]])
    
    # Parameters
    MarketClearingModel.Cost = Param(MarketClearingModel.IU, initialize= ModelInput.Cost)  # Cost per unit for each producer-unit pair
    MarketClearingModel.max_utility = Param(MarketClearingModel.JC, initialize= ModelInput.max_utility)  # Max utility per consumer for each consumer-demand pair
    MarketClearingModel.p_max =  Param(MarketClearingModel.IU,initialize= ModelInput.p_max)
    MarketClearingModel.d_max = Param(MarketClearingModel.JC, initialize= ModelInput.d_max)
    MarketClearingModel.o = {(i, u): MarketClearingModel.Cost[i, u] for i, u in MarketClearingModel.IU}
    MarketClearingModel.b = {(j, c): MarketClearingModel.max_utility[j, c] for j, c in MarketClearingModel.JC}
    
    # Variables
    MarketClearingModel.p = Var(MarketClearingModel.IU, within=NonNegativeReals)
    MarketClearingModel.d = Var(MarketClearingModel.JC, within=NonNegativeReals)
    MarketClearingModel.Lambda = Var()
    MarketClearingModel.mu_p_min = Var(MarketClearingModel.IU, within=NonNegativeReals)
    MarketClearingModel.mu_p_max = Var(MarketClearingModel.IU, within=NonNegativeReals)
    MarketClearingModel.mu_d_min = Var(MarketClearingModel.JC, within=NonNegativeReals)
    MarketClearingModel.mu_d_max = Var(MarketClearingModel.JC, within=NonNegativeReals)
    
    # Starting value
    #MarketClearingModel.Lambda.value = 2
        
    # Add constraints to the model
    MarketClearingModel.Objective = Objective(rule=Social_Welfare_objective, sense=maximize)
    MarketClearingModel.Balance_Constraint = Constraint(rule=Balance_Constraint)
    MarketClearingModel.min_power_constraint = Constraint(MarketClearingModel.IU, rule=min_power_constraint)
    MarketClearingModel.max_power_constraint = Constraint(MarketClearingModel.IU, rule=max_power_constraint)
    MarketClearingModel.min_demand_constraint = Constraint(MarketClearingModel.JC, rule=min_demand_constraint)
    MarketClearingModel.max_demand_constraint = Constraint(MarketClearingModel.JC, rule=max_demand_constraint)
            
    # Add suffix for duals
    MarketClearingModel.dual = Suffix(direction=Suffix.IMPORT)

    return MarketClearingModel

##########################################################################
# SOLVE MODEL FUNCTION
##########################################################################
def Solve_Model(Model):
    Select_Solve_Locally_NEOS = get_values_from_user("\n Please select: \n 1 - Solve Locally \n 2 - Solve using NEOS Server \n Type the value: ", list(range(1, 3)))   
    if Select_Solve_Locally_NEOS == 1:
        (Select_Solver, solver_name) = Linear_or_Non_Linear()
    else:
        os.environ['NEOS_EMAIL'] = input('\n Please provide your NEOS Server email: ')
        (Select_Solver, solver_name) = Linear_or_Non_Linear()
    
    if Select_Solve_Locally_NEOS == 1:
        Locally_or_NEOS_Server = 'Locally'
        solver = SolverFactory(solver_name)
        
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
        Locally_or_NEOS_Server = 'NEOS SERVER'
        neos_solver = SolverManagerFactory('neos')
        # Start timer
        start_time_neos = time.time()
        
        # Solve model using NEOS Server
        result = neos_solver.solve(Model,solver= solver_name, tee=True,load_solutions=True, suffixes=['dual'])
        
        # End timer
        end_time_neos = time.time()
        
        # Compute elapsed time
        elapsed_time_neos = end_time_neos - start_time_neos
              
    return [Model, elapsed_time_neos, solver_name, Locally_or_NEOS_Server]

if __name__ == "__main__":
    main()
