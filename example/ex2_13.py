#
# Example 2.13 - Allen Holder
#

from __future__ import division # safety with double division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import os


# Instantiate and name the model
M = AbstractModel()
M.name = "Example 2.13"

# Sets
M.NumYears = Param(within=NonNegativeIntegers)
M.Age = RangeSet(0,M.NumYears-1)
M.Year = RangeSet(1,M.NumYears+1)
def CreateArcSet(M):
   ArcSet = []
   for (i,j) in M.Year * M.Year:
      if i < j:
         ArcSet.append((i,j))
   return ArcSet
M.Arc = Set(within=M.Year*M.Year, initialize=CreateArcSet)

# Parameters
M.MaintCost = Param(M.Age, within=NonNegativeIntegers)
M.PurchCost = Param(M.Year, within=NonNegativeIntegers)
def CalcCostCoef(M,i,j):
   return value(M.PurchCost[i] + sum(M.MaintCost[k] for k in M.Age if k < j-i))
M.Cost = Param(M.Arc, within=NonNegativeIntegers, initialize=CalcCostCoef)

# Variables
M.Flow = Var(M.Arc, bounds=(0,1))

# Objective
def CalcCost(M):
   return sum(M.Flow[i,j]*M.Cost[i,j] for (i,j) in M.Arc)
M.TotalCost = Objective(rule=CalcCost, sense=minimize)
# Constraints
def EnsureSupplyDemand(M,k):
   if k == 1:
      return sum(M.Flow[k,j] for j in M.Year if (k,j) in M.Arc) <= 1
   elif k == len(M.Year):
      return sum(M.Flow[i,k] for i in M.Year if (i,k) in M.Arc) >= 1
   else:
      return sum(M.Flow[i,k] for i in M.Year if (i,k) in M.Arc) == \
             sum(M.Flow[k,j] for j in M.Year if (k,j) in M.Arc)
M.DemandBound = Constraint(M.Year, rule=EnsureSupplyDemand)

# Create a problem instance
this_dir = os.path.dirname(__file__)
data_path = os.path.join(this_dir, "ex2_13.dat")
instance = M.create_instance(data_path)

# Indicate which solver to use
Opt = SolverFactory("gurobi")

# Generate a solution
Soln = Opt.solve(instance)
instance.solutions.load_from(Soln)

# Print the output
print("Termination Condition was "+str(Soln.Solver.Termination_condition))
display(instance)