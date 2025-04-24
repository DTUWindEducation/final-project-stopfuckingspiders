from bem_solvers import BEMSolver, BEMSolverOpt

# Run standard analysis
print("Running standard BEM solver...")
standard = BEMSolver()
standard.run()

# Run optimal analysis
print("Running optimal BEM solver...")
optimal = BEMSolverOpt() 
optimal.run()

# Or combine results for comparison
results = {
    "standard": standard.results,
    "optimal": optimal.results
}