# Linear programming examples

from scipy.optimize import linprog
def linear_programming_farmer():
    # problem formulation
    obj = [-1.2, -1.7]
    lhs_eq = [[1, 1]]
    rhs_eq = [5000]
    lhs_ieq = [[1, 0], [0, 1]]
    rhs_ieq = [3000, 4000]
    bounds = [(0, None), (0, None)]

    # solve
    res = linprog(c=obj, A_eq=lhs_eq, b_eq=rhs_eq, A_ub=lhs_ieq, b_ub=rhs_ieq, bounds=bounds, method='simplex')

    # print results
    print("potatoes:", res.x[0])
    print("carrots:", res.x[1])
    print("profit:", -res.fun)


from pulp import *     #https://github.com/coin-or/pulp
def linear_programming_farmer2():
        
    # problem formulation
    model = LpProblem(sense=LpMaximize)

    x_p = LpVariable(name="potatoes", lowBound=0)
    x_c = LpVariable(name="carrots", lowBound=0)

    # constaints
    model += x_p       <= 3000  # potatoes
    model +=       x_c <= 4000  # carrots
    model += x_p + x_c <= 5000  # fertilizer

    #objective function
    model += x_p * 1.2 + x_c * 1.7

    # solve (without being verbose)
    status = model.solve(PULP_CBC_CMD(msg=False))
    print("potatoes:", x_p.value())
    print("carrots:", x_c.value())
    print("profit:", value(model.objective))