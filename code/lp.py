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


from scipy.optimize import linprog
def linear_programming_factory():
    # Coefficients of objective function are negative, because the problem is maximization.
    obj = [-20, -12, -30, -15] 
    # LHS matrix of inequality equations
    lhs = [[1, 1, 1, 1],
    [3, 2, 2, 0],
    [0, 1, 5, 3]]
    # RHS matrix of inequality equations
    rhs = [50,
        100,
        90]
    lp_opt = linprog(c=obj,
                 A_ub=lhs,
                 b_ub=rhs,
                 method = 'interior-point')
    
    print(lp_opt)



from pulp import *     #https://github.com/coin-or/pulp
def linear_programming_factory2():
  
    # problem formulation
    model = LpProblem(sense=LpMaximize)

    x_1 = LpVariable(name="product1", lowBound=0)
    x_2 = LpVariable(name="product2", lowBound=0)
    x_3 = LpVariable(name="product3", lowBound=0)
    x_4 = LpVariable(name="product4", lowBound=0)


    # constaints
    model += x_1 + x_2 + x_3 + x_4 <= 50  
    model += 3*x_1 + 2*x_2 + 2*x_3 <= 100
    model += x_2 + 5*x_3 + 3*x_4 <= 90

    #objective function
    model += 20*x_1 + 12*x_2 + 30*x_3 + 15*x_4

    # solve (without being verbose)
    status = model.solve(PULP_CBC_CMD(msg=False))
    print("product1:", x_1.value())
    print("product2:", x_2.value())
    print("product3:", x_3.value())
    print("product4:", x_4.value())
    print("profit:", value(model.objective))