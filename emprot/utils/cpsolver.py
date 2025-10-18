from ortools.sat.python import cp_model

def solve(score1d, rst2d, hint=None, time_limit=1200, num_workers=2, log_search_progress=True, **kwargs):
    model = cp_model.CpModel()

    # add decision variable
    print("Preprocess 1/3 add decision variable")
    n = len(score1d)
    x = [model.NewBoolVar(f'x[{i}]') for i in range(n)]

    # process rst2d with self-exclusion
    for i in range(n):
        rst2d[i][i] = 1

    # add rst
    print("Preprocess 2/3 add rst")
    for i in range(n):
        model.Add(sum(x[k] for k in range(n) if rst2d[i][k] == 1) <= 1).OnlyEnforceIf(x[i])
 
    # add hint
    if hint is not None:
        for idx in hint:
            model.AddHint(x[idx], True)

    # set objective 
    print("Preprocess 3/3 set objectives")
    objectives = []
    for i in range(n):
        objectives.append(score1d[i] * x[i])
    model.Maximize(sum(objectives))

    # solve
    print("Solving")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = num_workers
    solver.parameters.log_search_progress = log_search_progress
    solution_printer = cp_model.ObjectiveSolutionPrinter()
    status = solver.SolveWithSolutionCallback(model, solution_printer)

    """
    return status
    cp_model.UNKNOWN: 0 - unknown
    cp_model.MODEL_INVALID: 1 - invalid model
    cp_model.FEASIBLE: 2 - found one solution
    cp_model.INFEASIBLE: 3 - no solution
    cp_model.OPTIMAL: 4 - optimal
    cp_model.INTERRUPTED: 5 - time limit reached
    """

    # get result
    results = []
    if (status == cp_model.OPTIMAL) or \
        (status == cp_model.FEASIBLE) or \
        (status == cp_model.INTERRUPTED and solver.ResponseStats().has_feasible_solution()):
        for i in range(n):
            if solver.BooleanValue(x[i]):
                results.append(i)
    return results, solver.ObjectiveValue(), status


if __name__ == '__main__':
    score = [2.2, 4.5, 2.9, 1.8]
    rst = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ]
    result, score, status = solve(score, rst, hint=[0,1,2])
    print(result)
    print(score)
    print(status)
