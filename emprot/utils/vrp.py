from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

def fragment_tracing(
    m, 
    n, 
    distance_matrix, 
    residue_counts, 
    alpha=0.5,
    time_limit=10,
    log_search=True, 
    conflict_pairs=None, 
    same_chain_pairs=None, 
):
    
    data = {}
    data['distance_matrix'] = distance_matrix
    data['residue_counts'] = residue_counts
    data['num_vehicles'] = n
    data['depot'] = 0  # virtual depot

    total_residues = sum(residue_counts)
    avg_residues = total_residues / n

    # setting up model
    manager = pywrapcp.RoutingIndexManager(m + 1, n, data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node == 0 or to_node == 0:
            return 0
        else:
            return data['distance_matrix'][from_node - 1][to_node - 1]

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)

    # set up Arc Cost
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    # residue 
    def residue_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return 0 if from_node == 0 else data['residue_counts'][from_node - 1]

    residue_callback_index = routing.RegisterUnaryTransitCallback(residue_callback)

    # add residue dimension
    routing.AddDimension(
        residue_callback_index,
        0,  # slack
        int(total_residues),  # max capacity
        True,  # start cumul to zero
        'Residue'
    )
    residue_dimension = routing.GetDimensionOrDie('Residue')

    # soft upper bound 
    target = int(avg_residues * 1.2)

    for vehicle_id in range(n):
        end_index = routing.End(vehicle_id)
        residue_dimension.SetCumulVarSoftUpperBound(end_index, target, int((1 - alpha) * 100))


    # set conflict pairs
    if conflict_pairs:
        print("# Adding restraints between {} conflict pairs".format(len(conflict_pairs)))
        for p in conflict_pairs:
            ia = manager.NodeToIndex(p[0] + 1)
            ib = manager.NodeToIndex(p[1] + 1)
            routing.solver().Add(
                routing.VehicleVar(ia) != routing.VehicleVar(ib)
            )


    # set same chain pairs
    if same_chain_pairs:
        print("# Adding restraints between {} same chain pairs".format(len(same_chain_pairs)))
        for p in same_chain_pairs:
            ia = manager.NodeToIndex(p[0] + 1)
            ib = manager.NodeToIndex(p[1] + 1)
            routing.solver().Add(
                routing.VehicleVar(ia) == routing.VehicleVar(ib)
            )


    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = time_limit
    search_parameters.log_search = log_search

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        routes = []
        total_distance = 0
        total_residues_per_chain = []

        for vehicle_id in range(n):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            prev_index = index

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index != 0:
                    segment = node_index - 1
                    route.append(segment)
                prev_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += distance_callback(prev_index, index)

            routes.append(route)
            total_distance += route_distance

            #print("="*100)
            #print(prev_index)
            #print(solution.Value(residue_dimension.CumulVar(prev_index)))
            #print("="*100)

            total_residues_per_chain.append(
                solution.Value(residue_dimension.CumulVar(prev_index))
            )

        imbalance = max(total_residues_per_chain) - min(total_residues_per_chain)

        return {
            'routes': routes,
            'total_distance': total_distance,
            'residue_distribution': total_residues_per_chain,
            'residue_imbalance': imbalance,
            'avg_residues': avg_residues
        }
    else:
        return None

