# -*- coding: utf-8 -*-

'''gavrptw/core.py'''

import os
import io
import random
import time
from csv import DictWriter
from deap import base, creator, tools
from . import BASE_DIR
from .utils import make_dirs_for_file, exist, load_instance, merge_rules
import matplotlib.pyplot as plt


def ind2route(individual, instance):
    '''Convert an individual's sequence into feasible routes with a focus on minimizing the number of vehicles.'''
    vehicle_capacity = instance['vehicle_capacity']
    depart_due_time = instance['depart']['due_time']
    distance_matrix = instance['distance_matrix']
    
    # Store demand, service time, and time windows (ready_time, due_time) of each customer
    demands = [instance[f'customer_{i}']['demand'] for i in individual]
    service_times = [instance[f'customer_{i}']['service_time'] for i in individual]
    ready_times = [instance[f'customer_{i}']['ready_time'] for i in individual]
    due_times = [instance[f'customer_{i}']['due_time'] for i in individual]
    
    # Initialize variables
    n = len(individual)
    routes = []
    route = []
    current_load = 0
    current_time = 0
    last_customer_id = 0
    infeasibility_penalty = 0
    
    # Iterate through customers in the individual's sequence
    for i in range(n):
        customer_id = individual[i]
        demand = demands[i]
        service_time = service_times[i]
        ready_time = ready_times[i]
        due_time = due_times[i]
        
        travel_time = distance_matrix[last_customer_id][customer_id]
        arrival_time = current_time + travel_time
        wait_time = max(0, ready_time - arrival_time)
        service_end_time = arrival_time + wait_time + service_time
        return_time = distance_matrix[customer_id][0]
        
        # Check feasibility of adding the customer
        if (current_load + demand <= vehicle_capacity) and \
           (service_end_time + return_time <= depart_due_time):
            # Add customer to the current route
            route.append(customer_id)
            current_load += demand
            current_time = service_end_time
        else:
            # Start a new route, but check if this can be done feasibly within the time constraints
            if len(routes) > 0 and current_time + distance_matrix[0][customer_id] > depart_due_time:
                infeasibility_penalty += 1
            
            # Save the current route and start a new one
            routes.append(route)
            route = [customer_id]
            current_load = demand
            current_time = distance_matrix[0][customer_id] + service_time
            # Check if starting the new route is feasible within the customer's time window
            if not (ready_time <= current_time <= due_time):
                infeasibility_penalty += 1
        
        last_customer_id = customer_id
    
    if route:
        # Save the last route if it contains customers
        routes.append(route)
    
    # Optionally handle or return infeasibility penalties
    return routes


def print_route(route, instance, merge=False):
    '''gavrptw.core.print_route(route, instance, merge=False)'''
    route_str = '0'
    sub_route_count = 0
    total_capacity = 0
    total_distance = 0
    vehicle_capacity = instance['vehicle_capacity']

    for sub_route in route:
        sub_route_count += 1
        sub_route_str = '0'
        sub_route_capacity = 0
        sub_route_distance = 0
        last_customer_id = 0

        for customer_id in sub_route:
            sub_route_str = f'{sub_route_str} - {customer_id}'
            route_str = f'{route_str} - {customer_id}'

            # Calculate capacity
            demand = instance[f'customer_{customer_id}']['demand']
            sub_route_capacity += demand

            # Calculate distance
            sub_route_distance += instance['distance_matrix'][last_customer_id][customer_id]
            last_customer_id = customer_id

        # Add distance from last customer to depot
        sub_route_distance += instance['distance_matrix'][last_customer_id][0]

        sub_route_str = f'{sub_route_str} - 0'
        route_str = f'{route_str} - 0'

        if not merge:
            print(f'  Vehicle {sub_route_count}\'s route: {sub_route_str}')
            print(f'    Capacity used: {sub_route_capacity} / {vehicle_capacity}')
            print(f'    Distance: {sub_route_distance} km')

        total_capacity += sub_route_capacity
        total_distance += sub_route_distance

    if merge:
        print(route_str)
        print(f'Total capacity used: {total_capacity} / {vehicle_capacity}')
        print(f'Total distance: {total_distance} km')

def eval_vrptw1(individual, instance, unit_cost=1000.0, init_cost=100.0, wait_cost=0.1, delay_cost=0.1):
    total_distance = 0
    total_wait_time = 0
    total_delay_time = 0
    
    route = ind2route(individual, instance)
    
    for sub_route in route:
        sub_route_distance = 0
        last_customer_id = 0
        current_time = 0
        
        for customer_id in sub_route:
            # Calculate section distance
            distance = instance['distance_matrix'][last_customer_id][customer_id]
            
            # Update sub-route distance
            sub_route_distance += distance
            
            # Update current time (arrival at customer_id)
            current_time += distance
            
            # Retrieve customer time window and service time
            customer_info = instance[f'customer_{customer_id}']
            service_time = customer_info['service_time']
            ready_time = customer_info['ready_time']
            due_time = customer_info['due_time']
            
            # Calculate waiting time if arriving early
            if current_time < ready_time:
                wait_time = ready_time - current_time
                total_wait_time += wait_time
                current_time = ready_time  # Wait until the customer is ready
            else:
                wait_time = 0
            
            # Calculate delay time if arriving late
            if current_time > due_time:
                delay_time = current_time - due_time
                total_delay_time += delay_time
            else:
                delay_time = 0
            
            # Update current time with service time
            current_time += service_time
            
            # Update last customer ID
            last_customer_id = customer_id
        
        # Calculate distance back to the depot
        sub_route_distance += instance['distance_matrix'][last_customer_id][0]
        
        # Add sub-route distance to total distance
        total_distance += sub_route_distance
    
    # Calculate total cost considering the different penalties
    total_cost = (unit_cost * total_distance) + (wait_cost * total_wait_time) + (delay_cost * total_delay_time) + init_cost
    
    # Fitness is typically inversely related to cost
    fitness = 1 / total_cost if total_cost > 0 else float('inf')
    
    return (fitness, total_distance)

def eval_vrptw2(individual, instance):
    '''gavrptw.core.eval_vrptw(individual, instance, unit_cost=1.0, init_cost=0, wait_cost=0,
        delay_cost=0)'''
    ###emission coeifficients
    ##NAEI Coeifficients
    a,b,c,d,e,f,g,v=4529.765238, 60.24714667, 0.289025869, 0.016975237, -0.000108584, 9.74901e-07, 0, 50.0
    ##MEET Coeifficients
    k,l,g1,q,r,u=1.27,0.0614,1,-0.0011,-0.00235,-1.33
    sub_route_emission_cost=0
    total_distance=0
    total_cost = 0
    travel_time_total=0
    service_time_total=0
    route = ind2route(individual, instance)
    time_windows = [
        (0.0, 60.0, 30),  # TW 0 AM 
        (61.0, 120.0, 30),# TW 1 AM 
        (121.0, 180.0, 30),# TW 2 AM 
        (181.0, 240.0, 30),# TW 3 AM 
        (241.0, 300.0, 30),# TW 4 AM 
        (301.0, 360.0, 30),# TW 5 AM 
        (361.0, 420.0, 30), # TW 6 AM 
        (421.0, 480.0, 30),#TW 7 AM 
        (481.0, 540.0, 28.64632),#TW 8 AM 
        (541.0, 600.0, 28.64632),#TW 9 AM 
        (601.0, 660.0, 28.64632),#TW 10 AM 
        (661.0, 720.0, 28.64632),#TW 11 AM 
        (721.0, 780.0, 29.451),#TW 12 AM 
        (781.0, 840.0, 29.451),#TW 1 PM 
        (841.0, 900.0, 29.451),#TW 2 PM 
        (901.0, 960.0, 28.9682),#TW 3 PM 
        (961.0, 1020.0,28.9682),#TW 4 PM 
        (1021.0,1080.0,27.84165),#TW 5 PM 
        (1081.0,1140.0, 30),#TW 6 PM 
        (1141.0,1200.0, 30),#TW 7 PM 
        (1201.0,1260.0, 30),#TW 8 PM 
        (1261.0,1320.0, 30),#TW 9 PM 
        (1321.0,1380.0, 30),#TW 10 PM      
        (1381.0,1440.0, 30),#TW 11 PM 
    ]
    def speed_function(time):
        hour = time 
        for start, end, speed in time_windows:
            if start <= hour < end:
                return speed
        return 50
    for sub_route in route:
        sub_route_distance = 0
        elapsed_time = 0
        last_customer_id = 0
        for customer_id in sub_route:
            # Calculate section distance
            distance = instance['distance_matrix'][last_customer_id][customer_id]
            # Update sub-route distance
            sub_route_distance = sub_route_distance + distance
            # Calculate time cost
            speed = speed_function((instance[f'customer_{customer_id}']['ready_time']+(instance[f'customer_{customer_id}']['due_time']))/2)
            travel_time = distance /speed
            travel_time_total+=travel_time
            arrival_time = elapsed_time + travel_time 
            # Update elapsed time
            elapsed_time = arrival_time + instance[f'customer_{customer_id}']['service_time']
            service_time_total=service_time_total+ instance[f'customer_{customer_id}']['service_time']
            # Update last customer ID
            last_customer_id = customer_id
        # Calculate transport cost
        sub_route_distance = sub_route_distance + instance['distance_matrix'][last_customer_id][0]
       #emission calculation
       ##1- NAEI MODEL
        #sub_route_emission_cost=sub_route_emission_cost+((a/speed+b+c*speed+d*speed**2+e*speed**3+f*speed**4+g*speed**5)*sub_route_distance)
       ##1- MEET MODEL 
        sub_route_emission_cost=sub_route_emission_cost+((k+l*g1+q*g1**2+r*speed+u/speed)*sub_route_distance) #* carbon_cost
        # Obtain sub-route cost
        sub_route_cost = sub_route_emission_cost 
        total_distance=total_distance+sub_route_distance
        # Update total cost
        total_cost = total_cost + sub_route_cost
    
    fitness = 1/total_cost
  
    return (fitness,total_distance)


def apply_mapping(gene, mapping):
    while gene in mapping:
        gene = mapping[gene]
    return gene

def cx_partially_matched(ind1, ind2):
    size = min(len(ind1), len(ind2))
    cxpoint1, cxpoint2 = sorted(random.sample(range(size), 2))

    part1 = ind2[cxpoint1:cxpoint2+1]
    part2 = ind1[cxpoint1:cxpoint2+1]

    # Create mappings
    mapping1to2 = {part1[i]: part2[i] for i in range(len(part1))}
    mapping2to1 = {part2[i]: part1[i] for i in range(len(part2))}

    rule1to2 = list(zip(part1, part2))
    is_fully_merged = False
    while not is_fully_merged:
        rule1to2, is_fully_merged = merge_rules(rule1to2)

    rule2to1 = {rule[1]: rule[0] for rule in rule1to2}

    # Create children by applying mapping outside the crossover points
    child1 = [apply_mapping(gene, mapping2to1) if i < cxpoint1 or i > cxpoint2 else gene for i, gene in enumerate(ind1)]
    child2 = [apply_mapping(gene, mapping1to2) if i < cxpoint1 or i > cxpoint2 else gene for i, gene in enumerate(ind2)]

    # Insert the crossover parts
    child1[cxpoint1:cxpoint2+1] = part2
    child2[cxpoint1:cxpoint2+1] = part1

    return child1, child2


import random

def mut_shuffle_indexes(individual):
    """Apply a 2-opt swap mutation to the individual."""
    size = len(individual)
    if size < 2:
        return individual  # No mutation possible for individuals with fewer than 2 elements
    
    # Ensure i < j to simplify the slicing and reversing
    i, j = sorted(random.sample(range(size), 2))
    
    # Perform the 2-opt swap: reverse the segment between i and j inclusive
    individual[i:j+1] = individual[i:j+1][::-1]
    
    return individual

def greedy_insertion_heuristic(ind_size):
    # Initialize an empty permutation
    permutation = []
    
    # Create a list of all elements
    remaining = list(range(1, ind_size + 1))
    
    # Randomly choose a starting point
    current = random.choice(remaining)
    permutation.append(current)
    remaining.remove(current)
    
    # Greedy insertion
    while remaining:
        # Find the position that minimizes insertion cost
        min_cost = float('inf')
        best_position = -1
        for i in range(len(permutation) + 1):
            cost = 0
            if i > 0:
                cost += abs(permutation[i - 1] - remaining[0])
            if i < len(permutation):
                cost += abs(permutation[i] - remaining[0])
            if cost < min_cost:
                min_cost = cost
                best_position = i
        
        # Insert the element at the best position
        permutation.insert(best_position, remaining[0])
        remaining.pop(0)
    
    return permutation

def initialize_population(ind_size, pop_size):
    population = []
    for _ in range(pop_size):
        permutation = greedy_insertion_heuristic(ind_size)
        population.append(creator.Individual(permutation))
    return population



def local_search(individual, toolbox, max_iters=100):
    # Clone the initial individual
    best = toolbox.clone(individual)
    # Evaluate the fitness of the cloned individual
    best_fitness, best_distance = toolbox.evaluate(best)
    best.fitness.values = (best_fitness,)
    
    # Initialize the history list to keep track of individuals and their fitnesses
    history = [(toolbox.clone(best), best_fitness, best_distance)]
    
    for _ in range(max_iters):
        # Clone the current best individual to create a neighbor
        neighbor = toolbox.clone(best)
        # Mutate the neighbor
        toolbox.mutate(neighbor)
        # Evaluate the fitness of the neighbor
        neighbor_fitness, neighbor_distance = toolbox.evaluate(neighbor)
        neighbor.fitness.values = (neighbor_fitness,)
        
        # Append the neighbor and its fitness to the history list
        history.append((toolbox.clone(neighbor), neighbor_fitness, neighbor_distance))
        
        # If the neighbor is better than the current best, update the best individual
        if neighbor.fitness.values[0] > best.fitness.values[0]:
            best = neighbor
            best_fitness = neighbor_fitness
            best_distance = neighbor_distance
    
    # Return the best individual, its distance, and the history list
    return best, best_distance



def plot_route(route, instance):
    plt.figure(figsize=(10, 8))
    plt.title('Vehicle Routes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Fixed depot coordinates
    depot_x = 40
    depot_y = 50
    
    # Plot depot
    plt.plot(depot_x, depot_y, marker='s', color='red', markersize=10, label='Depot')

    # Plot customers
    for customer_key in instance:
        if customer_key.startswith('customer_'):
            customer_id = int(customer_key.split('_')[1])  # Extract customer ID
            customer_x = instance[customer_key]['coordinates']['x']
            customer_y = instance[customer_key]['coordinates']['y']
            plt.plot(customer_x, customer_y, marker='o', color='blue', markersize=5)

    # Plot routes
    for idx, sub_route in enumerate(route):
        route_x = [depot_x]
        route_y = [depot_y]
        for customer_id in sub_route:
            customer_key = f'customer_{customer_id}'
            customer_x = instance[customer_key]['coordinates']['x']
            customer_y = instance[customer_key]['coordinates']['y']
            route_x.append(customer_x)
            route_y.append(customer_y)
        route_x.append(depot_x)
        route_y.append(depot_y)
        plt.plot(route_x, route_y, marker='o', label=f'Vehicle {idx + 1}')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_gavrptw(instance_name,ind_size, pop_size,cx_pb, mut_pb, n_gen, export_csv=False, customize_data=False):
    start_time = time.time()  # Début du chronomètre
    best_distance = None
    fitness_distance_data = []

    if customize_data:
        json_data_dir = os.path.join(BASE_DIR, 'data', 'json_customize')
    else:
        json_data_dir = os.path.join(BASE_DIR, 'data', 'json')
    json_file = os.path.join(json_data_dir, f'{instance_name}.json')
    instance = load_instance(json_file=json_file)
    if instance is None:
        return

    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    toolbox.register('indexes', random.sample, range(1, ind_size + 1), ind_size)
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', lambda: initialize_population(ind_size, pop_size))
    toolbox.register('evaluate', eval_vrptw1, instance=instance)
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', cx_partially_matched)
    toolbox.register('mutate', mut_shuffle_indexes)
    pop = toolbox.population()

    csv_data = []

    print('Start of evolution')
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, (fit, distance) in zip(pop, fitnesses):
        ind.fitness.values = (fit,)  # using the fitness value
        fitness_distance_data.append((fit, distance)) 
        print(f'Evaluated individual: {ind}, Fitness: {fit}, Distance: {distance}')
    print(f'  Evaluated {len(pop)} individuals')

    max_stagnation = 1000
    stagnation_count = 0
    prev_best_fitness = None

    for gen in range(n_gen):
        print(f'-- Generation {gen} --')
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_pb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mut_pb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Apply local search to each offspring and update fitness_distance_data
        for i in range(len(offspring)):
            offspring[i], distance = local_search(offspring[i], toolbox)
            fitness_distance_data.append((offspring[i].fitness.values[0], distance))

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, (fit, distance) in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit,)  # Utiliser seulement la valeur de fitness
            fitness_distance_data.append((fit, distance)) 
            print(f'Evaluated individual: {ind}, Fitness: {fit}, Distance: {distance}')
        print(f'  Evaluated {len(invalid_ind)} individuals')

        pop[:] = offspring
        
        fits_and_distances = [(ind.fitness.values[0], distance) for ind, (_, distance) in zip(pop, fitness_distance_data)]
        fits = [fit for fit, _ in fits_and_distances]
        distances = [distance for _, distance in fits_and_distances]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum([x ** 2 for x in fits])
        std = abs(sum2 / length - mean * 2) * 0.5
        print(f'  Min {min(fits)}')
        print(f'  Max {max(fits)}')
        print(f'  Avg {mean}')
        print(f'  Std {std}')
        if export_csv:
            csv_row = {
                'generation': gen,
                'evaluated_individuals': len(invalid_ind),
                'min_fitness': min(fits),
                'max_fitness': max(fits),
                'avg_fitness': mean,
                'std_fitness': std,
            }
            csv_data.append(csv_row)

        current_best_fitness = max(fits)
        if prev_best_fitness is not None and current_best_fitness == prev_best_fitness:
            stagnation_count += 1
        else:
            stagnation_count = 0
        prev_best_fitness = current_best_fitness

        if stagnation_count >= max_stagnation:
            print(f'Stagnation detected. Stopping early at generation {gen}.')
            break

    print('-- End of (successful) evolution --')
    best_ind = tools.selBest(pop, 1)[0]
    best_fitness = best_ind.fitness.values[0]
    for fit, distance in fitness_distance_data:
        if fit == best_fitness:
            best_distance = distance 
    print(f'Best individual: {best_ind}')
    print(f'Fitness: {best_ind.fitness.values[0]}')
    print_route(ind2route(best_ind, instance),instance)
    print(f'Total emissions in Grams: {1 / best_ind.fitness.values[0]}')
    print(f'Total distance in km: {best_distance}')

    end_time = time.time()  # Fin du chronomètre
    execution_time = end_time - start_time
    print(f'Total execution time: {execution_time:.2f} seconds')

    #plot_route(ind2route(best_ind, instance), instance)