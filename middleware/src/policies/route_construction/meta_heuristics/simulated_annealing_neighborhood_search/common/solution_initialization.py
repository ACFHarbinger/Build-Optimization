"""
Initialization strategies for vehicle routing solutions.
Extracts constructive heuristics from the legacy solutions module.
"""


def _categorize_bins_by_zone(bins, bins_coordinates):
    """Group bins into zones based on longitude."""
    max_lng = max(bins_coordinates["Lng"])
    min_lng = min(bins_coordinates["Lng"])
    lng_amp = max_lng - min_lng
    zone_1_l = min_lng + lng_amp / 3
    zone_2_l = min_lng + 2 * lng_amp / 3

    zone_bins = {1: [], 2: [], 3: []}  # type: ignore[var-annotated]
    for i in range(len(bins)):
        bin_id = bins[i]
        try:
            lng = bins_coordinates.iloc[bin_id - 1 if bin_id > 0 else 0]["Lng"]
        except IndexError:
            continue

        if min_lng <= lng < zone_1_l:
            zone_bins[1].append(bin_id)
        elif zone_1_l <= lng < zone_2_l:
            zone_bins[2].append(bin_id)
        else:
            zone_bins[3].append(bin_id)
    return zone_bins


def _get_bin_stock(data, bin_id, E, B):
    row = data[data["#bin"] == bin_id]
    if row.empty:
        return 0
    return (row.iloc[0]["Stock"] + row.iloc[0]["Accum_Rate"]) * E * B


def _find_closest_valid_bin(current_bin, potential_bins_in_zone, available_bins, current_route, distance_matrix):
    """Find the closest bin in the zone that is available and valid."""
    if not potential_bins_in_zone:
        return None

    row = distance_matrix[current_bin][:]
    candidates = [b for b in potential_bins_in_zone if b in available_bins and b not in current_route]

    if not candidates:
        return None

    closest_bin = min(candidates, key=lambda b: row[b])
    return closest_bin


def find_initial_solution(  # noqa: C901
    data, bins_coordinates, distance_matrix, number_of_bins, vehicle_capacity, E, B, rng
):
    """
    Construct a feasible initial solution for the routing problem.

    Processes overflow predictions and builds routes using a construction heuristic.

    Args:
        data (pd.DataFrame): Bin weights and metadata.
        bins_coordinates (List): Lat/Lon pairs (DataFrame in original context).
        distance_matrix (np.ndarray): Shortest path distances.
        number_of_bins (int): Scale of the problem.
        vehicle_capacity (float): Max tanker capacity.
        E (float): Bin volume.
        B (float): Bin density.
        rng (Random): Random number generator.

    Returns:
        List[List[int]]: Constructed routes.
    """
    bins = list(data["#bin"][1 : number_of_bins + 1])
    depot_id = data["#bin"].iloc[0]

    # Zones logic (simplified from original)
    # Original logic extracted Lng manually
    lng_vals = bins_coordinates["Lng"].values
    min_lng, max_lng = lng_vals.min(), lng_vals.max()
    amp = max_lng - min_lng
    th1 = min_lng + amp / 3
    th2 = min_lng + 2 * amp / 3

    zones = {1: [], 2: [], 3: []}  # type: ignore[var-annotated]
    # Skip depot (index 0)
    for i in range(1, len(lng_vals)):
        lng = lng_vals[i]
        bin_id = i  # Assuming 1-based indexing for bin IDs based on original code
        if bin_id not in bins:
            continue

        if lng < th1:
            zones[1].append(bin_id)
        elif lng < th2:
            zones[2].append(bin_id)
        else:
            zones[3].append(bin_id)

    routes_list = []

    while bins:
        current_route = [depot_id]
        space_occupied = _get_bin_stock(data, depot_id, E, B)
        # Note: Depot typically has 0 stock, but kept for consistency

        current_bin = depot_id

        # Try finding bins in zones 1, then 2, then 3
        # Original code nested while loops for zones.
        # We can streamline this.

        for zone_id in [1, 2, 3]:
            potential_bins = zones[zone_id]

            while space_occupied < vehicle_capacity and potential_bins:
                next_bin = _find_closest_valid_bin(current_bin, potential_bins, bins, current_route, distance_matrix)

                if next_bin is None:
                    break

                # Check capacity
                stock = _get_bin_stock(data, next_bin, E, B)
                if space_occupied + stock > vehicle_capacity:
                    break

                # Add bin
                current_route.append(next_bin)
                space_occupied += stock

                # Update state
                if next_bin in bins:
                    bins.remove(next_bin)
                if next_bin in potential_bins:
                    potential_bins.remove(next_bin)

                current_bin = next_bin

            if space_occupied >= vehicle_capacity:
                break

        # Close route
        current_route.append(depot_id)
        routes_list.append(current_route)

        # Safety break if no bins added (avoid infinite loop)
        if len(current_route) <= 2 and not bins:
            break
        if len(current_route) <= 2 and bins:
            # Force pick one if stuck? Or logic handles it?
            # If stuck, pick first available
            random_bin = rng.choice(bins)
            routes_list[-1].insert(1, random_bin)
            bins.remove(random_bin)
            # Remove from zones
            for z in zones.values():
                if random_bin in z:
                    z.remove(random_bin)

    return routes_list


def compute_initial_solution(data, bins_coordinates, distance_matrix, vehicle_capacity, id_to_index):
    """
    Simplified initial solution builder for SANS policy.

    Args:
        data (pd.DataFrame): Bin metadata.
        bins_coordinates (List): Coordinates.
        distance_matrix (np.ndarray): Distances.
        vehicle_capacity (float): Tanker capacity.
        id_to_index (Dict): Mapping.

    Returns:
        Tuple: (initial_routes, removed_bins).
    """
    depot = data["#bin"].iloc[0]
    all_bins = list(data["#bin"][1:])
    bins_coordinates = {b: coord for b, coord in bins_coordinates.items() if b in all_bins or b == depot}
    stocks = dict(zip(data["#bin"], data["Stock"], strict=False))

    rotas = []
    bins_restantes = set(all_bins)

    # Enquanto houver bins a serem coletados
    while bins_restantes:
        # Inicializa a rota a partir do depósito
        current_route = [depot]
        carga = 0
        atual = depot
        bins_restantes_inicial = set(bins_restantes)

        # Enquanto houver bins restantes e espaço no veículo
        while bins_restantes_inicial:
            # Filtra os bins que podem ser adicionados sem ultrapassar a capacidade do veículo
            candidatos = [b for b in bins_restantes_inicial if carga + stocks[b] <= vehicle_capacity]
            if not candidatos:
                break

            # Encontra o bin mais próximo
            idx_atual = id_to_index[atual]
            proximo = min(candidatos, key=lambda b: distance_matrix[idx_atual][id_to_index[b]])

            # Adiciona o bin mais próximo à rota
            current_route.append(proximo)
            carga += stocks[proximo]
            bins_restantes_inicial.remove(proximo)
            atual = proximo

        # Retorna ao depósito
        current_route.append(depot)

        # Se a rota tiver mais de 2 pontos (depot + bins), adiciona à lista de rotas
        if len(current_route) > 2:
            rotas.append(current_route)

        # Remove os bins que já foram coletados
        bins_restantes = bins_restantes - set(current_route[1:-1])
    return rotas
