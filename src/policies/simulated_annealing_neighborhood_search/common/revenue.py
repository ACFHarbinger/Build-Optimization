"""
Revenue calculations for SANS.
"""


def compute_waste_collection_revenue(routes_list, data, E, B, R):
    """
    Calculate the total revenue from waste collected in the current routes.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        data (pd.DataFrame): Bin data (Stock and Accum_Rate).
        E (float): Bin volume.
        B (float): Bin density.
        R (float): Revenue per kg.

    Returns:
        float: Total collection revenue.
    """
    total_revenue_route = 0
    total_revenue = 0
    for i in routes_list:
        for j in i:
            stock = data["Stock"][j]
            accumulation_rate = data["Accum_Rate"][j]
            bin_stock = stock + accumulation_rate
            revenue_per_bin = bin_stock * E * B * R
            total_revenue_route += revenue_per_bin
        total_revenue += total_revenue_route
        total_revenue_route = 0
    return total_revenue
