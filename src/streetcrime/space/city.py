import os.path  # To check if files exist
import time  # To measure the time of the loading of the files

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from mesa_frames.agent import GeoAgentDF
from pyproj import CRS

# import numpy as np # For weight parser
# from typing import Callable # For **layers
# from scipy.stats import skewnorm # For income distribtion
# from shapely.geometry import Point # for find_neighborhood_by_pos
# from streetcrime.utility import _method_parser #for method parsing
# import json #for downloading from overpass turbo


class City:
    """
    The City class is the GeoSpace in which agents move.
    It is used to store the road & public transport network, the buildings and neighborhoods dataframes.

    Attributes
    ----------
    crs : pyproj.CRS
        The crs of the city. Should be projected.
    city_name : str
        City to pass to `osmnx` to download layers.
    roads_nodes : gpd.GeoDataFrame | None
        The nodes of the road network. None if not loaded.
    roads_edges : gpd.GeoDataFrame | None
        The edges of the road network. None if not loaded.
    roads : nx.DiGraph | None
        The road network. None if not loaded.
    paths : pd.DataFrame
        The dataframe containing all pairs shortest paths of the road network.
    **layers : str | (Callable, list[str]) | (Callable, dict[str, str]
        Currently DEPRECATED, do not use.
        The layers of the city. Can be 'roads', 'public_transport', 'neighborhoods' or 'buildings'
        If str, it is the path to the file containing the layer
        If (Callable, dict[str, str]) it is a method of city with keyword parameters of the associated function
        If (Callable, list[str]) it is a method of city with ordered parameters of the associated function

    """

    def __init__(self, crs: CRS, city_name: str) -> None:
        """
        Parameters
        ----------
        crs : pyproj.CRS
            The crs of the city. Should be projected.
        city_name : str
            City to pass to `osmnx` to download layers.

        Raises:
        ----------
        ValueError
            If `crs` is not projected
        """

        if not CRS(crs).is_projected:
            raise ValueError("Input CRS has to be projected")

        self.crs = crs
        self.city_name = city_name
        self.roads_nodes: gpd.GeoDataFrame | None = None
        self.roads_edges: gpd.GeoDataFrame | None = None
        self.roads: nx.DiGraph | None = None

        # Read paths files
        if os.path.isfile(f"outputs/city/{self.city_name.split(',')[0]}_paths.csv"):
            self.paths: pd.DataFrame = pd.read_csv(
                f"outputs/city/{self.city_name.split(',')[0]}_paths.csv"
            )
            # need to read strings as list types for .explode(), faster than using ast.literal_eval
            self.paths.path = self.paths.path.str.strip("[]").str.split(", ")
        else:
            self.paths: pd.DataFrame = pd.DataFrame(
                columns=["node", "destination", "path"]
            )

        # If additonal layers are specified
        """if len(layers) > 0:
            print("Loading layers : ...")
            for layer, arg in layers.items():
                if os.path.isfile(arg):
                    layer_time = time.time()
                    print(f"Loading {layer} from {arg}")
                    match layer:
                        case "neighborhoods":
                            self.neighborhoods = gpd.read_file(arg).set_index('id')
                            self.income_distribution = skewnorm(a = self.neighborhoods['city_ae'].iloc[0], 
                                                                loc = self.neighborhoods['city_loce'].iloc[0], 
                                                                scale = self.neighborhoods['city_scalee'].iloc[0]) 
                            self.neighborhoods.drop(columns = ['city_ae', 'city_loce', 'city_scalee'], inplace = True)
                        case _:
                            setattr(self, layer, gpd.read_file(arg))
                    print(f"Loaded {layer}: " + "--- %s seconds ---" % (time.time() - layer_time))
                else:
                    setattr(self, layer, _method_parser(self, arg))"""

    def load_data(
        self,
        tolerance: int = 15,
        traffic_factor: int = 1,
        roads_file: str | None = None,
        buildings_file: str | None = None,
        building_categories: pd.DataFrame | None = None,
        boundaries_file: str | None = None,
    ) -> None:
        """Obtains the data of the city from OSMNX and saves it in the specified files
        or reads data from the file passed as arguments.

        Parameters
        ----------
        tolerance : int, default = 15
            The tolerance according to which intersectation are consolidated with `osmnx.simplification.consolidate_intersections`.
        traffic_factor : int, default = 1
            The factor with which speeds will be refactored. If not specified, velocities will be set to max speed per road.
        roads_file : str, default = None
            The path to the file containing the roads or to save the roads to if not present. If None, the default path is outputs/city/{city_name}_roads.gpkg
        buildings_file : str, default = None
            The path to the file containing the buildings or to save the buildings to if not present. If None, he default path is outputs/city/{city_name}_buildings.gpkg
        building_categories : pd.DataFrame, default = None
            The dataframe containing the categories of the buildings. Look at source code for default values.

        Raises
        ------
        ValueError
            If `traffic_factor` is not between 0 and 1 or if `tolerance` is not greater than 0
        """

        # Check parameters
        if not 0 < traffic_factor <= 1:
            raise ValueError(
                f"Traffic factor set to {traffic_factor}. Should be between 0 and 1"
            )
        if tolerance <= 0:
            raise ValueError(f"Tolerance set to {tolerance}. Should be greater than 0")

        if not roads_file:
            if not os.path.isdir("outputs/city"):
                os.makedirs("outputs/city")
            roads_file = f"outputs/city/{self.city_name.split(',')[0]}_roads.gpkg"

        # Obtain roads
        if os.path.isfile(roads_file):
            print(f"Roads already downloaded in {roads_file}. Loading...")
            if ".gpkg" in roads_file:
                # TODO: remove useless columns
                self.roads_nodes = gpd.read_file(roads_file, layer="nodes").set_index(
                    "osmid"
                )
                self.roads_edges = gpd.read_file(roads_file, layer="edges").set_index(
                    ["u", "v", "key"]
                )
                self.roads = nx.DiGraph(
                    ox.graph_from_gdfs(self.roads_nodes, self.roads_edges)
                )
                self.roads_edges = (
                    self.roads_edges.sort_values("travel_time")
                    .reset_index()
                    .drop(columns=["key"])
                    .groupby(["u", "v"])
                    .first()
                )
                return

        roads = self._obtain_roads(tolerance, traffic_factor)

        # Obtain buildings
        buildings = self._obtain_buildings(buildings_file, building_categories)
        buildings["nearest_node"] = buildings.geometry.apply(
            lambda x: ox.nearest_nodes(roads, x.centroid.x, x.centroid.y)
        )
        buildings = buildings.groupby("nearest_node").count()
        buildings = buildings[["home", "work", "activity", "open_day", "open_night"]]
        nx.set_node_attributes(roads, buildings.transpose().to_dict())
        self.roads_nodes = ox.graph_to_gdfs(roads, nodes=True, edges=False)
        ox.save_graph_geopackage(roads, filepath=roads_file, directed=True)
        print(f"Saved roads in {roads_file}")
        self.roads = nx.DiGraph(roads)

        # Save boundaries
        (
            ox.geocoder.geocode_to_gdf(self.city_name)
            .to_crs(self.crs)
            .to_file(f"outputs/city/{self.city_name.split(',')[0]}_boundaries.gpkg")
        )

    def get_random_nodes(
        self,
        function: str | None = None,
        time: str | None = None,
        agent: GeoAgentDF | None = None,
        decision_rule: str | None = None,
        n=1,
    ) -> int | pd.Series:
        """
        This method act on the City.roads_nodes to obtain a random node id based on the function and the agent that requested it.

        Parameters
        ----------
        function : str | None, optional
            The function of the node that the agent is looking for, can be "home", "work", "act". By default None does not impose a restriction.
        time : str | None, optional
            The time of the day that the agent is looking for, can be "open_day" or "open_night". By default None does not impose a restriction.
        agent : GeoAgentDF | None, optional
            FUTURE IMPLEMENTATION: The agent that is looking for a node, by default None.
        decision_rule : str | None, optional
            FUTURE IMPLEMENTATION: The decision rule that the agent is using to choose the node, by default None.
        n : int, optional
            The number of nodes to return, by default 1.

        Returns
        -------
        int | pd.Series[int]
            The id of the node or a series of ids of the nodes.
        """
        weights = None

        if function and time:
            weights = self.roads_nodes[function] * self.roads_nodes[time]
        elif function:
            weights = self.roads_nodes[function]

        # if decision_rule:
        #    node = self.roads_nodes.sample(n = n)

        if n == 1:
            return self.roads_nodes.sample(n=1, weights=weights).index[0]
        else:
            return self.roads_nodes.sample(n=n, weights=weights, replace=True).index

    def _obtain_roads(
        self, tolerance: int = 15, traffic_factor: int = 1
    ) -> nx.MultiDiGraph:
        """Download roads using `osmnx.graph_from_place`

        Parameters
        ----------
        tolerance : int, default = 15
            The tolerance according to which intersectation are consolidated with `osmnx.simplification.consolidate_intersections`.
        traffic_factor : int, default = 1
            The factor with which speeds will be refactored. If not specified, velocities will be set to max speed per road.


        Returns
        -------
        `networkx.MultiDiGraph`
            returns a networkx Multi Directed Graph with the road network of the city
        """
        start_time = time.time()
        print(f"Downloading roads of {self.city_name}...")

        # Downloading the full dataset of roads
        roads = ox.graph_from_place(
            self.city_name
        )  # simplification is already activated
        roads = ox.projection.project_graph(roads, to_crs=self.crs)

        if tolerance > 0:
            roads = ox.simplification.consolidate_intersections(
                roads, tolerance=tolerance
            )

        roads = ox.speed.add_edge_speeds(roads)
        nodes, edges = ox.graph_to_gdfs(roads, nodes=True, edges=True)

        # Calculate the speed of each edge
        if traffic_factor != 1:
            edges["speed_kph"] = edges["speed_kph"] * traffic_factor
            roads = ox.graph_from_gdfs(nodes, edges)

        # Add travel times
        roads = ox.speed.add_edge_travel_times(roads)

        # Remove nodes without outgoing edges
        out_degree = pd.Series(dict(roads.out_degree))
        roads.remove_nodes_from(out_degree[out_degree == 0].index)

        # Find all pairs shortest paths (faster than lazy computation at each iteration)
        paths = dict(nx.all_pairs_dijkstra_path(roads, weight="travel_time"))
        data = []
        # Iterate through the nested dictionary
        for origin, destinations in paths.items():
            for destination, path in destinations.items():
                # Append a tuple with the origin, destination, and path to the list
                data.append((origin, destination, path))
        paths = pd.DataFrame(data, columns=["node", "destination", "path"])
        paths.astype({"node": "int64", "destination": "float64"})
        paths.to_csv(
            f"outputs/city/{self.city_name.split(',')[0]}_paths.csv", index=False
        )
        self.paths = paths

        print("Downloaded roads: " + "--- %s seconds ---" % (time.time() - start_time))
        return roads

    def _obtain_buildings(
        self,
        buildings_file: str | None = None,
        building_categories: pd.DataFrame | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Download buildings using `osmnx.features_from_place`

        Parameters
        ----------
        buildings_file : str, default = None
            The path to the file containing the buildings or to save the buildings to if not present. If None, he default path is outputs/city/{city_name}_buildings.gpkg
        building_categories : pd.DataFrame, default = None
            The dataframe containing the categories of the buildings. Look at source code for default values.

        Returns
        -------
        `geopandas.GeoDataFrame`
            returns a geopandas GeoDataFrame with the buildings of the city
        """
        start_time = time.time()

        if not buildings_file:
            buildings_file = (
                f"outputs/city/{self.city_name.split(',')[0]}_buildings.gpkg"
            )

        print(f"Downloading buildings of {self.city_name}...")
        buildings = ox.features_from_place(self.city_name, tags={"building": True})
        print(
            "Downloaded buildings: " + "--- %s seconds ---" % (time.time() - start_time)
        )
        buildings.to_crs(self.crs, inplace=True)
        buildings.reset_index(inplace=True)
        buildings = buildings[["geometry", "building"]]
        # categorize buildings
        if not building_categories:
            building_categories = [
                ["apartments", True, None, None, True, None],
                ["barracks", True, None, None, True, None],
                ["bungalow", True, None, None, True, True],
                ["cabin", True, None, True, True, True],
                ["detached", True, None, None, True, True],
                ["dormitory", True, None, None, True, True],
                ["farm", True, True, None, True, True],
                ["ger", True, None, None, True, True],
                ["hotel", None, True, True, True, True],
                ["house", True, None, None, True, True],
                ["houseboat", True, None, True, True, True],
                ["residential", True, None, None, True, True],
                ["semidetached_house", True, None, None, True, True],
                ["static_caravan", True, None, None, True, True],
                ["stilt_house", True, None, True, True, True],
                ["terrace", True, None, None, True, True],
                ["tree_house", True, None, True, True, None],
                ["trullo", True, None, None, True, True],
                ["commercial", None, True, True, True, None],
                ["industrial", None, True, None, True, None],
                ["kiosk", None, True, True, True, True],
                ["office", None, True, None, True, None],
                ["retail", None, True, True, True, True],
                ["supermarket", None, True, True, True, True],
                ["warehouse", None, True, None, True, None],
                ["church", None, True, True, True, None],
                ["mosque", None, True, True, True, None],
                ["synagogue", None, True, True, True, None],
                ["temple", None, True, True, True, None],
                ["school", None, True, None, True, None],
                ["university", None, True, None, True, None],
                ["hospital", None, True, True, True, True],
                ["fire_station", None, True, None, True, True],
                ["government", None, True, None, True, None],
                ["yes", True, True, True, True, True],
            ]
            columns = ["building", "home", "work", "activity", "open_day", "open_night"]
            building_categories = pd.DataFrame(building_categories, columns=columns)
        buildings = buildings.merge(building_categories, on="building", how="left")
        if buildings_file:
            buildings.to_file(buildings_file)
            print(f"Saved buildings in {buildings_file}")
        return buildings

    """TODO: implement public transport retrieval without passing through overpass turbo
    def obtaining_public_transport(self):
        # Load JSON data from Overpass API
        data = json.load(your_json_data)

        # Process relations
        for relation in data["elements"]:
            if relation["type"] == "relation":
                # Initialize an empty list to store line strings
                line_strings = []
                for member in relation["members"]:
                    if member["type"] == "way":
                        points = [
                            (node["lon"], node["lat"]) for node in member["geometry"]
                        ]
                        line_strings.append(LineString(points))

                # Combine line strings into a single geometry
                # This could be a MultiLineString or another appropriate geometry type
                relation_geometry = combine_line_strings(line_strings)

                # Do something with the relation geometry"""

    '''TODO: delete, deprecated by get_random_nodes()
    def get_random_building(self, 
                            function: str = None,
                            agent : mesa.Agent = None,
                            decision_rule : str = None)-> int:
        """This method act on the City.buildings_df to obtain a random building id based on the function and the agent that requested it.
        Args:
            agent (agent, optional): The agent that is looking for a building, by default None.
            function (str, optional): The function of the building that the agent is looking for, by default None. Can be "home", "work", "day_act" or "night_act".  
        
        Returns:
            int: The id of the building
        
        """
        #TODO: maybe weights for home should also be in the buildings df?
        if decision_rule is None:
            _building = self.buildings[self.buildings['home'] == 1].sample(n = 1)
        else:
            weights = self._weights_parser(agent, decision_rule)
            if function == "home":
                #_neighborhood = self.neighborhoods.sample(n=1, weights=weights)
                # _building = self.buildings[(self.buildings['home']) & (self.buildings['neighborhood'] == _neighborhood.index[0])].sample(n = 1)
                _building = self.buildings[self.buildings['home'] == 1].sample(n = 1)
            else:
                _building = self.buildings[self.buildings[function] == 1].sample(n=1, weights=weights)
        return _building.index[0]'''

    '''TODO: implement with DF
      def find_neighborhood_by_pos(self, position: Point) -> int:
        """Find the neighborhood in which the position passed as argument is contained

        Parameters
        ----------
            position (Point): The position to find the containing neighborhood

        Returns:
        ----------
            int: The id of the neighborhood
        """
        try:
            intersecting_neighborhoods = self.neighborhoods[self.neighborhoods.geometry.contains(position)]
            index = intersecting_neighborhoods.index[0]
        except IndexError:
            warn(f"Position {position} is not contained in any neighborhood", RuntimeWarning)
            index = None
        return index'''

    """#TODO: delete, deprecated by current implementation
    def _weights_parser(self, agent: mesa.Agent, decision_rule : str):
        decision_rule = decision_rule.split(",")
        df = getattr(self, decision_rule[0]) 
        df = pd.eval(decision_rule[1].strip(), target=df)        
        return df.weights.replace(np.inf, 0)"""
