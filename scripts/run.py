import argparse

import mesa
import mesa_geo as mg
from src.model.model import StreetCrime
from src.visualization.server import (
    agent_draw,
    clock_element,
    status_chart,
)

def make_parser():
    parser = argparse.ArgumentParser("Agents and Networks in Python")
    return parser

if __name__ == "__main__":
    city_params = {"crs": "epsg:7791", "resident_speed": 0.5}
    model_params = {
        "buildings_file": r"C:\Users\adiad.SPECTRE\OneDrive - Università Commerciale Luigi Bocconi\Documenti\Università\Third Year\Thesis\thesis\data\raw\DBT_2020\DBT 2020 - SHAPE\EDIFC_CR_EDF_IS.shp",
        "buildings_fun_file": r"C:\Users\adiad.SPECTRE\OneDrive - Università Commerciale Luigi Bocconi\Documenti\Università\Third Year\Thesis\thesis\data\raw\DBT_2020\DBT 2020 - SHAPE\EDIFC_EDIFC_USO.dbf",
        "roads_file": r"C:\Users\adiad.SPECTRE\OneDrive - Università Commerciale Luigi Bocconi\Documenti\Università\Third Year\Thesis\thesis\data\processed\EL_STR.shp",
        "crs": city_params["crs"],
        "show_roads": True,
        "num_residents": mesa.visualization.Slider(
            "Number of residents", value=50, min_value=10, max_value=150, step=10
        ),
        "resident_speed": mesa.visualization.Slider(
            "Resident Walking Speed (m/s)",
            value=city_params["resident_speed"],
            min_value=0.1,
            max_value=1.5,
            step=0.1,
        ),
    }
    map_element = mg.visualization.MapModule(agent_draw, map_height=600, map_width=600)
    server = mesa.visualization.ModularServer(
        StreetCrime,
        [map_element, clock_element, status_chart],
        "Milan Street Crime",
        model_params,
    )
    server.launch()
