"""
Functionality to assist calculating the direct damages to assets due to hazards.
"""

from abc import ABC, abstractmethod
import re
from typing import Union

import pandas as pd
import snkit

from open_gira import fields
from open_gira.utils import natural_sort


class ReturnPeriodMap(ABC):
    """
    Abstract class defining interface for return period hazard maps.
    """

    # identifying string containing other, inferred attributes, should be
    # unique among any collection of maps
    name: str
    # name of scenario, e.g. rcp4p5, historical
    scenario: str
    # year for which hazard map is valid (may be past, present or future)
    year: int
    # expect hazard to recur on average every return_period years
    return_period_years: float

    def __init__(self):
        if type(self) is ReturnPeriodMap:
            raise RuntimeError("Abstract class; please subclass rather than directly instantiating")

    @property
    @abstractmethod
    def without_RP(self):
        """
        A name identifying the kind of hazard, without any return period.

        N.B. For collapsing return periods into expected annual damages (EAD)
        it is useful to generate a name without return period information.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def without_model(self) -> str:
        """
        A name identifying the attributes of the hazard, without any model
        information (climate model / subsidence).
        """
        raise NotImplementedError

    @property
    def annual_probability(self) -> float:
        """Likelihood of hazard occurring in any given year"""
        return 1 / self.return_period_years

    def __eq__(self, other):
        """Maps are equal if their name is"""
        return self.name == other.name

    def __lt__(self, other):
        """
        Permit sorting by annual probability (i.e. least likely events first),
        the usual order for an expected annual damages integration.
        """
        return self.annual_probability < other.annual_probability

    def __hash__(self):
        """Map name string should capture all that is unique about object."""
        return hash(self.name)


class AqueductFlood(ReturnPeriodMap):
    """
    Class holding information about aqueduct return period flood maps.

    Each point in these raster flood maps is an inundation depth for a given
    combination of e.g. scenaro, climate model, year, return period
    (probability).
    """

    # there are two subcategories of aqueduct flood map
    COASTAL = "inuncoast"
    RIVERINE = "inunriver"

    # coastal models may or may not include a subsidence component
    WITH_SUBSIDENCE = "wtsub"
    WITHOUT_SUBSIDENCE = "nosub"

    def __init__(self, name: str):
        """
        Infer attributes from name.

        Arguments:
            name (str): Name string expected to be in one of the following formats:
                Riverine:
                    inunriver_rcp8p5_00000NorESM1-M_2080_rp00005
                Coastal:
                    inuncoast_rcp8p5_wtsub_2080_rp1000_0_perc_50
        """

        if len(name.split(".")) > 1:
            raise ValueError(f"{name=} contains dots; remove any file extension")

        # store the original string for later use
        self.name = name

        map_type, *split_name = name.split("_")

        if map_type == self.RIVERINE:

            # unpack rest of name
            scenario, climate_model, year, return_period_years = split_name

            self.riverine = True
            self.coastal = False
            self.model = climate_model

        elif map_type == self.COASTAL:

            # unpack rest of name
            scenario, sub_str, year, return_period_years, *slr_perc_list = split_name

            if sub_str == self.WITH_SUBSIDENCE:
                subsidence = True
            elif sub_str == self.WITHOUT_SUBSIDENCE:
                subsidence = False
            else:
                raise ValueError(f"malformed aqueduct subsidence string {sub_str=}")

            # sea level rise percentile
            slr_perc_str = "_".join(slr_perc_list)

            # N.B. default is 95th percentile
            if slr_perc_str == "0":
                slr_percentile = 95.0
            elif slr_perc_str == "0_perc_50":
                slr_percentile = 50.0
            elif slr_perc_str == "0_perc_05":
                slr_percentile = 5.0
            else:
                raise ValueError(
                    f"malformed aqueduct sea level percentile string {slr_perc_str=}"
                )

            self.riverine = False
            self.coastal = True
            self.subsidence = subsidence
            self.model = sub_str
            self.slr_percentile = slr_percentile

        else:
            raise ValueError(
                f"do not recognise hazard {map_type=}, "
                f"{name=} must begin with either {self.RIVERINE} or {self.COASTAL}"
            )

        # attributes common to riverine and coastal maps
        self.year = int(year)
        self.return_period_years = float(return_period_years.replace("rp", ""))
        self.scenario = scenario

        return

    @property
    def without_model(self) -> str:
        """
        A name identifying the attributes of the hazard, without any model
        information (climate model / subsidence).
        """
        split_name = self.name.split("_")
        split_name.pop(2)  # index of climate model / subsidence component
        return "_".join(split_name)

    @property
    def without_RP(self) -> str:
        """
        A name identifying the attributes of the hazard, without any return
        period.

        N.B. For collapsing return periods into expected annual damages (EAD)
        it is useful to generate a name without return period information.
        """

        split_name = self.name.split("_")
        split_name.pop(4)  # index of return period element for river and coastal map names
        return "_".join(split_name)


def generate_rp_maps(names: list[str], prefix: Union[None, str] = None) -> list[ReturnPeriodMap]:
    """
    Given a list of strings, generate some ReturnPeriodMap objects. Optionally
    remove a prefix string from the input.
    """
    if prefix is not None:
        names = [re.sub(f"^{prefix}", "", name) for name in names]
    return [get_rp_map(name) for name in natural_sort(names)]


def get_rp_map(name: str) -> ReturnPeriodMap:
    """
    Given a name string, return an instance of the appropriate ReturnPeriodMap
    subclass.
    """

    # registry of implemented return period constructors
    # new return period map types must be added here
    prefix_class_map: dict[str, type[ReturnPeriodMap]] = {
        AqueductFlood.RIVERINE: AqueductFlood,
        AqueductFlood.COASTAL: AqueductFlood,
    }

    # choose constructor on name prefix
    prefix, *_ = name.split("_")

    # return a concrete subclass of ReturnPeriodMap
    return prefix_class_map[prefix](name)


def rail_rehab_cost(row: pd.Series, rehab_costs: pd.DataFrame) -> float:
    """
    Determine the cost of rehabilitation for a given rail segment (row).

    Args:
        row: Railway segment
        rehab_costs: Table of rehabilitation costs for various rail types

    Returns:
        Minimum cost, maximum cost, units of cost
    """

    # bridge should be a boolean type after data cleaning step
    if row.bridge:
        asset_type = "bridge"
    else:
        asset_type = "rail"

    return float(rehab_costs.loc[rehab_costs["asset_type"] == asset_type, "rehab_cost_USD_per_km"])


def road_rehab_cost(row: pd.Series, rehab_costs: pd.DataFrame) -> float:
    """
    Fetch the cost of rehabilitation for a given road segment (row).

    Args:
        row: Road segment
        rehab_costs: Table of rehabilitation costs for various road types

    Returns:
        Cost in units of USD per km per lane.
    """

    # bridge should be boolean after data cleaning step
    if row.bridge:
        highway_type = "bridge"
    else:
        highway_type = row.tag_highway

    if row.paved:
        condition = "paved"
    else:
        condition = "unpaved"

    return float(
        rehab_costs["rehab_cost_USD_per_km_per_lane"].loc[
            (rehab_costs.highway == highway_type) & (rehab_costs.road_cond == condition)
        ].squeeze()
    )


def annotate_rehab_cost(network: snkit.network.Network, network_type: str, rehab_cost: pd.DataFrame) -> snkit.network.Network:
    """
    Label a network with rehabilitation costs to be used in subsequent direct
    damage cost estimate.

    Args:
        network: Network with edges to label.
        network_type: Category string, currently either road or rail.
        rehab_cost: Table containing rehabilitation cost data per km. See lookup
            functions for more requirements.

    Returns:
        Network labelled with rehabilitation cost data.
    """

    if network_type == "road":
        network.edges[fields.REHAB_COST] = network.edges.apply(road_rehab_cost, axis=1, args=(rehab_cost,)) * network.edges.lanes
    elif network_type == "rail":
        network.edges[fields.REHAB_COST] = network.edges.apply(rail_rehab_cost, axis=1, args=(rehab_cost,))
    else:
        raise ValueError(f"No lookup function available for {network_type=}")

    return network