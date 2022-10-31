"""
Given an exposure estimate and some damage curves, calculate the damage
fraction for exposed assets.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
import logging
import sys
import warnings
from glob import glob
from os.path import splitext, basename, join

import geopandas as gpd
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import simpson

import utils
from plot_damage_distributions import natural_sort


# exposure table hazard intensity fields expected to be prefixed as such
HAZARD_PREFIX = "hazard-"
# exposure table field containing the cost to rebuild per unit length
REHABILITATION_COST_FIELD = "rehab_cost_USD_per_km"
# length of edge calculated after intersection
SPLIT_LENGTH_FIELD = "length_km"


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
    def family_name(self):
        """
        A name identifying the kind of hazard, without any return period.

        N.B. For collapsing return periods into expected annual damages (EAD)
        it is useful to generate a name without return period information.
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
            self.climate_model = climate_model

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
    def family_name(self) -> str:
        """
        A name identifying the attributes of the hazard, without any return
        period.

        N.B. For collapsing return periods into expected annual damages (EAD)
        it is useful to generate a name without return period information.
        """

        split_name = self.name.split("_")
        split_name.pop(4)  # index of return period element for river and coastal map names
        return "_".join(split_name)


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


def load_damage_curves(damage_curves_dir: str, hazard_type: str, asset_types: set) -> dict[str, pd.DataFrame]:
    """
    Load damage curves from CSVs on disk

    Expected to reside in following structure:
    <damage_curves_dir>/<hazard_type>/<asset_type>.csv

    Damage curve files may have comments, these are lines starting with COMMENT_PREFIX

    Args:
        damage_curves_dir (str): Path to folder containing hazards
        hazard_type (str): Name of hazard folder containing asset specific curves
        asset_types (set): Asset types we require damage curves for

    Returns (dict[str, pd.DataFrame):
        Mapping from asset_type to respective damage curve
    """

    # lines beginning with this character will be ignored by pandas
    COMMENT_PREFIX: str = "#"

    # fetch damage curves for relevant hazard type
    damage_curve_paths = glob(join(damage_curves_dir, hazard_type, "*.csv"))

    damage_curves: dict[str, pd.DataFrame] = {
        # curves expected to be named as a value of Asset class, e.g. RoadAssets.BRIDGE -> road_bridge.csv
        # dict is asset_type: dataframe with hazard intensity [0, inf] and damage fraction [0, 1]
        splitext(basename(path))[0]: pd.read_csv(path, comment=COMMENT_PREFIX) for path in damage_curve_paths
    }

    for asset_type, damage_curve in damage_curves.items():
        # check hazard intensity and damage fraction values are 0 or positive real
        assert ((damage_curve >= 0).all()).all()
        # check damage fraction is less than or equal to 1
        assert (damage_curve.iloc[:, 1] <= 1).all()

    if not set(damage_curves.keys()).issuperset(asset_types):
        raise RuntimeError(f"requested {asset_types=} not all found: {damage_curves.keys()=}")

    return damage_curves


def rejoin_and_save(left: pd.DataFrame, right: gpd.GeoDataFrame, path: str) -> None:
    """
    Take a left dataframe (assets, hazard columns) and a right dataframe (assets,
    non-hazard columns) and join, validate and save the result to disk.

    Arguments:
        left (pd.DataFrame): Typically numeric data, e.g. inundation, damage
            cost, etc. per asset
        right (gpd.GeoDataFrame): Typically non-hazard data, e.g. geometry,
            metadata, etc. per asset.
    """

    joined = gpd.GeoDataFrame(left.join(right, validate="one_to_one"))
    joined = joined[natural_sort(joined.columns)]

    assert len(right) == len(joined)
    assert len(left) == len(joined)
    assert "edge_id" in joined.columns

    logging.info(f"{joined.shape}")
    joined.to_parquet(path)

    return


if __name__ == "__main__":

    try:
        unsplit_path: str = snakemake.input["unsplit"]
        exposure_path: str = snakemake.input["exposure"]
        damage_fraction_path: str = snakemake.output["damage_fraction"]
        damage_cost_path: str = snakemake.output["damage_cost"]
        expected_annual_damages_path: str = snakemake.output["expected_annual_damages"]
        damage_curves_dir: str = snakemake.config["direct_damages"]["curves_dir"]
        network_type: str = snakemake.params["network_type"]
        hazard_type: str = snakemake.params["hazard_type"]
        asset_types: set[str] = set(snakemake.config["direct_damages"]["asset_types"])
    except NameError:
        raise ValueError("Must be run via snakemake.")

    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

    # Ignore geopandas parquet implementation warnings
    # NB though that .geoparquet is not the format to use for archiving.
    warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")

    # load curves first so if we fail here, we've failed early
    # and we don't try and load the (potentially large) exposure file
    damage_curves = load_damage_curves(damage_curves_dir, hazard_type, asset_types)
    logging.info(f"Available damage curves: {damage_curves.keys()}")

    logging.info("Reading exposure (network/raster intersection) data")
    exposure: gpd.GeoDataFrame = gpd.read_parquet(exposure_path)
    logging.info(f"{exposure.shape=}")

    if exposure.empty:
        logging.info("No data in geometry column, writing empty files.")

        # snakemake requires that output files exist, even if empty
        for path in (damage_fraction_path, damage_cost_path, expected_annual_damages_path):
            utils.write_empty_frames(path)
        sys.exit(0)  # exit gracefully so snakemake will continue

    # column groupings for data selection
    hazard_columns = [col for col in exposure.columns if col.startswith(HAZARD_PREFIX)]
    non_hazard_columns = list(set(exposure.columns) - set(hazard_columns))

    # calculate damages for assets we have damage curves for
    damage_fraction_by_asset_type = []
    logging.info(f"Exposed assets {set(exposure.asset_type)}")
    for asset_type in set(exposure.asset_type) & set(damage_curves.keys()):

        logging.info(f"Processing {asset_type=}")
        damage_curve: pd.DataFrame = damage_curves[asset_type]

        # pick out rows of asset type and columns of hazard intensity
        asset_type_mask: gpd.GeoDataFrame = exposure.asset_type == asset_type
        asset_exposure: pd.DataFrame = pd.DataFrame(exposure.loc[asset_type_mask, hazard_columns])

        # create interpolated damage curve for given asset type
        hazard_intensity, damage_fraction = damage_curve.iloc[:, 0], damage_curve.iloc[:, 1]
        # if curve of length n, where x < x_0, y = y_0 and where x > x_n, y = y_n
        bounds = tuple(f(damage_fraction) for f in (min, max))
        interpolated_damage_curve = interp1d(hazard_intensity, damage_fraction, kind='linear', fill_value=bounds, bounds_error=False)

        # apply damage_curve function to exposure table
        # the return value of interpolated_damage_curve is a numpy array
        logging.info("Calculating damage fractions")
        damage_fraction_for_asset_type = pd.DataFrame(
            interpolated_damage_curve(asset_exposure),
            index=asset_exposure.index,
            columns=asset_exposure.columns
        )

        # store the computed direct damages and any columns we started with
        # (other than exposure)
        damage_fraction_by_asset_type.append(
            pd.concat(
                [
                    damage_fraction_for_asset_type,
                    exposure.loc[asset_type_mask, non_hazard_columns]
                ],
                axis="columns"
            )
        )

    # concatenate damage fractions for different asset types into single dataframe
    damage_fraction: gpd.GeoDataFrame = gpd.GeoDataFrame(pd.concat(damage_fraction_by_asset_type))
    # write to disk
    logging.info(f"Writing {damage_fraction.shape=} to disk")
    damage_fraction.to_parquet(damage_fraction_path)

    # multiply the damage fraction estimates by a cost to rebuild the asset
    # units are: 1 * USD/km * km = USD
    logging.info("Calculating direct damage costs")
    direct_damages_only = damage_fraction[hazard_columns] \
        .multiply(damage_fraction[REHABILITATION_COST_FIELD], axis="index") \
        .multiply(damage_fraction[SPLIT_LENGTH_FIELD], axis="index")

    logging.info("Reading raw network data for unsplit geometry")
    unsplit: gpd.GeoDataFrame = gpd.read_parquet(unsplit_path)
    logging.info(f"{unsplit.shape=}")

    # join the other fields with the direct damage estimates
    logging.info("Unifying rasterised segments and summing damage costs")

    # grouping on edge_id, sum all direct damage estimates to give a total dollar cost per edge
    direct_damages = pd.concat(
        [direct_damages_only, damage_fraction["edge_id"]],
        axis="columns"
    ).set_index("edge_id")
    grouped_direct_damages = direct_damages.groupby(direct_damages.index).sum()

    # integrate over return periods for expected annual damages
    # remove the prefix we added earlier in the pipeline to mark these columns as hazard estimates
    hazard_name_no_prefix: list[str] = [(col.replace(HAZARD_PREFIX, "")) for col in natural_sort(hazard_columns)]
    rp_maps: list[ReturnPeriodMap] = [get_rp_map(name) for name in hazard_name_no_prefix]

    # generate a mapping from a 'family' of hazards to their set of return period maps
    rp_map_families: dict[str, set[ReturnPeriodMap]] = defaultdict(set)
    for rp_map in rp_maps:
        rp_map_families[rp_map.family_name].add(rp_map)

    expected_annual_damages = {}
    for family_name, rp_maps in rp_map_families.items():

        sorted_rp_maps: list[ReturnPeriodMap] = sorted(rp_maps)

        # [0, 1] valued decimal probabilities (least to most probable now we've sorted)
        probabilities: list[float] = [rp_map.annual_probability for rp_map in sorted_rp_maps]
        # family subset of grouped_direct_damages
        family_column_names: list[str] = [f"{HAZARD_PREFIX}{rp_map.name}" for rp_map in sorted_rp_maps]
        family_direct_damages: pd.DataFrame = grouped_direct_damages[family_column_names]

        # integrate the damage as a function of probability curve using Simpson's rule
        # Simpson's rule as the function to be integrated is non-linear
        expected_annual_damages[family_name] = simpson(family_direct_damages, x=probabilities, axis=1)

    # lose columns like "cell_indicies" or rastered length measures that are specific to _rastered_ edges
    non_hazard_output_columns = list(set(non_hazard_columns) & set(unsplit.columns))
    unsplit_subset = unsplit[non_hazard_output_columns].set_index("edge_id", drop=False)

    # rejoin cost estimates with geometry and metadata columns and write to disk
    logging.info("Writing out direct damages per return period map")
    rejoin_and_save(grouped_direct_damages, unsplit_subset, damage_cost_path)
    logging.info("Writing out expected annual damages")
    rejoin_and_save(
        pd.DataFrame(data=expected_annual_damages, index=unsplit_subset.index),
        unsplit_subset,
        expected_annual_damages_path
    )

    logging.info("Done")
