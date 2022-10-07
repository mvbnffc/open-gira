"""
Test runner and file comparison code for snakemake integration tests.
"""

import json
import os
import re
import shutil
import sys
import subprocess as sp
from pprint import pformat
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import geopandas as gpd
import pandas as pd


def printerr(s: str):
    """
    Print to stderr (for visibility when invoked via pytest).

    N.B. To view STDOUT as well as STDERR, invoke pytest with -s flag.
    """
    print(s, file=sys.stderr)


def run_snakemake_test(rule_name: str, targets: Tuple[str]):
    """
    Create a temporary working directory, copy input files to it, run a
    snakemake rule and compare the outputs of that rule with some expected
    output.

    Args:
        rule_name (str): Name of rule to invoke
        targets (tuple[str]): Desired output files and directories for
            snakemake to generate
    """

    if not isinstance(targets, tuple):
        raise TypeError(f"Expect desired outputs as tuple, got {type(targets)=}")

    with TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir) / "workdir"
        results_dir = workdir / "results"
        data_path = Path(f"tests/integration/{rule_name}/data")
        expected_path = Path(f"tests/integration/{rule_name}/expected")

        # check necessary testing data is present
        assert data_path.exists()
        assert expected_path.exists()

        # Copy data to the temporary workdir.
        shutil.copytree(data_path, workdir)
        auxilliary_dirs = ["config", "external_files", "bundled_data"]
        for folder in auxilliary_dirs:
            folder_path = workdir / folder
            shutil.copytree(f"tests/{folder}", folder_path)
            assert folder_path.exists()

        # print to stdout information available when pytest run with -s flag
        # or on test failure
        print("\n\nFile system prior to running:")
        os.system(f"tree -l {results_dir}")
        print("")

        command_args = [
            "python",
            "-m",
            "snakemake",
            "-c1",  # single core
            "--reason",  # show snakemake's reasoning, helps with debugging
            "--configfile",  # use test specific configuration
            "tests/config/config.yaml",
            "--allowed-rules",  # only run the specified rule, no precursors
            rule_name,
            "--directory",  # use the temporary directory to work in
            str(workdir),
            *targets,  # files/folders/rules to create/run
        ]
        command = " ".join(command_args)
        print(command)
        os.system(command)

        print("\nFile system post run:")
        os.system(f"tree -l {results_dir}")

        print("\nRequired results post run:")
        os.system(f"tree -l {expected_path / 'results'}")

        OutputChecker(data_path, expected_path, workdir, auxilliary_dirs).check()


class OutputChecker:
    def __init__(self, data_path, expected_path, workdir, ignore_folders):
        self.data_path = data_path
        self.expected_path = expected_path
        self.workdir = workdir
        self.ignore_folders = ignore_folders

    def check(self):
        input_files = set(
            (Path(path) / f).relative_to(self.data_path)
            for path, subdirs, files in os.walk(self.data_path)
            for f in files
        )
        expected_files = set(
            (Path(path) / f).relative_to(self.expected_path)
            for path, subdirs, files in os.walk(self.expected_path)
            for f in files
        )
        unexpected_files = set()

        produced_files = []
        for path, subdirs, files in os.walk(self.workdir):
            if any([folder for folder in self.ignore_folders if folder in path]):
                # skip comparison for these folders
                continue
            for f in files:
                f = (Path(path) / f).relative_to(self.workdir)
                if ".snakemake" in str(f):
                    # ignore .snakemake/, foo/bar/.snakemake_timestamp, etc.
                    continue
                if f in expected_files:
                    produced_files.append(Path(path) / f)
                    self.compare_files(self.workdir / f, self.expected_path / f)
                elif f in input_files:
                    # ignore input files
                    pass
                else:
                    unexpected_files.add(f)
        else:
            # the loop could exit successfully if no files at all are are
            # found, so check that we've got the right number
            if len(produced_files) != len(expected_files):
                raise ValueError(
                    f"\n{len(produced_files)=} but {len(expected_files)=}"
                    f"\n{produced_files=}"
                    f"\n{expected_files=}"
                )

        if unexpected_files:
            raise ValueError(
                "Unexpected files: {}".format(sorted(map(str, unexpected_files)))
            )

    def compare_files(self, generated_file: Path, expected_file: Path) -> None:
        """
        Compare two files to check if they are equal by some definition.

        Methods vary by filetype.
        """

        printerr(f">>> Compare files:\n{generated_file}\n{expected_file}")

        # PARQUET
        if re.search(r"\.(geo)?parquet$", str(generated_file), re.IGNORECASE):
            if re.search(r"\.geoparquet$", str(generated_file), re.IGNORECASE):
                """
                NOTE: This test will **fail** if the geoparquet file does not contain geography data columns.
                This can happen where the convert_to_geoparquet job does not find any roads to write.
                We leave this failure in because it is usually unintentional that you're testing with a
                dataset where _none_ of the slices have road data, and these tests should be targeted at
                slices that _do_ have road data.
                """
                read = gpd.read_parquet
            elif re.search(r"\.parquet$", str(generated_file), re.IGNORECASE):
                read = pd.read_parquet
            else:
                raise RuntimeError(f"couldn't identify read function for {generated_file}")

            generated = read(generated_file)
            expected = read(expected_file)

            self.compare_dataframes(generated, expected)

        # JSON
        elif re.search(r"\.(geo)?json$", str(generated_file), re.IGNORECASE):
            with open(generated_file, 'r') as fp:
                generated = json.load(fp)
            with open(expected_file, 'r') as fp:
                expected = json.load(fp)

            if json.dumps(generated, sort_keys=True) != json.dumps(expected, sort_keys=True):
                printerr(">>> Method: compare sorted JSON strings")
                printerr(f">>> generated:\n{pformat(generated)}")
                printerr(f">>> expected:\n{pformat(expected)}")
                raise AssertionError("JSON files do not match")

        # JPG, PDF, PNG & SVG images
        elif re.search(r"\.(jpg|jpeg|pdf|png|svg|tif|tiff)$", str(generated_file), re.IGNORECASE):
            try:
                sp.check_output(["tests/visual_compare.sh", generated_file, expected_file])
            except sp.CalledProcessError as e:
                printerr(">>> Method: visual hash comparison (imagemagick's identify)")
                printerr(f">>> ERROR:\n>>> {e.stdout}")
                raise e

        # any other file type
        else:
            try:
                sp.check_output(["cmp", generated_file, expected_file])
            except sp.CalledProcessError as e:
                printerr(">>> Method: binary comparison (cmp)")
                printerr(f">>> ERROR:\n>>> {e.stdout}")
                raise e

        printerr(">>> Files are a match")

    @staticmethod
    def compare_dataframes(generated: pd.DataFrame, expected: pd.DataFrame) -> None:
        """
        Compare two dataframes, raise ValueError if they aren't the same.
        """
        # after sorting the columns so they're in the same order,
        # use dataframe.equals to quickly check for complete table equality
        # unfortunately there is an edge case this method doesn't catch...
        if not generated.sort_index(axis="columns").equals(expected.sort_index(axis="columns")):
            printerr(">>> Method: compare (geo)pandas dataframes")

            # do some basic shape and schema checks
            if len(generated) != len(expected):
                raise ValueError(
                    f"tables not of same length, {len(generated)=} & {len(expected)=}"
                )
            if difference := set(generated.columns) ^ set(expected.columns):
                raise ValueError(
                    f"tables do not have same schema: {difference=} cols are in one but not both"
                )

            # there is a case where df.equals(identical_df) can return False despite all elements being equal
            # this is when comparing Nones in the same position: https://github.com/pandas-dev/pandas/issues/20442
            mismatch_cols = set()
            for col in generated.columns:
                if any(generated[col] != expected[col]):
                    mismatch_cols.add(col)

            for col in mismatch_cols:

                # do the discrepancies occur only where there are null values (NaN & None)?
                unequal_only_where_null = all(expected[col].isna() == (expected[col].values != generated[col].values))
                if not unequal_only_where_null:
                    printerr(f"{col=} {unequal_only_where_null=}")

                    # let's try and find failing rows by converting to str
                    MAX_FAILURES_TO_PRINT = 5
                    failures = 0
                    for row in range(len(generated)):
                        gen_str = str(generated[col][row: row + 1].values)
                        exp_str = str(expected[col][row: row + 1].values)
                        if gen_str != exp_str:
                            failures += 1
                            if failures > MAX_FAILURES_TO_PRINT:
                                continue
                            else:
                                printerr(f">>> FAILURE at {col=}, {row=}: {gen_str} != {exp_str}")

                    if failures > 0:
                        raise ValueError(f"{failures} row mismatch(es) between tables")

                else:
                    # None != None according to pandas, and this is responsible for the apparent mismatch
                    # we can safely say that this column is equal
                    continue
