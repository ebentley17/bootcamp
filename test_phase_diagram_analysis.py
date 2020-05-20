import numpy as np
import pandas as pd
import pytest
import glob

import phase_diagram_analysis as pda


def make_test_data():
    test_df = pd.DataFrame(
        data={
            "Sample ID": ["Peptide_100_0.5", "Buffer_100_0.5", "Peptide_100_0.5"],
            "1 (nm)": [350, 350, 350],
            "1 (Abs)": [1, 2, 3],
            "2 (nm)": [600, 600, 600],
            "2 (Abs)": [1, 2, 3],
        }
    )

    return test_df


def test_clean_up_columns():
    messy_df = pd.DataFrame(
        data={
            "real_column": [1.0, 2.0, 3.0, np.nan],
            "Unnamed 1": [1, 2, 3, np.nan],
            "Unnamed 2": [1, 2, 3, np.nan],
            "User name": [1, 2, 3, np.nan],
            "#": [1, 2, 3, np.nan],
            "column_nas": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    cleaned = pda.clean_up_columns(messy_df)

    bad_col_names = ["Unnamed", "User name", "#", "column_nas"]
    for name in bad_col_names:
        assert name not in cleaned.columns

    assert len(cleaned["real_column"]) == 3

    # running on a pre-cleaned df should have no effect
    cleaned_again = pda.clean_up_columns(cleaned)
    assert (cleaned_again == cleaned).all().all()


def test_rename_abs_columns_by_wavelength_simple_case():
    renamed = pda.rename_abs_columns_by_wavelength(make_test_data())

    bad_col_names = ("1 (nm)", "1 (Abs)", "2 (nm)", "2 (Abs)")
    for col_name in bad_col_names:
        assert col_name not in renamed.columns

    new_col_names = ("Abs 350", "Abs 600")
    for col_name in new_col_names:
        assert col_name in renamed.columns


def test_rename_abs_columns_by_wavelength():
    test_df = pd.DataFrame(
        data={
            "Sample ID": ["Peptide1_100_0.5", "Peptide2_100_0.5", "Peptide3_100_0.5"],
            "1 (nm)": [350, 350, 400],
            "1 (Abs)": [1, 2, 3],
            "2 (nm)": [400, 600, 600],
            "2 (Abs)": [4, 5, 6],
        }
    )
    
    renamed = pda.rename_abs_columns_by_wavelength(test_df)
    
    assert len(renamed) == len(test_df)    
    
    bad_col_names = ("1 (nm)", "1 (Abs)", "2 (nm)", "2 (Abs)")
    for col_name in bad_col_names:
        assert col_name not in renamed.columns
    
    new_col_names = ("Abs 350", "Abs 400", "Abs 600")
    for col_name in new_col_names:
        assert col_name in renamed.columns
    
    for value in [1, 2]:
        assert value in renamed["Abs 350"].values
    
    for value in [3, 4]:
        assert value in renamed["Abs 400"].values
        
    for value in [5, 6]:
        assert value in renamed["Abs 600"].values
    

def test_ParseKey():
    # TODO
    test_key = pda.ParseKey(("hi", str), separator=" ")
    test_key.separator = " "
    test_key.parse_key = ("hi", str)
    test_key.column_names = ("hi",)
    
    with pytest.raises(RuntimeError):
        bad_key = pda.ParseKey(10, separator=" ")
        
    with pytest.raises(RuntimeError):
        bad_key = pda.ParseKey(("10", "hi", str), separator="")  

    with pytest.raises(RuntimeError):
        bad_key = pda.ParseKey((str, "type"), separator=" ")
    
    with pytest.raises(RuntimeError):
        bad_key = pda.ParseKey(("str", "type"), separator=" ")
    
    
def test_make_columns_by_parse_key():
    test_data = make_test_data()
    df_new_cols = pda._make_columns_by_parse_key(test_data, ParseKey=pda.parse_rna_peptide)

    assert test_data.columns.isin(df_new_cols.columns).all()
    for key, _ in pda.parse_rna_peptide.parse_key:
        assert key in df_new_cols.columns

    with pytest.warns(UserWarning):
        bad_parse_key = pda.ParseKey(("This column fails", list), separator="_")
        pda._make_columns_by_parse_key(test_data, bad_parse_key)


def test_identify_buffer_measurements():
    test_df = make_test_data()
    assert (pda._identify_buffer_measurements(test_df) == [1]).all()

    test_df.loc[2, "Sample ID"] = "Buffer_100_0.5"
    assert (pda._identify_buffer_measurements(test_df) == [1, 2]).all()

    test_df.loc[:, "Sample ID"] = "Peptide_100_0.5"
    assert len(pda._identify_buffer_measurements(test_df)) == 0


def test_handle_incorrectly_named_samples_drop_all():
    test_df = make_test_data()
    dropped = pda._handle_incorrectly_named_samples(
        test_df, [0], drop_incorrectly_named_samples=True, drop_buffers=True
    )
    assert len(dropped) == 1
    assert len(pda._identify_buffer_measurements(dropped)) == 0


def test_handle_incorrectly_named_samples_drop_none():
    test_df = make_test_data()
    with pytest.warns(UserWarning):
        not_dropped = pda._handle_incorrectly_named_samples(
            test_df, [0], drop_incorrectly_named_samples=False, drop_buffers=False
        )
    assert (test_df == not_dropped).all().all()


def test_handle_incorrectly_named_samples_drop_buffers_only():
    test_df = make_test_data()
    with pytest.warns(UserWarning):
        dropped = pda._handle_incorrectly_named_samples(
            test_df, [0], drop_incorrectly_named_samples=False, drop_buffers=True
        )
    assert len(dropped) == 2
    assert len(pda._identify_buffer_measurements(dropped)) == 0


def test_handle_incorrectly_named_samples_keep_buffers_only():
    test_df = make_test_data()
    dropped = pda._handle_incorrectly_named_samples(
        test_df, [0], drop_incorrectly_named_samples=True, drop_buffers=False
    )
    assert len(dropped) == 2
    assert len(pda._identify_buffer_measurements(dropped)) == 1


def test_analyze_sample_names_buffer_dropping_only():
    test_df = make_test_data()
    dropped = pda.analyze_sample_names(test_df, drop_buffers=True)
    assert len(dropped) == 2
    assert len(pda._identify_buffer_measurements(dropped)) == 0

    not_dropped = pda.analyze_sample_names(test_df, drop_buffers=False)
    assert len(not_dropped) == 3
    assert len(pda._identify_buffer_measurements(not_dropped)) == 1
    assert len(not_dropped.loc[not_dropped["Sample ID"] == "blank", :]) == 1


def test_analyze_sample_names_drop_incorrectly_named_samples():
    test_df = make_test_data()
    test_df.loc[2, "Sample ID"] = "hello, world!"
    dropped_both = pda.analyze_sample_names(
        test_df, drop_incorrectly_named_samples=True, drop_buffers=True
    )
    assert len(dropped_both) == 1
    assert len(pda._identify_buffer_measurements(dropped_both)) == 0

    dropped_incorrect_not_buffers = pda.analyze_sample_names(
        test_df, drop_incorrectly_named_samples=True, drop_buffers=False
    )
    assert len(dropped_incorrect_not_buffers) == 2
    assert len(pda._identify_buffer_measurements(dropped_incorrect_not_buffers)) == 1

    with pytest.warns(UserWarning) as sample_name_divergence:
        pda.analyze_sample_names(test_df, drop_incorrectly_named_samples=False)
    assert (
        "Sample names do not adhere to requirements"
        in sample_name_divergence[0].message.args[0]
    )
    assert (
        "Identify incorrectly named samples by running analyze_sample_names"
        in sample_name_divergence[0].message.args[0]
    )


def test_analyze_sample_names_new_columns():
    analyzed = pda.analyze_sample_names(make_test_data())
    new_cols = ("Peptide", "Peptide concentration (uM)", "RNA/Peptide Ratio")
    for col_name in new_cols:
        assert col_name in analyzed.columns

    assert analyzed["Peptide concentration (uM)"].dtype == float
    assert analyzed["RNA/Peptide Ratio"].dtype == float


def test_analyze_sample_names_new_columns_kDNA_parse_key():
    analyzed = pda.analyze_sample_names(make_test_data(), ParseKey=pda.parse_kdna_mg2)
    new_cols = ("kDNA sample type", "DNA concentration (ng/uL)", "Mg2+ concentration")
    for col_name in new_cols:
        assert col_name in analyzed.columns

    assert analyzed["DNA concentration (ng/uL)"].dtype == float
    assert analyzed["Mg2+ concentration"].dtype == float


def test_analyze_sample_names_contents():
    analyzed = pda.analyze_sample_names(make_test_data())
    assert (analyzed["Peptide"] == "Peptide").all()
    assert (analyzed["Peptide concentration (uM)"] == 100.0).all()
    assert (analyzed["RNA/Peptide Ratio"] == 0.5).all()

    test_data = make_test_data()
    test_data.loc[1, "Sample ID"] = "DifferentPeptide_100_0.5"
    analyzed = pda.analyze_sample_names(test_data)
    assert "Peptide" in analyzed["Peptide"].values
    assert "DifferentPeptide" in analyzed["Peptide"].values


def test_break_out_date_and_time():
    test_df = make_test_data()
    test_df["Date and Time"] = "Date Time PM"
    broken_out = pda.break_out_date_and_time(test_df)
    new_cols = ["Date", "Time"]
    for name in new_cols:
        assert name in broken_out.columns
    assert "Date and Time" not in broken_out.columns
    assert (broken_out["Date"] == "Date").all()
    assert (broken_out["Time"] == "Time PM").all()


def test_drop_zeros():
    test_df = make_test_data()
    test_df.loc[1, "1 (nm)"] = 0
    assert 0 in test_df["1 (nm)"].values

    dropped = pda.drop_zeros(test_df, "1 (nm)")
    assert len(dropped) == 2
    assert 0 not in dropped["1 (nm)"].values


def test_drop_zeros_several_columns():
    test_df = make_test_data()
    test_df.loc[1, "1 (nm)"] = 0
    test_df.loc[2, "2 (nm)"] = 0
    assert 0 in test_df["1 (nm)"].values
    assert 0 in test_df["2 (nm)"].values

    dropped = pda.drop_zeros(test_df, ["1 (nm)", "2 (nm)"])
    assert len(dropped) == 1
    assert 0 not in dropped["1 (nm)"].values
    assert 0 not in dropped["2 (nm)"].values


def test_tidy_nanodrop_data(to_glob="test_data/*.tsv", **kwargs):
    """For testing a variety of input kwargs"""

    file_list = glob.glob(to_glob)
    df = pda.tidy_nanodrop_data(file_list, **kwargs)

    assert (
        df.columns == [
            "Sample ID",
            "Abs 350",
            "Abs 600",
            "Peptide",
            "Peptide concentration (uM)",
            "RNA/Peptide Ratio",
            "Date",
            "Time",
        ]
    ).all()

    float_cols = [
        "Abs 350",
        "Abs 600",
        "Peptide concentration (uM)",
        "RNA/Peptide Ratio",
    ]
    for col in float_cols:
        assert df[col].dtype == float

    return df


def test_tidy_nanodrop_data_defaults():
    df = test_tidy_nanodrop_data(to_glob="test_data/*.tsv")

    for peptide in df["Peptide"].values:
        assert "RG" in peptide


def test_tidy_nanodrop_data_read_csv():
    df = test_tidy_nanodrop_data(
        to_glob="test_data/*.csv",
        file_reader_kwargs={},
        drop_incorrectly_named_samples=True,
    )

    for peptide in df["Peptide"].values:
        assert "RG" in peptide


def test_tidy_nanodrop_data_raise_incorrect_samples_warning():
    with pytest.warns(UserWarning) as sample_name_divergence:
        df = test_tidy_nanodrop_data(
            to_glob="test_data/*.csv",
            file_reader_kwargs={},
            drop_incorrectly_named_samples=False,
        )
    assert (
        "Identify incorrectly named samples by running analyze_sample_names on your DataFrame."
        in sample_name_divergence[0].message.args[0]
    )

    # with drop_incorrectly_named_samples=False, not all Peptide values are filled
    assert "RG7" in df["Peptide"].values


def test_tidy_nanodrop_data_explicitly_keep_buffers():
    df = test_tidy_nanodrop_data(
        to_glob="test_data/*.csv",
        file_reader_kwargs={},
        drop_incorrectly_named_samples=True,
        drop_buffers=False,
    )

    assert "RG7" in df["Peptide"].values

    assert len(pda._identify_buffer_measurements(df)) > 0

    
def test_nanodrop_data_nonmatching_column_names():
    # bad_input contains a file without the "Sample ID" category
    # all samples from that file will be dropped, but an extra empty column makes it into the output
    file_list = glob.glob("test_data/bad_input/*.tsv")
    with pytest.warns(UserWarning):
        df = pda.tidy_nanodrop_data(file_list,
                                    drop_incorrectly_named_samples=True)
        
    float_cols = [
        "Abs 350",
        "Abs 600",
        "Peptide concentration (uM)",
        "RNA/Peptide Ratio",
    ]
    for col in float_cols:
        assert df[col].dtype == float
        
    for peptide in df["Peptide"].values:
        assert "RG" in peptide