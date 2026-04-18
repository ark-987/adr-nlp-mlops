import great_expectations as gx
import pandas as pd


def create_suite():
    context = gx.get_context()

    # Load data
    df = pd.read_csv("data/raw/drugsComTrain_raw.csv")

    # SAFE datasource handling
    datasource = None

    for ds in context.sources.values():
        if ds.name == "pandas_source":
            datasource = ds
            break

    if datasource is None:
        datasource = context.sources.add_pandas("pandas_source")

    # Create asset
    data_asset = datasource.add_dataframe_asset(name="drugs_data")
    batch_request = data_asset.build_batch_request(dataframe=df)

    validator = context.get_validator(batch_request=batch_request)

    # Expectations
    validator.expect_column_to_exist("review")
    validator.expect_column_values_to_not_be_null("review")
    validator.expect_column_values_to_be_of_type("review", "str")

    validator.save_expectation_suite(discard_failed_expectations=False)

    print("Suite created successfully")