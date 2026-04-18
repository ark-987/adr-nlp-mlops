import great_expectations as gx

def validate_dataframe(df, stage="unknown"):
    # ✅ FORCE persistent context
    context = gx.get_context(context_root_dir="great_expectations")

    # ✅ ALWAYS use add_or_update (no duplication errors)
    datasource = context.sources.add_or_update_pandas(name="pandas_source")

    data_asset = datasource.add_dataframe_asset(name=f"{stage}_asset")
    batch_request = data_asset.build_batch_request(dataframe=df)

    validator = context.get_validator(batch_request=batch_request)

    # Expectations
    validator.expect_column_to_exist("review")
    validator.expect_column_values_to_not_be_null("review")
    validator.expect_column_values_to_be_of_type("review", "str")

    results = validator.validate()

    if not results.success:
        raise ValueError(f"❌ GE validation failed at stage: {stage}")

    print(f"✔ GE validation passed at stage: {stage}")

    return results
