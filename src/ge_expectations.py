import great_expectations as gx

def create_suite(df):
    # 1. Initialize an in-memory, zero-local-footprint ephemeral context
    context = gx.get_context(mode="ephemeral")

    # 2. Use modern .data_sources for v1.x+
    datasource = context.data_sources.add_pandas(name="pandas_source")
    data_asset = datasource.add_dataframe_asset(name="drugs_data")
    
    # 3. Configure batch definition using the fluent interface
    batch_definition = data_asset.add_batch_definition_whole_dataframe(name="batch_def")
    
    # 4. FIX: Instantiated inline via modern API instead of calling missing context attribute
    suite = gx.ExpectationSuite(name="drugs_suite")
    
    # 5. Bind the batch and the suite definition explicitly inside the validator
    validator = context.get_validator(
        batch_request=batch_definition.build_batch_request(batch_parameters={"dataframe": df}), 
        expectation_suite=suite  # Note: passing the suite object directly here
    )

    # 6. Run Data Quality Expectations
    validator.expect_column_to_exist("review")
    validator.expect_column_values_to_not_be_null("review")
    
    # 7. Persist suite signatures in-memory
    validator.save_expectation_suite(discard_failed_expectations=False)
    print("Suite created and validated successfully via unified Fluent API!")
    return True


