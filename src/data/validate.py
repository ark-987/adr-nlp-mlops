import great_expectations as gx


def validate_dataframe(df, stage="unknown"):

    # Convert pandas → GX dataset
    gx_df = gx.from_pandas(df)

    # Define expectations inline (programmatic)
    results = gx_df.validate(
        expectations=[
            gx.expectations.ExpectColumnToExist(column="review"),
            gx.expectations.ExpectColumnValuesToNotBeNull(column="review"),
            gx.expectations.ExpectColumnValuesToBeOfType(
                column="review",
                type_="str"
            ),
        ]
    )

    if not results["success"]:
        raise ValueError(f"❌ Validation failed at stage: {stage}")

    # -------------------------
    # CUSTOM BUSINESS RULES
    # -------------------------
    if len(df) < 100:
        raise ValueError(f"{stage}: Dataset too small")

    if df["review"].astype(str).str.len().mean() < 10:
        raise ValueError(f"{stage}: Text too short — likely bad data")

    print(f" Validation passed at stage: {stage}")

    return results