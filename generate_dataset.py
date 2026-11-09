# ======================================================================
# Customer Complaint Decision-Support Tool
# Synthetic dataset generation pipeline
# Adapted from the Colab notebook (Untitled0.ipynb) for local/offline
# execution. Logic is UNCHANGED from the original notebook; only the
# Colab file-upload/download calls were replaced with plain filesystem
# I/O so the script can run outside Google Colab.
# ======================================================================

import sys
import pandas as pd
import numpy as np
import sqlite3

# Jupyter's `display()` is not defined in a plain script; fall back to print.
def display(x):
    print(x)

# Reproducible random generator (same seed as the original notebook)
rng = np.random.default_rng(42)

# --------------------------------------------------
# Load CFPB data (local file, in place of Colab's files.upload())
# --------------------------------------------------
filename = sys.argv[1] if len(sys.argv) > 1 else "complaints.csv"

df = pd.read_csv(
    filename,
    dtype={"Complaint ID": "string"},
    low_memory=False
)

print("File loaded:", filename)
print("\nDataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumn names:")
print(df.columns.tolist())

# --------------------------------------------------
# STEP 2 — Clean and prepare CFPB source data
# --------------------------------------------------

# Required CFPB columns
required_map = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Issue": "issue"
}

# Optional CFPB columns
optional_map = {
    "Sub-product": "sub_product",
    "Sub-issue": "sub_issue",
    "Submitted via": "submission_channel",
    "Consumer complaint narrative": "complaint_narrative"
}


# --------------------------------------------------
# Check required columns
# --------------------------------------------------

missing_required = [
    col for col in required_map
    if col not in df.columns
]

if missing_required:
    raise ValueError(
        f"Missing REQUIRED CFPB columns: {missing_required}"
    )


# --------------------------------------------------
# Keep available columns
# --------------------------------------------------

available_columns = list(required_map.keys())

for col in optional_map:
    if col in df.columns:
        available_columns.append(col)

df_base = df[available_columns].copy()


# --------------------------------------------------
# Rename columns
# --------------------------------------------------

rename_map = {
    **required_map,
    **optional_map
}

df_base = df_base.rename(columns=rename_map)


# --------------------------------------------------
# Create missing optional fields as NULL
# --------------------------------------------------

for new_name in optional_map.values():

    if new_name not in df_base.columns:
        df_base[new_name] = pd.NA


# --------------------------------------------------
# Complaint ID
# --------------------------------------------------

df_base["complaint_id"] = (
    df_base["complaint_id"]
    .astype("string")
    .str.strip()
)


# --------------------------------------------------
# Date
# --------------------------------------------------

df_base["date_received"] = pd.to_datetime(
    df_base["date_received"],
    errors="coerce"
)


# --------------------------------------------------
# Clean categorical text
# --------------------------------------------------

text_columns = [
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "submission_channel"
]

for col in text_columns:

    df_base[col] = (
        df_base[col]
        .astype("string")
        .str.strip()
    )


# --------------------------------------------------
# Audit
# --------------------------------------------------

print("Clean base dataframe created successfully!")

print("\nRows:")
print(len(df_base))

print("\nFinal columns:")
print(df_base.columns.tolist())

print("\nMissing values:")
print(df_base.isna().sum())

print("\nDuplicate Complaint IDs:")
print(
    df_base["complaint_id"]
    .duplicated()
    .sum()
)

print("\nInvalid/missing dates:")
print(
    df_base["date_received"]
    .isna()
    .sum()
)# --------------------------------------------------
# STEP 3 — Fix timezone and filter project period
# --------------------------------------------------

# Remove timezone information because CFPB date_received
# is being used as a calendar date in this project
df_base["date_received"] = (
    pd.to_datetime(
        df_base["date_received"],
        errors="coerce",
        utc=True
    )
    .dt.tz_localize(None)
)


# --------------------------------------------------
# Project time window
# --------------------------------------------------

start_date = pd.Timestamp("2023-01-01")
end_date = pd.Timestamp("2026-09-23")


# Keep complaints inside project period
df_project = df_base[
    (df_base["date_received"] >= start_date)
    &
    (df_base["date_received"] <= end_date)
].copy()


# --------------------------------------------------
# Create temporary time fields
# --------------------------------------------------

df_project["year"] = (
    df_project["date_received"].dt.year
)

df_project["year_month"] = (
    df_project["date_received"].dt.to_period("M")
)


# --------------------------------------------------
# Chronological split
# --------------------------------------------------

def assign_split(date):

    if date.year in [2023, 2024]:
        return "train"

    elif date.year == 2025:
        return "development"

    elif date.year == 2026:
        return "test"

    return None


df_project["dataset_split"] = (
    df_project["date_received"]
    .apply(assign_split)
)


# --------------------------------------------------
# Potential right censoring
# --------------------------------------------------

observation_limit = (
    end_date - pd.Timedelta(days=30)
)

df_project["potentially_censored"] = (
    df_project["date_received"]
    > observation_limit
)


# --------------------------------------------------
# Validation / audit
# --------------------------------------------------

print("===== PROJECT TIME AUDIT =====")

print("\nRows after filtering:")
print(len(df_project))

print("\nEarliest complaint:")
print(df_project["date_received"].min())

print("\nLatest complaint:")
print(df_project["date_received"].max())

print("\nComplaints by year:")
print(
    df_project["year"]
    .value_counts()
    .sort_index()
)

print("\nDataset split:")
print(
    df_project["dataset_split"]
    .value_counts()
)

print("\nPotentially censored complaints:")
print(
    df_project["potentially_censored"].sum()
)

outside = (
    (df_project["date_received"] < start_date)
    |
    (df_project["date_received"] > end_date)
).sum()

print("\nRows outside project period:")
print(outside)# --------------------------------------------------
# STEP 4 — Create departments table
# --------------------------------------------------

departments = pd.DataFrame({
    "department_id": [
        "D01",
        "D02",
        "D03",
        "D04",
        "D05",
        "D06"
    ],
    "department_name": [
        "General Customer Service",
        "Credit Card",
        "Mortgage",
        "Loans",
        "Fraud / Risk",
        "Escalations"
    ]
})


# --------------------------------------------------
# Validate departments
# --------------------------------------------------

assert departments["department_id"].is_unique
assert departments["department_name"].is_unique
assert departments.isna().sum().sum() == 0


print("===== DEPARTMENTS TABLE =====")
display(departments)

print("\nNumber of departments:")
print(len(departments))


# --------------------------------------------------
# Inspect the actual CFPB product categories
# --------------------------------------------------

print("\n===== CFPB PRODUCTS =====")

product_counts = (
    df_project["product"]
    .value_counts(dropna=False)
)

print(product_counts)


# --------------------------------------------------
# Inspect common issues
# --------------------------------------------------

print("\n===== TOP 30 CFPB ISSUES =====")

issue_counts = (
    df_project["issue"]
    .value_counts(dropna=False)
)

print(issue_counts.head(30))# --------------------------------------------------
# STEP 5 — Create initial routing rules
# --------------------------------------------------

def assign_reference_department(product, issue):

    product_text = str(product).lower()
    issue_text = str(issue).lower()

    # ------------------------------------------
    # 1. Fraud / Risk overrides product routing
    # ------------------------------------------

    fraud_keywords = [
        "fraud",
        "identity theft",
        "unauthorized",
        "stolen",
        "scam"
    ]

    if any(word in issue_text for word in fraud_keywords):
        return "D05"


    # ------------------------------------------
    # 2. Credit-card-related complaints
    # ------------------------------------------

    if (
        "credit card" in product_text
        or "prepaid card" in product_text
    ):
        return "D02"


    # ------------------------------------------
    # 3. Mortgage complaints
    # ------------------------------------------

    if "mortgage" in product_text:
        return "D03"


    # ------------------------------------------
    # 4. Loan complaints
    # ------------------------------------------

    loan_keywords = [
        "loan",
        "student loan",
        "vehicle loan",
        "payday"
    ]

    if any(word in product_text for word in loan_keywords):
        return "D04"


    # ------------------------------------------
    # 5. Everything else
    # ------------------------------------------

    return "D01"


# --------------------------------------------------
# Apply routing rules
# --------------------------------------------------

df_project["reference_department_id"] = (
    df_project.apply(
        lambda row: assign_reference_department(
            row["product"],
            row["issue"]
        ),
        axis=1
    )
)


# --------------------------------------------------
# Validate IDs
# --------------------------------------------------

valid_department_ids = set(
    departments["department_id"]
)

assert df_project[
    "reference_department_id"
].isin(
    valid_department_ids
).all()


# --------------------------------------------------
# Inspect routing distribution
# --------------------------------------------------

print("===== REFERENCE ROUTING DISTRIBUTION =====")

routing_counts = (
    df_project[
        "reference_department_id"
    ]
    .value_counts()
    .sort_index()
)

print(routing_counts)


print("\n===== ROUTING PERCENTAGES =====")

routing_percentages = (
    df_project[
        "reference_department_id"
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
    .sort_index()
)

print(routing_percentages)


# --------------------------------------------------
# Inspect sample complaints
# --------------------------------------------------

print("\n===== SAMPLE ROUTING RESULTS =====")

display(
    df_project[
        [
            "complaint_id",
            "product",
            "issue",
            "reference_department_id"
        ]
    ].sample(
        min(20, len(df_project)),
        random_state=42
    )
)# --------------------------------------------------
# STEP 6 — Generate synthetic customers
# --------------------------------------------------

# Sort complaints first
df_project = (
    df_project
    .sort_values(
        ["date_received", "complaint_id"]
    )
    .reset_index(drop=True)
)

n_complaints = len(df_project)

print("Complaints to assign:", n_complaints)


# --------------------------------------------------
# Customer complaint-count distribution
# --------------------------------------------------

# Approximate design:
# 80% of customers = 1 complaint
# 13% = 2 complaints
# 5% = 3 complaints
# 2% = 4 or 5 complaints

count_options = [1, 2, 3, 4, 5]

count_probabilities = [
    0.80,
    0.13,
    0.05,
    0.015,
    0.005
]


# --------------------------------------------------
# Generate enough customer complaint counts
# --------------------------------------------------

customer_sizes = []

while sum(customer_sizes) < n_complaints:

    size = rng.choice(
        count_options,
        p=count_probabilities
    )

    customer_sizes.append(size)


# --------------------------------------------------
# Create customer IDs
# --------------------------------------------------

customer_assignment = []

for i, size in enumerate(
    customer_sizes,
    start=1
):

    customer_id = f"C{i:06d}"

    customer_assignment.extend(
        [customer_id] * size
    )


# Remove extra assignments
customer_assignment = (
    customer_assignment[:n_complaints]
)

# Spread repeat customers across the timeline
rng.shuffle(customer_assignment)

df_project["customer_id"] = (
    customer_assignment
)


# --------------------------------------------------
# Create customers table
# --------------------------------------------------

customer_ids = sorted(
    df_project["customer_id"].unique()
)

customers = pd.DataFrame({
    "customer_id": customer_ids
})


# 90% Individual / 10% Business
customers["customer_type"] = rng.choice(
    ["Individual", "Business"],
    size=len(customers),
    p=[0.90, 0.10]
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert customers["customer_id"].is_unique

assert df_project[
    "customer_id"
].notna().all()

assert set(
    df_project["customer_id"]
).issubset(
    set(customers["customer_id"])
)

assert customers[
    "customer_type"
].isin(
    ["Individual", "Business"]
).all()


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n===== CUSTOMERS TABLE =====")

print("Number of synthetic customers:")
print(len(customers))

print("\nCustomer type distribution:")
print(
    customers[
        "customer_type"
    ]
    .value_counts()
)

print("\nCustomer type percentages:")
print(
    customers[
        "customer_type"
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


print("\n===== COMPLAINTS PER CUSTOMER =====")

complaints_per_customer = (
    df_project
    .groupby("customer_id")
    .size()
)

print(
    complaints_per_customer
    .value_counts()
    .sort_index()
)


print("\nSample customers:")
display(
    customers.head(10)
)# --------------------------------------------------
# STEP 7 — Generate complaint timestamps
# and build complaints table
# CORRECTED VERSION
# --------------------------------------------------

# Make sure date_received contains DATE only
df_project["date_received"] = (
    pd.to_datetime(
        df_project["date_received"],
        errors="coerce"
    )
    .dt.normalize()
)


# --------------------------------------------------
# Higher probability during daytime,
# but complaints can still arrive 24/7
# --------------------------------------------------

hour_weights = [
    1, 1, 1, 1, 1, 1,      # 12 AM - 5 AM
    3, 4, 5, 6, 7, 8,      # 6 AM - 11 AM
    9, 9, 9, 9, 8, 8,      # 12 PM - 5 PM
    7, 6, 5, 4, 3, 2       # 6 PM - 11 PM
]

hour_probs = np.array(
    hour_weights,
    dtype=float
)

hour_probs = (
    hour_probs /
    hour_probs.sum()
)


# --------------------------------------------------
# Generate hour, minute, second
# --------------------------------------------------

n = len(df_project)

hours = rng.choice(
    np.arange(24),
    size=n,
    p=hour_probs
)

minutes = rng.integers(
    0,
    60,
    size=n
)

seconds = rng.integers(
    0,
    60,
    size=n
)


# --------------------------------------------------
# Create exact complaint timestamp
# --------------------------------------------------

df_project["complaint_timestamp"] = (
    df_project["date_received"]
    + pd.to_timedelta(hours, unit="h")
    + pd.to_timedelta(minutes, unit="m")
    + pd.to_timedelta(seconds, unit="s")
)


# --------------------------------------------------
# Sort by exact complaint time
# --------------------------------------------------

df_project = (
    df_project
    .sort_values(
        [
            "complaint_timestamp",
            "complaint_id"
        ]
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# Build physical complaints table
# --------------------------------------------------

complaints = df_project[
    [
        "complaint_id",
        "customer_id",
        "date_received",
        "complaint_timestamp",
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "submission_channel",
        "complaint_narrative"
    ]
].copy()


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert complaints[
    "complaint_id"
].notna().all()

assert complaints[
    "complaint_id"
].is_unique

assert complaints[
    "customer_id"
].notna().all()

assert set(
    complaints["customer_id"]
).issubset(
    set(customers["customer_id"])
)


# --------------------------------------------------
# Check date/timestamp consistency
# --------------------------------------------------

date_mismatches = (
    complaints["complaint_timestamp"]
    .dt.normalize()
    !=
    complaints["date_received"]
    .dt.normalize()
).sum()


print("===== COMPLAINTS TABLE CREATED =====")

print("\nRows:")
print(len(complaints))

print("\nColumns:")
print(complaints.columns.tolist())

print("\nDuplicate complaint IDs:")
print(
    complaints[
        "complaint_id"
    ].duplicated().sum()
)

print("\nDate/timestamp mismatches:")
print(date_mismatches)

assert date_mismatches == 0


print("\nSample complaints:")
display(
    complaints.head(10)
)# --------------------------------------------------
# STEP 8 — Generate prediction_time
# and prior_complaint_count
# --------------------------------------------------

complaints_work = complaints.copy()


# --------------------------------------------------
# Generate prediction delay
# 30 seconds to 5 minutes
# --------------------------------------------------

prediction_delay_seconds = rng.integers(
    30,
    301,
    size=len(complaints_work)
)

complaints_work["prediction_time"] = (
    complaints_work["complaint_timestamp"]
    +
    pd.to_timedelta(
        prediction_delay_seconds,
        unit="s"
    )
)


# --------------------------------------------------
# Validate prediction occurs after complaint intake
# --------------------------------------------------

invalid_prediction_times = (
    complaints_work["prediction_time"]
    <=
    complaints_work["complaint_timestamp"]
).sum()

print(
    "Predictions before/equal to complaint time:",
    invalid_prediction_times
)

assert invalid_prediction_times == 0


# --------------------------------------------------
# Calculate prior complaint count
# --------------------------------------------------
# Definition:
# Number of earlier complaints from the SAME
# synthetic customer before the current complaint
# was received.
# --------------------------------------------------

complaints_work = (
    complaints_work
    .sort_values(
        [
            "customer_id",
            "complaint_timestamp",
            "complaint_id"
        ]
    )
    .reset_index(drop=True)
)

complaints_work["prior_complaint_count"] = (
    complaints_work
    .groupby("customer_id")
    .cumcount()
)


# --------------------------------------------------
# Restore global chronological order
# --------------------------------------------------

complaints_work = (
    complaints_work
    .sort_values(
        [
            "complaint_timestamp",
            "complaint_id"
        ]
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

negative_counts = (
    complaints_work[
        "prior_complaint_count"
    ] < 0
).sum()

assert negative_counts == 0


print("\n===== STEP 8 RESULTS =====")

print("\nPrediction delay minimum:")
print(
    (
        complaints_work["prediction_time"]
        -
        complaints_work["complaint_timestamp"]
    ).min()
)

print("\nPrediction delay maximum:")
print(
    (
        complaints_work["prediction_time"]
        -
        complaints_work["complaint_timestamp"]
    ).max()
)

print("\nPrior complaint count distribution:")
print(
    complaints_work[
        "prior_complaint_count"
    ]
    .value_counts()
    .sort_index()
)


print("\nSample:")
display(
    complaints_work[
        [
            "complaint_id",
            "customer_id",
            "complaint_timestamp",
            "prediction_time",
            "prior_complaint_count"
        ]
    ].head(15)
)# --------------------------------------------------
# STEP 9 — Generate department capacity timeline
# --------------------------------------------------

# Baseline staffing by department
baseline_staff = {
    "D01": 30,   # General Customer Service
    "D02": 20,   # Credit Card
    "D03": 16,   # Mortgage
    "D04": 18,   # Loans
    "D05": 12,   # Fraud / Risk
    "D06": 8     # Escalations
}

# Each available staff member can handle roughly
# 8 active cases at a time
cases_per_staff = 8


# --------------------------------------------------
# Capacity timeline
# --------------------------------------------------

capacity_start = (
    complaints["complaint_timestamp"]
    .min()
    .floor("4h")
)

capacity_end = (
    complaints["complaint_timestamp"]
    .max()
    .ceil("4h")
)

capacity_times = pd.date_range(
    start=capacity_start,
    end=capacity_end,
    freq="4h"
)

print("Capacity start:", capacity_start)
print("Capacity end:", capacity_end)
print("Number of capacity timestamps:", len(capacity_times))


# --------------------------------------------------
# Prepare complaint arrivals by department
# --------------------------------------------------

# Bring the reference department created earlier
# together with complaint timestamps
routing_lookup = df_project[
    [
        "complaint_id",
        "reference_department_id"
    ]
].copy()

arrival_data = complaints[
    [
        "complaint_id",
        "complaint_timestamp"
    ]
].merge(
    routing_lookup,
    on="complaint_id",
    how="left"
)

assert arrival_data[
    "reference_department_id"
].notna().all()


# --------------------------------------------------
# Generate capacity snapshots
# --------------------------------------------------

capacity_rows = []

capacity_counter = 1

# Pre-sort arrival timestamps once so each 48-hour workload
# count can be calculated with binary search. Filtering the full
# complaint dataframe inside every capacity snapshot is prohibitively
# slow for multi-year datasets.
all_arrival_times = np.sort(
    arrival_data["complaint_timestamp"].to_numpy(dtype="datetime64[ns]")
)

arrival_times_by_department = {
    department_id: np.sort(
        arrival_data.loc[
            arrival_data["reference_department_id"] == department_id,
            "complaint_timestamp"
        ].to_numpy(dtype="datetime64[ns]")
    )
    for department_id in baseline_staff
}

for timestamp in capacity_times:

    hour = timestamp.hour
    weekday = timestamp.weekday()

    # Monday = 0 ... Sunday = 6
    is_weekend = weekday >= 5

    for department_id, base_staff in baseline_staff.items():

        # ------------------------------------------
        # Staffing multiplier by time of day
        # ------------------------------------------

        if 8 <= hour < 20:
            time_multiplier = 1.00

        elif 20 <= hour < 24:
            time_multiplier = 0.65

        else:
            time_multiplier = 0.45

        # Weekend staffing is lower
        weekend_multiplier = (
            0.65 if is_weekend else 1.00
        )

        scheduled_staff = round(
            base_staff
            * time_multiplier
            * weekend_multiplier
        )

        # Small random employee unavailability
        unavailable_staff = rng.binomial(
            n=max(scheduled_staff, 0),
            p=0.05
        )

        available_staff = max(
            0,
            scheduled_staff - unavailable_staff
        )


        # ------------------------------------------
        # Recent complaint volume
        # Previous 48 hours for this department
        # ------------------------------------------

        window_start = (
            timestamp
            - pd.Timedelta(hours=48)
        )

        department_arrival_times = arrival_times_by_department[
            department_id
        ]
        window_start64 = np.datetime64(window_start, "ns")
        timestamp64 = np.datetime64(timestamp, "ns")
        recent_arrivals = int(
            np.searchsorted(
                department_arrival_times, timestamp64, side="right"
            )
            - np.searchsorted(
                department_arrival_times, window_start64, side="right"
            )
        )


        # D06 is mainly escalation-based,
        # so give it some workload even though
        # complaints are not initially routed there.
        if department_id == "D06":

            arrivals_in_window = int(
                np.searchsorted(
                    all_arrival_times, timestamp64, side="right"
                )
                - np.searchsorted(
                    all_arrival_times, window_start64, side="right"
                )
            )
            recent_arrivals = max(
                recent_arrivals,
                int(0.12 * arrivals_in_window)
            )


        # ------------------------------------------
        # Synthetic current open cases
        # ------------------------------------------

        # Base backlog around 55%–80% of capacity
        if available_staff > 0:

            total_case_capacity = (
                available_staff
                * cases_per_staff
            )

            utilization = rng.uniform(
                0.55,
                0.80
            )

            base_open_cases = round(
                total_case_capacity
                * utilization
            )

            # Recent complaint arrivals add workload
            current_open_cases = (
                base_open_cases
                +
                recent_arrivals
            )

            # Sometimes departments are overloaded
            if rng.random() < 0.05:

                overload_extra = rng.integers(
                    1,
                    max(
                        2,
                        int(
                            total_case_capacity
                            * 0.20
                        )
                    )
                )

                current_open_cases += (
                    overload_extra
                )

        else:

            current_open_cases = (
                recent_arrivals
            )


        # ------------------------------------------
        # Available capacity
        # ------------------------------------------

        total_case_capacity = (
            available_staff
            * cases_per_staff
        )

        available_capacity = max(
            0,
            total_case_capacity
            -
            current_open_cases
        )


        # ------------------------------------------
        # Add row
        # ------------------------------------------

        capacity_rows.append({
            "capacity_id":
                f"CAP{capacity_counter:07d}",

            "department_id":
                department_id,

            "capacity_timestamp":
                timestamp,

            "available_staff":
                int(available_staff),

            "current_open_cases":
                int(current_open_cases),

            "available_capacity":
                int(available_capacity)
        })

        capacity_counter += 1


# --------------------------------------------------
# Create capacity dataframe
# --------------------------------------------------

capacity = pd.DataFrame(
    capacity_rows
)


# --------------------------------------------------
# Validate
# --------------------------------------------------

assert capacity[
    "capacity_id"
].is_unique

assert capacity[
    "department_id"
].isin(
    departments[
        "department_id"
    ]
).all()

assert (
    capacity[
        "available_staff"
    ] >= 0
).all()

assert (
    capacity[
        "current_open_cases"
    ] >= 0
).all()

assert (
    capacity[
        "available_capacity"
    ] >= 0
).all()

duplicate_snapshots = (
    capacity.duplicated(
        subset=[
            "department_id",
            "capacity_timestamp"
        ]
    ).sum()
)

assert duplicate_snapshots == 0


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n===== CAPACITY TABLE CREATED =====")

print("\nRows:")
print(len(capacity))

print("\nDuplicate department/time snapshots:")
print(duplicate_snapshots)

print("\nCapacity rows by department:")
print(
    capacity[
        "department_id"
    ].value_counts()
    .sort_index()
)

print("\nSample:")
display(
    capacity.head(12)
)# --------------------------------------------------
# STEP 10 — Create decision_context table
# CORRECTED VERSION
# --------------------------------------------------

# Bring reference department into complaints_work
context_base = complaints_work.merge(
    df_project[
        [
            "complaint_id",
            "reference_department_id"
        ]
    ],
    on="complaint_id",
    how="left"
)

context_base = context_base.rename(
    columns={
        "reference_department_id":
        "context_department_id"
    }
)

assert context_base[
    "context_department_id"
].notna().all()


# --------------------------------------------------
# Match capacity department-by-department
# --------------------------------------------------

matched_groups = []

for department_id in departments["department_id"]:

    # Complaints belonging to this context department
    left = (
        context_base[
            context_base[
                "context_department_id"
            ] == department_id
        ]
        .copy()
        .sort_values("prediction_time")
    )

    # Capacity snapshots for this department
    right = (
        capacity[
            capacity[
                "department_id"
            ] == department_id
        ]
        .copy()
        .sort_values("capacity_timestamp")
    )

    # Match latest snapshot at or BEFORE prediction
    matched = pd.merge_asof(
        left,
        right,
        left_on="prediction_time",
        right_on="capacity_timestamp",
        direction="backward"
    )

    matched_groups.append(matched)


# Combine all departments
context_with_capacity = pd.concat(
    matched_groups,
    ignore_index=True
)


# --------------------------------------------------
# Validate capacity matching
# --------------------------------------------------

missing_capacity = (
    context_with_capacity[
        "capacity_id"
    ].isna().sum()
)

print(
    "Complaints missing capacity snapshot:",
    missing_capacity
)

assert missing_capacity == 0


future_capacity = (
    context_with_capacity[
        "capacity_timestamp"
    ]
    >
    context_with_capacity[
        "prediction_time"
    ]
).sum()

print(
    "Future capacity snapshots used:",
    future_capacity
)

assert future_capacity == 0


# --------------------------------------------------
# Calculate workload
# current_open_cases / available_staff
# --------------------------------------------------

safe_staff = (
    context_with_capacity[
        "available_staff"
    ]
    .clip(lower=1)
)

context_with_capacity[
    "current_workload"
] = (
    context_with_capacity[
        "current_open_cases"
    ]
    /
    safe_staff
).round(2)


# --------------------------------------------------
# Generate synthetic known financial impact
# Around 30% unknown
# --------------------------------------------------

n_context = len(
    context_with_capacity
)

financial_known = (
    rng.random(n_context)
    >= 0.30
)

financial_values = rng.lognormal(
    mean=np.log(500),
    sigma=1.20,
    size=n_context
)

financial_values = np.clip(
    financial_values,
    0,
    50000
)

financial_values = np.round(
    financial_values,
    2
)

context_with_capacity[
    "known_financial_impact"
] = financial_values

context_with_capacity.loc[
    ~financial_known,
    "known_financial_impact"
] = np.nan


# --------------------------------------------------
# Create decision context IDs
# --------------------------------------------------

context_with_capacity = (
    context_with_capacity
    .sort_values(
        [
            "prediction_time",
            "complaint_id"
        ]
    )
    .reset_index(drop=True)
)

context_with_capacity[
    "decision_context_id"
] = [
    f"DC{i:07d}"
    for i in range(
        1,
        len(context_with_capacity) + 1
    )
]


# --------------------------------------------------
# Build physical decision_context table
# --------------------------------------------------

decision_context = (
    context_with_capacity[
        [
            "decision_context_id",
            "complaint_id",
            "context_department_id",
            "prediction_time",
            "prior_complaint_count",
            "known_financial_impact",
            "current_workload",
            "available_capacity"
        ]
    ]
    .copy()
)


# --------------------------------------------------
# Final validation
# --------------------------------------------------

assert decision_context[
    "decision_context_id"
].is_unique

assert decision_context[
    "complaint_id"
].is_unique

assert decision_context[
    "complaint_id"
].notna().all()

assert decision_context[
    "prediction_time"
].notna().all()

assert (
    decision_context[
        "prior_complaint_count"
    ] >= 0
).all()

assert (
    decision_context[
        "current_workload"
    ] >= 0
).all()

assert (
    decision_context[
        "available_capacity"
    ] >= 0
).all()


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n===== DECISION CONTEXT TABLE CREATED =====")

print("\nRows:")
print(len(decision_context))

print("\nUnique complaints:")
print(
    decision_context[
        "complaint_id"
    ].nunique()
)

print("\nMissing financial impact:")
print(
    decision_context[
        "known_financial_impact"
    ].isna().sum()
)

print("\nMissing financial impact percentage:")
print(
    round(
        decision_context[
            "known_financial_impact"
        ]
        .isna()
        .mean()
        * 100,
        2
    ),
    "%"
)

print("\nCurrent workload summary:")
print(
    decision_context[
        "current_workload"
    ].describe()
)

print("\nSample:")
display(
    decision_context.head(10)
)# --------------------------------------------------
# STEP 11 — Generate hidden oracle ground truth
# --------------------------------------------------

# Combine complaint information with decision context
oracle_base = (
    complaints[
        [
            "complaint_id",
            "customer_id",
            "product",
            "sub_product",
            "issue",
            "sub_issue"
        ]
    ]
    .merge(
        decision_context[
            [
                "complaint_id",
                "prior_complaint_count",
                "known_financial_impact",
                "current_workload"
            ]
        ],
        on="complaint_id",
        how="left"
    )
    .merge(
        df_project[
            [
                "complaint_id",
                "reference_department_id"
            ]
        ],
        on="complaint_id",
        how="left"
    )
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert len(oracle_base) == len(complaints)

assert oracle_base[
    "complaint_id"
].is_unique

assert oracle_base[
    "reference_department_id"
].notna().all()


# --------------------------------------------------
# Customer complaint propensity
# Hidden customer-level synthetic variable
# --------------------------------------------------

unique_customers = (
    oracle_base["customer_id"]
    .drop_duplicates()
)

customer_propensity = pd.DataFrame({
    "customer_id": unique_customers
})

# Most customers low/moderate propensity,
# smaller group has higher repeat tendency
customer_propensity[
    "customer_complaint_propensity"
] = rng.beta(
    a=2,
    b=6,
    size=len(customer_propensity)
)

oracle_base = oracle_base.merge(
    customer_propensity,
    on="customer_id",
    how="left"
)


# --------------------------------------------------
# Generate hidden complaint complexity
# --------------------------------------------------

# Beta distribution keeps most cases moderate,
# with fewer very complex cases
oracle_base[
    "complaint_complexity"
] = rng.beta(
    a=2.5,
    b=2.5,
    size=len(oracle_base)
)


# --------------------------------------------------
# Product/issue risk signal
# based on actual CFPB text categories
# --------------------------------------------------

def calculate_issue_risk(row):

    combined = (
        str(row["product"])
        + " "
        + str(row["issue"])
        + " "
        + str(row["sub_issue"])
    ).lower()

    risk = 0.20

    # Higher-risk signals
    high_risk_keywords = [
        "fraud",
        "identity theft",
        "unauthorized",
        "scam",
        "stolen",
        "foreclosure",
        "repossession",
        "threat",
        "legal",
        "debt collection"
    ]

    moderate_risk_keywords = [
        "payment",
        "credit reporting",
        "incorrect",
        "dispute",
        "fee",
        "interest",
        "closing",
        "transfer"
    ]

    if any(
        keyword in combined
        for keyword in high_risk_keywords
    ):
        risk += 0.35

    elif any(
        keyword in combined
        for keyword in moderate_risk_keywords
    ):
        risk += 0.15

    return min(risk, 1.0)


oracle_base[
    "issue_risk"
] = oracle_base.apply(
    calculate_issue_risk,
    axis=1
)


# --------------------------------------------------
# Prior complaint effect
# --------------------------------------------------

oracle_base[
    "prior_history_effect"
] = np.clip(
    oracle_base[
        "prior_complaint_count"
    ] / 4,
    0,
    1
)


# --------------------------------------------------
# Financial impact effect
# --------------------------------------------------

financial_filled = (
    oracle_base[
        "known_financial_impact"
    ]
    .fillna(0)
)

# Log transformation avoids huge dollar amounts
# dominating the score
oracle_base[
    "financial_impact_effect"
] = np.clip(
    np.log1p(financial_filled)
    /
    np.log1p(50000),
    0,
    1
)


# --------------------------------------------------
# Random unobserved variation
# --------------------------------------------------

risk_noise = rng.normal(
    loc=0,
    scale=0.08,
    size=len(oracle_base)
)


# --------------------------------------------------
# Hidden underlying risk score
# --------------------------------------------------

oracle_base[
    "underlying_risk_score"
] = (
    0.35
    * oracle_base["issue_risk"]

    + 0.10
    * oracle_base["prior_history_effect"]

    + 0.15
    * oracle_base["financial_impact_effect"]

    + 0.20
    * oracle_base["complaint_complexity"]

    + 0.10
    * oracle_base[
        "customer_complaint_propensity"
    ]

    + 0.10
    * (
        oracle_base["current_workload"]
        / 10
    ).clip(0, 1)

    + risk_noise
)

oracle_base[
    "underlying_risk_score"
] = np.clip(
    oracle_base[
        "underlying_risk_score"
    ],
    0,
    1
)


# --------------------------------------------------
# Generate TRUE urgency
# --------------------------------------------------

def assign_true_urgency(score):

    if score < 0.40:
        return "Low"

    elif score < 0.70:
        return "Medium"

    else:
        return "High"


oracle_base[
    "true_urgency"
] = (
    oracle_base[
        "underlying_risk_score"
    ]
    .apply(assign_true_urgency)
)


# --------------------------------------------------
# Baseline escalation probability
# --------------------------------------------------
# This is hidden baseline risk at intake.
#
# We will later create the FINAL actual escalation
# after manager decisions, routing and workflow
# information exist.
# --------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


escalation_logit = (
    -3.0

    + 4.0
    * oracle_base[
        "underlying_risk_score"
    ]

    + 0.50
    * oracle_base[
        "complaint_complexity"
    ]

    + 0.25
    * oracle_base[
        "prior_history_effect"
    ]
)

oracle_base[
    "baseline_escalation_probability"
] = sigmoid(
    escalation_logit
)


# --------------------------------------------------
# Create hidden oracle table
# --------------------------------------------------

oracle_ground_truth = oracle_base[
    [
        "complaint_id",
        "reference_department_id",
        "customer_complaint_propensity",
        "complaint_complexity",
        "issue_risk",
        "prior_history_effect",
        "financial_impact_effect",
        "underlying_risk_score",
        "true_urgency",
        "baseline_escalation_probability"
    ]
].copy()


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert oracle_ground_truth[
    "complaint_id"
].is_unique

assert (
    oracle_ground_truth[
        "underlying_risk_score"
    ].between(0, 1)
).all()

assert (
    oracle_ground_truth[
        "baseline_escalation_probability"
    ].between(0, 1)
).all()

assert oracle_ground_truth[
    "true_urgency"
].isin(
    [
        "Low",
        "Medium",
        "High"
    ]
).all()


# --------------------------------------------------
# Results
# --------------------------------------------------

print(
    "===== HIDDEN ORACLE DATA CREATED ====="
)

print("\nRows:")
print(
    len(oracle_ground_truth)
)

print("\nTrue urgency counts:")
print(
    oracle_ground_truth[
        "true_urgency"
    ].value_counts()
)

print("\nTrue urgency percentages:")
print(
    oracle_ground_truth[
        "true_urgency"
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nUnderlying risk summary:")
print(
    oracle_ground_truth[
        "underlying_risk_score"
    ].describe()
)

print("\nBaseline escalation probability:")
print(
    oracle_ground_truth[
        "baseline_escalation_probability"
    ].describe()
)

print("\nSample:")
display(
    oracle_ground_truth.head(10)
)# --------------------------------------------------
# STEP 11B — Recalibrate hidden risk distribution
# --------------------------------------------------

# Preserve the original raw score for audit purposes
oracle_base["raw_risk_score"] = (
    oracle_base["underlying_risk_score"].copy()
)


# --------------------------------------------------
# Calibrate synthetic risk
# --------------------------------------------------
# The first version produced almost no High cases.
# Stretching the score gives us a more useful
# Low / Medium / High distribution while preserving
# relative risk ordering.
# --------------------------------------------------

oracle_base["underlying_risk_score"] = np.clip(
    0.05
    +
    1.25
    * oracle_base["raw_risk_score"],
    0,
    1
)


# --------------------------------------------------
# Reassign TRUE urgency using SAME thresholds
# --------------------------------------------------

def assign_true_urgency(score):

    if score < 0.40:
        return "Low"

    elif score < 0.70:
        return "Medium"

    else:
        return "High"


oracle_base["true_urgency"] = (
    oracle_base["underlying_risk_score"]
    .apply(assign_true_urgency)
)


# --------------------------------------------------
# Recalculate baseline escalation probability
# --------------------------------------------------
# Slightly lower intercept compensates for the
# higher calibrated risk scores.
# --------------------------------------------------

escalation_logit = (
    -3.40

    + 4.0
    * oracle_base["underlying_risk_score"]

    + 0.50
    * oracle_base["complaint_complexity"]

    + 0.25
    * oracle_base["prior_history_effect"]
)


oracle_base[
    "baseline_escalation_probability"
] = (
    1
    /
    (
        1
        +
        np.exp(-escalation_logit)
    )
)


# --------------------------------------------------
# Rebuild hidden oracle table
# --------------------------------------------------

oracle_ground_truth = oracle_base[
    [
        "complaint_id",
        "reference_department_id",
        "customer_complaint_propensity",
        "complaint_complexity",
        "issue_risk",
        "prior_history_effect",
        "financial_impact_effect",
        "raw_risk_score",
        "underlying_risk_score",
        "true_urgency",
        "baseline_escalation_probability"
    ]
].copy()


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert oracle_ground_truth[
    "complaint_id"
].is_unique

assert (
    oracle_ground_truth[
        "underlying_risk_score"
    ].between(0, 1)
).all()

assert (
    oracle_ground_truth[
        "baseline_escalation_probability"
    ].between(0, 1)
).all()


# --------------------------------------------------
# Results
# --------------------------------------------------

print(
    "===== RECALIBRATED ORACLE ====="
)

print("\nTrue urgency counts:")
print(
    oracle_ground_truth[
        "true_urgency"
    ].value_counts()
)

print("\nTrue urgency percentages:")
print(
    oracle_ground_truth[
        "true_urgency"
    ]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nCalibrated risk summary:")
print(
    oracle_ground_truth[
        "underlying_risk_score"
    ].describe()
)

print(
    "\nBaseline escalation probability summary:"
)

print(
    oracle_ground_truth[
        "baseline_escalation_probability"
    ].describe()
)# --------------------------------------------------
# STEP 12 — Create model predictions table
# --------------------------------------------------

prediction_base = (
    decision_context[
        [
            "complaint_id",
            "prediction_time"
        ]
    ]
    .merge(
        oracle_ground_truth[
            [
                "complaint_id",
                "reference_department_id",
                "underlying_risk_score",
                "baseline_escalation_probability",
                "true_urgency"
            ]
        ],
        on="complaint_id",
        how="left"
    )
)


# --------------------------------------------------
# Generate imperfect model urgency score
# --------------------------------------------------
# Add noise so predictions are related to the truth
# but are not perfectly accurate.
# --------------------------------------------------

urgency_noise = rng.normal(
    loc=0,
    scale=0.10,
    size=len(prediction_base)
)

prediction_base[
    "model_urgency_score"
] = np.clip(
    prediction_base[
        "underlying_risk_score"
    ]
    +
    urgency_noise,
    0,
    1
)


# --------------------------------------------------
# Convert model score to predicted urgency
# --------------------------------------------------

def assign_predicted_urgency(score):

    if score < 0.40:
        return "Low"

    elif score < 0.70:
        return "Medium"

    else:
        return "High"


prediction_base[
    "predicted_urgency"
] = (
    prediction_base[
        "model_urgency_score"
    ]
    .apply(assign_predicted_urgency)
)


# --------------------------------------------------
# Create urgency confidence/probability
# --------------------------------------------------
# Higher confidence when score is farther away
# from class boundaries.
# --------------------------------------------------

def urgency_confidence(score):

    if score < 0.40:
        distance = 0.40 - score

    elif score < 0.70:
        distance = min(
            score - 0.40,
            0.70 - score
        )

    else:
        distance = score - 0.70

    confidence = (
        0.55
        +
        min(distance * 1.5, 0.40)
    )

    return round(
        min(confidence, 0.95),
        4
    )


prediction_base[
    "urgency_probability"
] = (
    prediction_base[
        "model_urgency_score"
    ]
    .apply(urgency_confidence)
)


# --------------------------------------------------
# Generate predicted escalation probability
# --------------------------------------------------
# Start from hidden baseline truth and add model error.
# --------------------------------------------------

escalation_noise = rng.normal(
    loc=0,
    scale=0.07,
    size=len(prediction_base)
)

prediction_base[
    "escalation_probability"
] = np.clip(
    prediction_base[
        "baseline_escalation_probability"
    ]
    +
    escalation_noise,
    0.01,
    0.99
).round(4)


# --------------------------------------------------
# Generate recommended department
# --------------------------------------------------
# About 85% correct reference routing.
# About 15% imperfect recommendations.
#
# D06 Escalations is not used as normal
# initial product routing.
# --------------------------------------------------

normal_departments = [
    "D01",
    "D02",
    "D03",
    "D04",
    "D05"
]

recommended_departments = []

for reference_department in prediction_base[
    "reference_department_id"
]:

    # Model gets routing correct about 85% of time
    if rng.random() < 0.85:

        recommended_departments.append(
            reference_department
        )

    else:

        alternative_departments = [
            dept
            for dept in normal_departments
            if dept != reference_department
        ]

        recommended_departments.append(
            rng.choice(
                alternative_departments
            )
        )


prediction_base[
    "recommended_department_id"
] = recommended_departments


# --------------------------------------------------
# Model version
# --------------------------------------------------

prediction_base[
    "model_version"
] = "complaint_model_v1"


# --------------------------------------------------
# Create prediction IDs
# --------------------------------------------------

prediction_base = (
    prediction_base
    .sort_values(
        [
            "prediction_time",
            "complaint_id"
        ]
    )
    .reset_index(drop=True)
)

prediction_base[
    "prediction_id"
] = [
    f"PRED{i:07d}"
    for i in range(
        1,
        len(prediction_base) + 1
    )
]


# --------------------------------------------------
# Build physical predictions table
# --------------------------------------------------

predictions = prediction_base[
    [
        "prediction_id",
        "complaint_id",
        "recommended_department_id",
        "prediction_time",
        "model_version",
        "predicted_urgency",
        "urgency_probability",
        "escalation_probability"
    ]
].copy()


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert predictions[
    "prediction_id"
].is_unique

assert len(predictions) == len(complaints)

assert predictions[
    "complaint_id"
].notna().all()

assert predictions[
    "recommended_department_id"
].isin(
    departments[
        "department_id"
    ]
).all()

assert predictions[
    "predicted_urgency"
].isin(
    [
        "Low",
        "Medium",
        "High"
    ]
).all()

assert predictions[
    "urgency_probability"
].between(
    0,
    1
).all()

assert predictions[
    "escalation_probability"
].between(
    0,
    1
).all()


# --------------------------------------------------
# Evaluate synthetic prediction quality
# --------------------------------------------------

evaluation = prediction_base.copy()

evaluation[
    "urgency_correct"
] = (
    evaluation[
        "predicted_urgency"
    ]
    ==
    evaluation[
        "true_urgency"
    ]
)

evaluation[
    "routing_correct"
] = (
    evaluation[
        "recommended_department_id"
    ]
    ==
    evaluation[
        "reference_department_id"
    ]
)


urgency_accuracy = (
    evaluation[
        "urgency_correct"
    ].mean()
    * 100
)

routing_accuracy = (
    evaluation[
        "routing_correct"
    ].mean()
    * 100
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print(
    "===== PREDICTIONS TABLE CREATED ====="
)

print("\nRows:")
print(
    len(predictions)
)

print("\nPredicted urgency distribution:")
print(
    predictions[
        "predicted_urgency"
    ]
    .value_counts()
)

print("\nSynthetic urgency accuracy:")
print(
    round(
        urgency_accuracy,
        2
    ),
    "%"
)

print("\nSynthetic routing accuracy:")
print(
    round(
        routing_accuracy,
        2
    ),
    "%"
)

print(
    "\nEscalation probability summary:"
)

print(
    predictions[
        "escalation_probability"
    ]
    .describe()
)

print("\nSample predictions:")
display(
    predictions.head(10)
)# --------------------------------------------------
# STEP 13 — Create manager decisions table
# --------------------------------------------------

decision_base = (
    predictions
    .merge(
        complaints[
            [
                "complaint_id",
                "product",
                "issue"
            ]
        ],
        on="complaint_id",
        how="left"
    )
)


# --------------------------------------------------
# Generate manager decision time
# --------------------------------------------------
# Manager acts 1–15 minutes after prediction.
# --------------------------------------------------

decision_delay_seconds = rng.integers(
    60,
    901,
    size=len(decision_base)
)

decision_base["decision_time"] = (
    decision_base["prediction_time"]
    +
    pd.to_timedelta(
        decision_delay_seconds,
        unit="s"
    )
)


# --------------------------------------------------
# Decide whether manager follows routing model
# --------------------------------------------------

decision_base[
    "follow_model_route"
] = (
    rng.random(
        len(decision_base)
    )
    < 0.80
)


# --------------------------------------------------
# Create selected department
# --------------------------------------------------

selected_departments = []

for _, row in decision_base.iterrows():

    model_department = (
        row[
            "recommended_department_id"
        ]
    )

    if row["follow_model_route"]:

        # Follow model recommendation
        selected_departments.append(
            model_department
        )

    else:

        # Manager uses simple product/issue
        # business knowledge to override model
        manager_rule_department = (
            assign_reference_department(
                row["product"],
                row["issue"]
            )
        )

        # If manager's rule gives same department
        # as model, deliberately choose another
        # valid operational department so this row
        # is a genuine override.
        if (
            manager_rule_department
            == model_department
        ):

            alternatives = [
                "D01",
                "D02",
                "D03",
                "D04",
                "D05"
            ]

            alternatives = [
                dept
                for dept in alternatives
                if dept != model_department
            ]

            manager_rule_department = (
                rng.choice(
                    alternatives
                )
            )

        selected_departments.append(
            manager_rule_department
        )


decision_base[
    "selected_department_id"
] = selected_departments


# --------------------------------------------------
# Manager-selected priority
# --------------------------------------------------
# Usually follows predicted urgency,
# but sometimes adjusts one level.
# --------------------------------------------------

priority_levels = [
    "Low",
    "Medium",
    "High"
]

selected_priorities = []

for predicted_priority in decision_base[
    "predicted_urgency"
]:

    if rng.random() < 0.82:

        # Usually follow model
        selected_priorities.append(
            predicted_priority
        )

    else:

        current_index = (
            priority_levels.index(
                predicted_priority
            )
        )

        possible_indexes = [
            i
            for i in range(
                len(priority_levels)
            )
            if i != current_index
        ]

        # Prefer adjacent priority level
        adjacent_indexes = [
            i
            for i in possible_indexes
            if abs(
                i - current_index
            ) == 1
        ]

        if adjacent_indexes:

            new_index = rng.choice(
                adjacent_indexes
            )

        else:

            new_index = rng.choice(
                possible_indexes
            )

        selected_priorities.append(
            priority_levels[
                new_index
            ]
        )


decision_base[
    "selected_priority"
] = selected_priorities


# --------------------------------------------------
# Manager escalation decision
# --------------------------------------------------
# Higher probability when:
# - predicted urgency is High
# - model escalation probability is high
# --------------------------------------------------

manager_escalation_probability = (
    0.05
    +
    0.55
    * decision_base[
        "escalation_probability"
    ]
    +
    0.20
    * (
        decision_base[
            "predicted_urgency"
        ] == "High"
    ).astype(float)
    +
    0.07
    * (
        decision_base[
            "predicted_urgency"
        ] == "Medium"
    ).astype(float)
)

manager_escalation_probability = np.clip(
    manager_escalation_probability,
    0.02,
    0.90
)


decision_base[
    "manager_escalation_decision"
] = (
    rng.random(
        len(decision_base)
    )
    <
    manager_escalation_probability
)


# --------------------------------------------------
# Create decision IDs
# --------------------------------------------------

decision_base = (
    decision_base
    .sort_values(
        [
            "decision_time",
            "complaint_id"
        ]
    )
    .reset_index(drop=True)
)

decision_base[
    "decision_id"
] = [
    f"DEC{i:07d}"
    for i in range(
        1,
        len(decision_base) + 1
    )
]


# --------------------------------------------------
# Build physical decisions table
# --------------------------------------------------

decisions = decision_base[
    [
        "decision_id",
        "complaint_id",
        "prediction_id",
        "selected_department_id",
        "decision_time",
        "selected_priority",
        "manager_escalation_decision"
    ]
].copy()


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert decisions[
    "decision_id"
].is_unique

assert decisions[
    "complaint_id"
].is_unique

assert len(decisions) == len(
    complaints
)

assert decisions[
    "selected_department_id"
].isin(
    departments[
        "department_id"
    ]
).all()

assert decisions[
    "selected_priority"
].isin(
    [
        "Low",
        "Medium",
        "High"
    ]
).all()


# Decision must occur after prediction
decision_time_check = (
    decisions
    .merge(
        predictions[
            [
                "prediction_id",
                "prediction_time"
            ]
        ],
        on="prediction_id",
        how="left"
    )
)

invalid_decision_times = (
    decision_time_check[
        "decision_time"
    ]
    <=
    decision_time_check[
        "prediction_time"
    ]
).sum()

assert invalid_decision_times == 0


# --------------------------------------------------
# Evaluate manager behavior
# --------------------------------------------------

manager_eval = (
    decisions
    .merge(
        predictions[
            [
                "prediction_id",
                "recommended_department_id",
                "predicted_urgency"
            ]
        ],
        on="prediction_id",
        how="left"
    )
)


route_follow_rate = (
    manager_eval[
        "selected_department_id"
    ]
    ==
    manager_eval[
        "recommended_department_id"
    ]
).mean() * 100


priority_follow_rate = (
    manager_eval[
        "selected_priority"
    ]
    ==
    manager_eval[
        "predicted_urgency"
    ]
).mean() * 100


escalation_rate = (
    decisions[
        "manager_escalation_decision"
    ].mean()
    * 100
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print(
    "===== DECISIONS TABLE CREATED ====="
)

print("\nRows:")
print(len(decisions))

print("\nManager follows model routing:")
print(
    round(
        route_follow_rate,
        2
    ),
    "%"
)

print("\nManager follows predicted priority:")
print(
    round(
        priority_follow_rate,
        2
    ),
    "%"
)

print("\nManager escalation decision rate:")
print(
    round(
        escalation_rate,
        2
    ),
    "%"
)

print("\nInvalid decision times:")
print(
    invalid_decision_times
)

print("\nSample decisions:")
display(
    decisions.head(10)
)# --------------------------------------------------
# STEP 14 — Generate complaint workflow events
# --------------------------------------------------

observation_end = pd.Timestamp(
    "2026-09-23 23:59:59"
)


# --------------------------------------------------
# Build event-generation working table
# --------------------------------------------------

event_work = (
    decisions
    .merge(
        complaints[
            [
                "complaint_id",
                "complaint_timestamp"
            ]
        ],
        on="complaint_id",
        how="left"
    )
    .merge(
        oracle_ground_truth[
            [
                "complaint_id",
                "reference_department_id",
                "complaint_complexity",
                "underlying_risk_score",
                "baseline_escalation_probability",
                "prior_history_effect",
                "financial_impact_effect"
            ]
        ],
        on="complaint_id",
        how="left"
    )
    .merge(
        decision_context[
            [
                "complaint_id",
                "current_workload"
            ]
        ],
        on="complaint_id",
        how="left"
    )
)


# Was manager's selected department the reference route?
event_work["routing_match"] = (
    event_work["selected_department_id"]
    ==
    event_work["reference_department_id"]
)


# --------------------------------------------------
# Final latent escalation probability
# --------------------------------------------------

event_work[
    "workflow_escalation_probability"
] = (
    event_work[
        "baseline_escalation_probability"
    ]

    + 0.18
    * event_work[
        "manager_escalation_decision"
    ].astype(float)

    + 0.10
    * (
        ~event_work[
            "routing_match"
        ]
    ).astype(float)

    + 0.08
    * event_work[
        "complaint_complexity"
    ]

    + 0.05
    * event_work[
        "prior_history_effect"
    ]

    + 0.05
    * (
        event_work[
            "current_workload"
        ] / 10
    ).clip(0, 1)

    + rng.normal(
        0,
        0.04,
        size=len(event_work)
    )
)

event_work[
    "workflow_escalation_probability"
] = np.clip(
    event_work[
        "workflow_escalation_probability"
    ],
    0.02,
    0.95
)


event_work["latent_will_escalate"] = (
    rng.random(
        len(event_work)
    )
    <
    event_work[
        "workflow_escalation_probability"
    ]
)


# --------------------------------------------------
# Generate events
# --------------------------------------------------

event_rows = []
event_counter = 1

operational_departments = [
    "D01",
    "D02",
    "D03",
    "D04",
    "D05"
]


def add_event(
    complaint_id,
    department_id,
    event_type,
    timestamp,
    status
):
    global event_counter

    event_rows.append({
        "event_id":
            f"EVT{event_counter:08d}",

        "complaint_id":
            complaint_id,

        "department_id":
            department_id,

        "event_type":
            event_type,

        "event_timestamp":
            timestamp,

        "status":
            status
    })

    event_counter += 1


for _, row in event_work.iterrows():

    complaint_id = row["complaint_id"]

    current_department = (
        row["selected_department_id"]
    )

    reference_department = (
        row["reference_department_id"]
    )

    decision_time = row["decision_time"]

    complexity = row[
        "complaint_complexity"
    ]

    workload = row[
        "current_workload"
    ]


    # ------------------------------------------
    # ASSIGNED
    # ------------------------------------------

    assigned_time = (
        decision_time
        +
        pd.Timedelta(
            minutes=int(
                rng.integers(1, 11)
            )
        )
    )

    if assigned_time <= observation_end:

        add_event(
            complaint_id,
            current_department,
            "ASSIGNED",
            assigned_time,
            "Assigned"
        )


    # ------------------------------------------
    # REVIEW_STARTED
    # ------------------------------------------

    review_time = (
        assigned_time
        +
        pd.Timedelta(
            minutes=int(
                rng.integers(5, 241)
            )
        )
    )

    last_event_time = review_time

    if review_time <= observation_end:

        add_event(
            complaint_id,
            current_department,
            "REVIEW_STARTED",
            review_time,
            "In Review"
        )


    # ------------------------------------------
    # INFORMATION_REQUESTED
    # More likely for complex complaints
    # ------------------------------------------

    info_probability = min(
        0.15
        +
        0.40 * complexity,
        0.70
    )

    information_requested = (
        rng.random()
        <
        info_probability
    )

    if information_requested:

        info_time = (
            last_event_time
            +
            pd.Timedelta(
                minutes=int(
                    rng.integers(
                        30,
                        721
                    )
                )
            )
        )

        if info_time <= observation_end:

            add_event(
                complaint_id,
                current_department,
                "INFORMATION_REQUESTED",
                info_time,
                "Waiting for Information"
            )

            last_event_time = info_time


    # ------------------------------------------
    # Determine number of transfers
    # ------------------------------------------

    if row["routing_match"]:

        transfer_count = rng.choice(
            [0, 1, 2, 3],
            p=[
                0.90,
                0.08,
                0.015,
                0.005
            ]
        )

    else:

        transfer_count = rng.choice(
            [0, 1, 2, 3],
            p=[
                0.55,
                0.30,
                0.10,
                0.05
            ]
        )


    # ------------------------------------------
    # TRANSFERRED events
    # ------------------------------------------

    for transfer_number in range(
        int(transfer_count)
    ):

        transfer_time = (
            last_event_time
            +
            pd.Timedelta(
                minutes=int(
                    rng.integers(
                        30,
                        721
                    )
                )
            )
        )

        # If routing is wrong, usually move
        # toward reference department
        if (
            current_department
            != reference_department
            and rng.random() < 0.75
        ):

            next_department = (
                reference_department
            )

        else:

            alternatives = [
                dept
                for dept
                in operational_departments
                if dept
                != current_department
            ]

            next_department = rng.choice(
                alternatives
            )


        if transfer_time <= observation_end:

            current_department = (
                next_department
            )

            add_event(
                complaint_id,
                current_department,
                "TRANSFERRED",
                transfer_time,
                "Transferred"
            )

            last_event_time = (
                transfer_time
            )


    # ------------------------------------------
    # ESCALATED event
    # ------------------------------------------

    if row["latent_will_escalate"]:

        escalation_time = (
            last_event_time
            +
            pd.Timedelta(
                minutes=int(
                    rng.integers(
                        30,
                        1441
                    )
                )
            )
        )

        if escalation_time <= observation_end:

            current_department = "D06"

            add_event(
                complaint_id,
                current_department,
                "ESCALATED",
                escalation_time,
                "Escalated"
            )

            last_event_time = (
                escalation_time
            )


    # ------------------------------------------
    # Generate resolution duration
    # ------------------------------------------

    base_resolution_hours = rng.lognormal(
        mean=np.log(36),
        sigma=0.75
    )

    resolution_multiplier = (
        1
        +
        1.20 * complexity
        +
        0.25 * transfer_count
        +
        0.50 * float(
            row["latent_will_escalate"]
        )
        +
        0.25 * float(
            information_requested
        )
        +
        0.15
        * max(
            workload - 8,
            0
        )
    )

    resolution_hours = (
        base_resolution_hours
        *
        resolution_multiplier
    )

    # Keep synthetic duration within
    # a practical range
    resolution_hours = np.clip(
        resolution_hours,
        2,
        720
    )


    candidate_resolution_time = (
        row["complaint_timestamp"]
        +
        pd.Timedelta(
            hours=float(
                resolution_hours
            )
        )
    )

    # Resolution must occur after all
    # previous observed workflow events
    candidate_resolution_time = max(
        candidate_resolution_time,
        last_event_time
        +
        pd.Timedelta(minutes=15)
    )


    # ------------------------------------------
    # RESOLVED
    # ------------------------------------------

    if (
        candidate_resolution_time
        <= observation_end
    ):

        add_event(
            complaint_id,
            current_department,
            "RESOLVED",
            candidate_resolution_time,
            "Resolved"
        )

        last_event_time = (
            candidate_resolution_time
        )


        # --------------------------------------
        # CLOSED
        # Usually follows resolution
        # --------------------------------------

        if rng.random() < 0.90:

            close_time = (
                candidate_resolution_time
                +
                pd.Timedelta(
                    minutes=int(
                        rng.integers(
                            60,
                            2881
                        )
                    )
                )
            )

            if close_time <= observation_end:

                add_event(
                    complaint_id,
                    current_department,
                    "CLOSED",
                    close_time,
                    "Closed"
                )


# --------------------------------------------------
# Build complaint_events table
# --------------------------------------------------

complaint_events = pd.DataFrame(
    event_rows
)

complaint_events = (
    complaint_events
    .sort_values(
        [
            "complaint_id",
            "event_timestamp",
            "event_id"
        ]
    )
    .reset_index(drop=True)
)


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert complaint_events[
    "event_id"
].is_unique

assert complaint_events[
    "complaint_id"
].isin(
    complaints[
        "complaint_id"
    ]
).all()

assert complaint_events[
    "department_id"
].isin(
    departments[
        "department_id"
    ]
).all()


valid_mapping = {
    "ASSIGNED":
        "Assigned",

    "REVIEW_STARTED":
        "In Review",

    "INFORMATION_REQUESTED":
        "Waiting for Information",

    "TRANSFERRED":
        "Transferred",

    "ESCALATED":
        "Escalated",

    "RESOLVED":
        "Resolved",

    "CLOSED":
        "Closed"
}


mapping_errors = (
    complaint_events.apply(
        lambda row:
        valid_mapping[
            row["event_type"]
        ]
        != row["status"],
        axis=1
    )
).sum()

assert mapping_errors == 0


# --------------------------------------------------
# Events cannot precede manager decision
# --------------------------------------------------

event_time_check = (
    complaint_events
    .merge(
        decisions[
            [
                "complaint_id",
                "decision_time"
            ]
        ],
        on="complaint_id",
        how="left"
    )
)

events_before_decision = (
    event_time_check[
        "event_timestamp"
    ]
    <
    event_time_check[
        "decision_time"
    ]
).sum()

assert events_before_decision == 0


# --------------------------------------------------
# Check chronological order
# --------------------------------------------------

chronology_errors = 0

for complaint_id, group in (
    complaint_events
    .groupby("complaint_id")
):

    times = (
        group[
            "event_timestamp"
        ]
        .sort_values()
    )

    if not times.is_monotonic_increasing:
        chronology_errors += 1

assert chronology_errors == 0


# --------------------------------------------------
# Results
# --------------------------------------------------

print(
    "===== COMPLAINT EVENTS TABLE CREATED ====="
)

print("\nTotal event rows:")
print(
    len(complaint_events)
)

print("\nEvent counts:")
print(
    complaint_events[
        "event_type"
    ].value_counts()
)

print("\nComplaints with escalation event:")
print(
    complaint_events.loc[
        complaint_events[
            "event_type"
        ] == "ESCALATED",
        "complaint_id"
    ].nunique()
)

print("\nComplaints with transfer:")
print(
    complaint_events.loc[
        complaint_events[
            "event_type"
        ] == "TRANSFERRED",
        "complaint_id"
    ].nunique()
)

print("\nComplaints resolved:")
print(
    complaint_events.loc[
        complaint_events[
            "event_type"
        ] == "RESOLVED",
        "complaint_id"
    ].nunique()
)

print("\nComplaints closed:")
print(
    complaint_events.loc[
        complaint_events[
            "event_type"
        ] == "CLOSED",
        "complaint_id"
    ].nunique()
)

print("\nEvents before decision:")
print(
    events_before_decision
)

print("\nStatus mapping errors:")
print(
    mapping_errors
)

print("\nSample events:")
display(
    complaint_events.head(20)
)# --------------------------------------------------
# STEP 15 — Create outcomes table
# --------------------------------------------------

# --------------------------------------------------
# Get RESOLVED events
# --------------------------------------------------

resolved_events = (
    complaint_events[
        complaint_events["event_type"]
        == "RESOLVED"
    ][
        [
            "complaint_id",
            "department_id",
            "event_timestamp"
        ]
    ]
    .copy()
    .rename(
        columns={
            "department_id":
                "resolved_department_id",

            "event_timestamp":
                "resolved_timestamp"
        }
    )
)


# --------------------------------------------------
# Get CLOSED events
# --------------------------------------------------

closed_events = (
    complaint_events[
        complaint_events["event_type"]
        == "CLOSED"
    ][
        [
            "complaint_id",
            "event_timestamp"
        ]
    ]
    .copy()
    .rename(
        columns={
            "event_timestamp":
                "closed_timestamp"
        }
    )
)


# --------------------------------------------------
# Identify complaints that escalated
# --------------------------------------------------

escalated_complaints = set(
    complaint_events.loc[
        complaint_events["event_type"]
        == "ESCALATED",
        "complaint_id"
    ]
)


# --------------------------------------------------
# Count transfers and information requests
# --------------------------------------------------

transfer_counts = (
    complaint_events[
        complaint_events["event_type"]
        == "TRANSFERRED"
    ]
    .groupby("complaint_id")
    .size()
    .rename("transfer_count")
)

information_counts = (
    complaint_events[
        complaint_events["event_type"]
        == "INFORMATION_REQUESTED"
    ]
    .groupby("complaint_id")
    .size()
    .rename("information_request_count")
)


# --------------------------------------------------
# Start with every complaint
# --------------------------------------------------

outcome_work = complaints[
    [
        "complaint_id",
        "complaint_timestamp"
    ]
].copy()


# Add resolved and closed event information
outcome_work = (
    outcome_work
    .merge(
        resolved_events,
        on="complaint_id",
        how="left"
    )
    .merge(
        closed_events,
        on="complaint_id",
        how="left"
    )
)


# Add transfer counts
outcome_work = outcome_work.merge(
    transfer_counts,
    on="complaint_id",
    how="left"
)


# Add information-request counts
outcome_work = outcome_work.merge(
    information_counts,
    on="complaint_id",
    how="left"
)


# Missing counts mean zero
outcome_work[
    "transfer_count"
] = (
    outcome_work[
        "transfer_count"
    ]
    .fillna(0)
    .astype(int)
)

outcome_work[
    "information_request_count"
] = (
    outcome_work[
        "information_request_count"
    ]
    .fillna(0)
    .astype(int)
)


# --------------------------------------------------
# Bring complaint complexity
# --------------------------------------------------

outcome_work = outcome_work.merge(
    oracle_ground_truth[
        [
            "complaint_id",
            "complaint_complexity"
        ]
    ],
    on="complaint_id",
    how="left"
)


# --------------------------------------------------
# Determine final status
# --------------------------------------------------

def determine_final_status(row):

    if pd.notna(
        row["closed_timestamp"]
    ):
        return "Closed"

    elif pd.notna(
        row["resolved_timestamp"]
    ):
        return "Resolved"

    else:
        return "Pending"


outcome_work[
    "final_status"
] = outcome_work.apply(
    determine_final_status,
    axis=1
)


# --------------------------------------------------
# Actual escalation
# --------------------------------------------------
# Resolved/closed cases:
# True if ESCALATED event exists,
# False otherwise.
#
# Pending cases:
# True if escalation already observed,
# otherwise unknown (NULL).
# --------------------------------------------------

actual_escalation = []

for _, row in outcome_work.iterrows():

    complaint_id = row[
        "complaint_id"
    ]

    escalated = (
        complaint_id
        in escalated_complaints
    )

    if escalated:

        actual_escalation.append(
            True
        )

    elif row["final_status"] in [
        "Resolved",
        "Closed"
    ]:

        actual_escalation.append(
            False
        )

    else:

        actual_escalation.append(
            pd.NA
        )


outcome_work[
    "actual_escalation"
] = pd.array(
    actual_escalation,
    dtype="boolean"
)


# --------------------------------------------------
# Resolution time in hours
# --------------------------------------------------

outcome_work[
    "resolution_time"
] = (
    (
        outcome_work[
            "resolved_timestamp"
        ]
        -
        outcome_work[
            "complaint_timestamp"
        ]
    )
    .dt.total_seconds()
    /
    3600
)

outcome_work[
    "resolution_time"
] = (
    outcome_work[
        "resolution_time"
    ]
    .round(2)
)


# --------------------------------------------------
# Final department
# --------------------------------------------------
# For resolved complaints, use the department
# that actually resolved the complaint.
# Pending complaints remain NULL.
# --------------------------------------------------

outcome_work[
    "final_department_id"
] = outcome_work[
    "resolved_department_id"
]


# --------------------------------------------------
# Date closed
# --------------------------------------------------

outcome_work[
    "date_closed"
] = (
    outcome_work[
        "closed_timestamp"
    ].dt.date
)


# --------------------------------------------------
# Generate synthetic internal handling cost
# --------------------------------------------------
# Cost depends on:
# - resolution duration
# - complexity
# - transfers
# - information requests
# - escalation
#
# Pending cases remain NULL.
# --------------------------------------------------

handling_cost = []

for _, row in outcome_work.iterrows():

    if pd.isna(
        row["resolution_time"]
    ):

        handling_cost.append(
            np.nan
        )

        continue


    escalation_cost = (
        180
        if row["actual_escalation"] is True
        else 0
    )

    base_cost = 75

    time_cost = (
        row["resolution_time"]
        * 6
    )

    complexity_cost = (
        row["complaint_complexity"]
        * 150
    )

    transfer_cost = (
        row["transfer_count"]
        * 60
    )

    information_cost = (
        row[
            "information_request_count"
        ]
        * 35
    )

    random_cost_variation = (
        rng.lognormal(
            mean=np.log(40),
            sigma=0.40
        )
    )

    total_cost = (
        base_cost
        +
        time_cost
        +
        complexity_cost
        +
        transfer_cost
        +
        information_cost
        +
        escalation_cost
        +
        random_cost_variation
    )

    handling_cost.append(
        round(
            float(total_cost),
            2
        )
    )


outcome_work[
    "handling_cost"
] = handling_cost


# --------------------------------------------------
# Create outcome IDs
# --------------------------------------------------

outcome_work = (
    outcome_work
    .sort_values(
        "complaint_timestamp"
    )
    .reset_index(drop=True)
)

outcome_work[
    "outcome_id"
] = [
    f"OUT{i:07d}"
    for i in range(
        1,
        len(outcome_work) + 1
    )
]


# --------------------------------------------------
# Build physical outcomes table
# --------------------------------------------------

outcomes = outcome_work[
    [
        "outcome_id",
        "complaint_id",
        "final_department_id",
        "actual_escalation",
        "resolution_time",
        "final_status",
        "date_closed",
        "handling_cost"
    ]
].copy()


# --------------------------------------------------
# Validation
# --------------------------------------------------

assert outcomes[
    "outcome_id"
].is_unique

assert outcomes[
    "complaint_id"
].is_unique

assert len(outcomes) == len(
    complaints
)


# Resolution times cannot be negative
assert (
    outcomes[
        "resolution_time"
    ]
    .dropna()
    >= 0
).all()


# Handling costs cannot be negative
assert (
    outcomes[
        "handling_cost"
    ]
    .dropna()
    >= 0
).all()


# --------------------------------------------------
# CLOSED validation
# --------------------------------------------------

closed_check = outcomes[
    outcomes["final_status"]
    == "Closed"
]

assert closed_check[
    "resolution_time"
].notna().all()

assert closed_check[
    "date_closed"
].notna().all()

assert closed_check[
    "final_department_id"
].notna().all()


# --------------------------------------------------
# RESOLVED validation
# --------------------------------------------------

resolved_check = outcomes[
    outcomes["final_status"]
    == "Resolved"
]

assert resolved_check[
    "resolution_time"
].notna().all()

assert resolved_check[
    "final_department_id"
].notna().all()

assert resolved_check[
    "date_closed"
].isna().all()


# --------------------------------------------------
# PENDING validation
# --------------------------------------------------

pending_check = outcomes[
    outcomes["final_status"]
    == "Pending"
]

assert pending_check[
    "resolution_time"
].isna().all()

assert pending_check[
    "date_closed"
].isna().all()


# --------------------------------------------------
# Every True actual_escalation must have
# an ESCALATED workflow event
# --------------------------------------------------

true_escalations = set(
    outcomes.loc[
        outcomes[
            "actual_escalation"
        ] == True,
        "complaint_id"
    ]
)

assert true_escalations.issubset(
    escalated_complaints
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print(
    "===== OUTCOMES TABLE CREATED ====="
)

print("\nRows:")
print(
    len(outcomes)
)

print("\nFinal status counts:")
print(
    outcomes[
        "final_status"
    ].value_counts()
)

print("\nActual escalation:")
print(
    outcomes[
        "actual_escalation"
    ].value_counts(
        dropna=False
    )
)

print("\nResolution time summary (hours):")
print(
    outcomes[
        "resolution_time"
    ].describe()
)

print("\nHandling cost summary:")
print(
    outcomes[
        "handling_cost"
    ].describe()
)

print("\nPending complaints:")
print(
    (
        outcomes[
            "final_status"
        ] == "Pending"
    ).sum()
)

print("\nSample outcomes:")
display(
    outcomes.head(10)
)# --------------------------------------------------
# STEP 16 — CREATE ACTUAL SQLITE DATABASE
# --------------------------------------------------

import os
import sqlite3
import pandas as pd
import numpy as np


# --------------------------------------------------
# Database file
# --------------------------------------------------

db_path = "customer_complaints.db"

# Remove old version if this cell is rerun
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)

# Turn on foreign-key enforcement
conn.execute("PRAGMA foreign_keys = ON;")


# --------------------------------------------------
# CREATE DATABASE SCHEMA
# --------------------------------------------------

schema_sql = """

-- ==================================================
-- 1. CUSTOMERS
-- ==================================================

CREATE TABLE customers (

    customer_id TEXT PRIMARY KEY,

    customer_type TEXT NOT NULL,

    CHECK (
        customer_type IN (
            'Individual',
            'Business'
        )
    )
);


-- ==================================================
-- 2. DEPARTMENTS
-- ==================================================

CREATE TABLE departments (

    department_id TEXT PRIMARY KEY,

    department_name TEXT
        NOT NULL
        UNIQUE
);


-- ==================================================
-- 3. COMPLAINTS
-- ==================================================

CREATE TABLE complaints (

    complaint_id TEXT PRIMARY KEY,

    customer_id TEXT NOT NULL,

    date_received TEXT NOT NULL,

    complaint_timestamp TEXT NOT NULL,

    product TEXT NOT NULL,

    sub_product TEXT,

    issue TEXT NOT NULL,

    sub_issue TEXT,

    submission_channel TEXT,

    complaint_narrative TEXT,

    FOREIGN KEY (
        customer_id
    )
    REFERENCES customers (
        customer_id
    ),

    CHECK (
        DATE(complaint_timestamp)
        =
        date_received
    )
);


-- ==================================================
-- 4. CAPACITY
-- ==================================================

CREATE TABLE capacity (

    capacity_id TEXT PRIMARY KEY,

    department_id TEXT NOT NULL,

    capacity_timestamp TEXT NOT NULL,

    available_staff INTEGER NOT NULL,

    current_open_cases INTEGER NOT NULL,

    available_capacity INTEGER NOT NULL,

    FOREIGN KEY (
        department_id
    )
    REFERENCES departments (
        department_id
    ),

    UNIQUE (
        department_id,
        capacity_timestamp
    ),

    CHECK (
        available_staff >= 0
    ),

    CHECK (
        current_open_cases >= 0
    ),

    CHECK (
        available_capacity >= 0
    )
);


-- ==================================================
-- 5. DECISION CONTEXT
-- ==================================================

CREATE TABLE decision_context (

    decision_context_id TEXT
        PRIMARY KEY,

    complaint_id TEXT
        NOT NULL
        UNIQUE,

    context_department_id TEXT
        NOT NULL,

    prediction_time TEXT
        NOT NULL,

    prior_complaint_count INTEGER
        NOT NULL,

    known_financial_impact REAL,

    current_workload REAL
        NOT NULL,

    available_capacity INTEGER
        NOT NULL,

    FOREIGN KEY (
        complaint_id
    )
    REFERENCES complaints (
        complaint_id
    ),

    FOREIGN KEY (
        context_department_id
    )
    REFERENCES departments (
        department_id
    ),

    CHECK (
        prior_complaint_count >= 0
    ),

    CHECK (
        known_financial_impact IS NULL
        OR
        known_financial_impact >= 0
    ),

    CHECK (
        current_workload >= 0
    ),

    CHECK (
        available_capacity >= 0
    )
);


-- ==================================================
-- 6. PREDICTIONS
-- ==================================================

CREATE TABLE predictions (

    prediction_id TEXT
        PRIMARY KEY,

    complaint_id TEXT
        NOT NULL,

    recommended_department_id TEXT
        NOT NULL,

    prediction_time TEXT
        NOT NULL,

    model_version TEXT
        NOT NULL,

    predicted_urgency TEXT
        NOT NULL,

    urgency_probability REAL
        NOT NULL,

    escalation_probability REAL
        NOT NULL,

    FOREIGN KEY (
        complaint_id
    )
    REFERENCES complaints (
        complaint_id
    ),

    FOREIGN KEY (
        recommended_department_id
    )
    REFERENCES departments (
        department_id
    ),

    UNIQUE (
        complaint_id,
        model_version,
        prediction_time
    ),

    CHECK (
        predicted_urgency IN (
            'Low',
            'Medium',
            'High'
        )
    ),

    CHECK (
        urgency_probability
        BETWEEN 0 AND 1
    ),

    CHECK (
        escalation_probability
        BETWEEN 0 AND 1
    )
);


-- ==================================================
-- 7. DECISIONS
-- ==================================================

CREATE TABLE decisions (

    decision_id TEXT
        PRIMARY KEY,

    complaint_id TEXT
        NOT NULL
        UNIQUE,

    prediction_id TEXT
        NOT NULL,

    selected_department_id TEXT
        NOT NULL,

    decision_time TEXT
        NOT NULL,

    selected_priority TEXT
        NOT NULL,

    manager_escalation_decision INTEGER
        NOT NULL,

    FOREIGN KEY (
        complaint_id
    )
    REFERENCES complaints (
        complaint_id
    ),

    FOREIGN KEY (
        prediction_id
    )
    REFERENCES predictions (
        prediction_id
    ),

    FOREIGN KEY (
        selected_department_id
    )
    REFERENCES departments (
        department_id
    ),

    CHECK (
        selected_priority IN (
            'Low',
            'Medium',
            'High'
        )
    ),

    CHECK (
        manager_escalation_decision
        IN (0, 1)
    )
);


-- ==================================================
-- 8. COMPLAINT EVENTS
-- ==================================================

CREATE TABLE complaint_events (

    event_id TEXT
        PRIMARY KEY,

    complaint_id TEXT
        NOT NULL,

    department_id TEXT
        NOT NULL,

    event_type TEXT
        NOT NULL,

    event_timestamp TEXT
        NOT NULL,

    status TEXT
        NOT NULL,

    FOREIGN KEY (
        complaint_id
    )
    REFERENCES complaints (
        complaint_id
    ),

    FOREIGN KEY (
        department_id
    )
    REFERENCES departments (
        department_id
    ),

    CHECK (
        event_type IN (
            'ASSIGNED',
            'REVIEW_STARTED',
            'INFORMATION_REQUESTED',
            'TRANSFERRED',
            'ESCALATED',
            'RESOLVED',
            'CLOSED'
        )
    ),

    CHECK (

        (event_type = 'ASSIGNED'
         AND status = 'Assigned')

        OR

        (event_type = 'REVIEW_STARTED'
         AND status = 'In Review')

        OR

        (event_type = 'INFORMATION_REQUESTED'
         AND status = 'Waiting for Information')

        OR

        (event_type = 'TRANSFERRED'
         AND status = 'Transferred')

        OR

        (event_type = 'ESCALATED'
         AND status = 'Escalated')

        OR

        (event_type = 'RESOLVED'
         AND status = 'Resolved')

        OR

        (event_type = 'CLOSED'
         AND status = 'Closed')
    )
);


-- ==================================================
-- 9. OUTCOMES
-- ==================================================

CREATE TABLE outcomes (

    outcome_id TEXT
        PRIMARY KEY,

    complaint_id TEXT
        NOT NULL
        UNIQUE,

    final_department_id TEXT,

    actual_escalation INTEGER,

    resolution_time REAL,

    final_status TEXT
        NOT NULL,

    date_closed TEXT,

    handling_cost REAL,

    FOREIGN KEY (
        complaint_id
    )
    REFERENCES complaints (
        complaint_id
    ),

    FOREIGN KEY (
        final_department_id
    )
    REFERENCES departments (
        department_id
    ),

    CHECK (
        actual_escalation IS NULL
        OR
        actual_escalation IN (0, 1)
    ),

    CHECK (
        resolution_time IS NULL
        OR
        resolution_time >= 0
    ),

    CHECK (
        handling_cost IS NULL
        OR
        handling_cost >= 0
    ),

    CHECK (
        final_status IN (
            'Pending',
            'Resolved',
            'Closed'
        )
    ),

    CHECK (

        (
            final_status = 'Pending'
            AND resolution_time IS NULL
            AND date_closed IS NULL
        )

        OR

        (
            final_status = 'Resolved'
            AND resolution_time IS NOT NULL
            AND date_closed IS NULL
        )

        OR

        (
            final_status = 'Closed'
            AND resolution_time IS NOT NULL
            AND date_closed IS NOT NULL
        )
    )
);


-- ==================================================
-- INDEXES
-- ==================================================

CREATE INDEX idx_complaints_customer_time
ON complaints (
    customer_id,
    complaint_timestamp
);


CREATE INDEX idx_complaints_date
ON complaints (
    date_received
);


CREATE INDEX idx_context_prediction_time
ON decision_context (
    prediction_time
);


CREATE INDEX idx_predictions_complaint_time
ON predictions (
    complaint_id,
    prediction_time
);


CREATE INDEX idx_predictions_model
ON predictions (
    model_version
);


CREATE INDEX idx_predictions_department
ON predictions (
    recommended_department_id
);


CREATE INDEX idx_decisions_department
ON decisions (
    selected_department_id
);


CREATE INDEX idx_capacity_department_time
ON capacity (
    department_id,
    capacity_timestamp
);


CREATE INDEX idx_events_complaint_time
ON complaint_events (
    complaint_id,
    event_timestamp
);


CREATE INDEX idx_events_department_time
ON complaint_events (
    department_id,
    event_timestamp
);


CREATE INDEX idx_outcomes_department
ON outcomes (
    final_department_id
);

"""


conn.executescript(schema_sql)

print("Database schema created successfully!")


# --------------------------------------------------
# HELPER FUNCTION
# Convert pandas values to SQLite-friendly formats
# --------------------------------------------------

def prepare_sqlite_dataframe(
    dataframe,
    datetime_columns=None,
    date_columns=None,
    boolean_columns=None
):

    out = dataframe.copy()

    datetime_columns = (
        datetime_columns or []
    )

    date_columns = (
        date_columns or []
    )

    boolean_columns = (
        boolean_columns or []
    )


    # ----------------------------------------------
    # Exact timestamps
    # ----------------------------------------------

    for col in datetime_columns:

        out[col] = pd.to_datetime(
            out[col],
            errors="coerce"
        ).dt.strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )


    # ----------------------------------------------
    # Calendar dates
    # ----------------------------------------------

    for col in date_columns:

        out[col] = pd.to_datetime(
            out[col],
            errors="coerce"
        ).dt.strftime(
            "%Y-%m-%d"
        )


    # ----------------------------------------------
    # SQLite booleans = 0/1/NULL
    # ----------------------------------------------

    for col in boolean_columns:

        out[col] = out[col].apply(
            lambda value:
                None
                if pd.isna(value)
                else int(bool(value))
        )


    # ----------------------------------------------
    # Replace pandas NA / NaN with Python None
    # ----------------------------------------------

    out = out.astype(object)

    out = out.where(
        pd.notna(out),
        None
    )

    return out


# --------------------------------------------------
# PREPARE TABLES FOR SQLITE
# --------------------------------------------------

customers_sql = prepare_sqlite_dataframe(
    customers
)


departments_sql = prepare_sqlite_dataframe(
    departments
)


complaints_sql = prepare_sqlite_dataframe(
    complaints,
    datetime_columns=[
        "complaint_timestamp"
    ],
    date_columns=[
        "date_received"
    ]
)


capacity_sql = prepare_sqlite_dataframe(
    capacity,
    datetime_columns=[
        "capacity_timestamp"
    ]
)


decision_context_sql = (
    prepare_sqlite_dataframe(
        decision_context,
        datetime_columns=[
            "prediction_time"
        ]
    )
)


predictions_sql = prepare_sqlite_dataframe(
    predictions,
    datetime_columns=[
        "prediction_time"
    ]
)


decisions_sql = prepare_sqlite_dataframe(
    decisions,
    datetime_columns=[
        "decision_time"
    ],
    boolean_columns=[
        "manager_escalation_decision"
    ]
)


complaint_events_sql = (
    prepare_sqlite_dataframe(
        complaint_events,
        datetime_columns=[
            "event_timestamp"
        ]
    )
)


outcomes_sql = prepare_sqlite_dataframe(
    outcomes,
    date_columns=[
        "date_closed"
    ],
    boolean_columns=[
        "actual_escalation"
    ]
)


# --------------------------------------------------
# LOAD DATA INTO DATABASE
# Insert parent tables first
# --------------------------------------------------

customers_sql.to_sql(
    "customers",
    conn,
    if_exists="append",
    index=False
)


departments_sql.to_sql(
    "departments",
    conn,
    if_exists="append",
    index=False
)


complaints_sql.to_sql(
    "complaints",
    conn,
    if_exists="append",
    index=False
)


capacity_sql.to_sql(
    "capacity",
    conn,
    if_exists="append",
    index=False
)


decision_context_sql.to_sql(
    "decision_context",
    conn,
    if_exists="append",
    index=False
)


predictions_sql.to_sql(
    "predictions",
    conn,
    if_exists="append",
    index=False
)


decisions_sql.to_sql(
    "decisions",
    conn,
    if_exists="append",
    index=False
)


complaint_events_sql.to_sql(
    "complaint_events",
    conn,
    if_exists="append",
    index=False
)


outcomes_sql.to_sql(
    "outcomes",
    conn,
    if_exists="append",
    index=False
)


conn.commit()


# --------------------------------------------------
# BASIC DATABASE AUDIT
# --------------------------------------------------

tables = [
    "customers",
    "departments",
    "complaints",
    "decision_context",
    "predictions",
    "decisions",
    "capacity",
    "complaint_events",
    "outcomes"
]


print(
    "\n===== DATABASE TABLE COUNTS ====="
)

for table in tables:

    count = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM {table}
        """
    ).fetchone()[0]

    print(
        f"{table:22s}: {count}"
    )


# --------------------------------------------------
# Foreign-key validation
# --------------------------------------------------

foreign_key_errors = (
    conn.execute(
        "PRAGMA foreign_key_check;"
    ).fetchall()
)

print(
    "\nForeign key errors:",
    len(foreign_key_errors)
)


# --------------------------------------------------
# Show tables actually stored in SQLite
# --------------------------------------------------

database_tables = pd.read_sql_query(
    """
    SELECT name
    FROM sqlite_master
    WHERE type = 'table'
    ORDER BY name;
    """,
    conn
)

print(
    "\n===== TABLES IN SQLITE DATABASE ====="
)

display(database_tables)


# --------------------------------------------------
# Database file information
# --------------------------------------------------

conn.close()

print(
    "\nDatabase created successfully:"
)

print(db_path)

print(
    "\nDatabase size:"
)

print(
    round(
        os.path.getsize(db_path)
        / (1024 * 1024),
        2
    ),
    "MB"
)# --------------------------------------------------
# STEP 17 — FINAL DATABASE VALIDATION
# --------------------------------------------------

import sqlite3
import pandas as pd

db_path = "customer_complaints.db"

conn = sqlite3.connect(db_path)

conn.execute("PRAGMA foreign_keys = ON;")


print("========================================")
print(" FINAL DATABASE VALIDATION")
print("========================================")


# --------------------------------------------------
# 1. FOREIGN KEY CHECK
# --------------------------------------------------

fk_errors = conn.execute(
    "PRAGMA foreign_key_check;"
).fetchall()

print("\n1. Foreign key errors:")
print(len(fk_errors))


# --------------------------------------------------
# 2. REQUIRED TABLE COUNT
# --------------------------------------------------

required_tables = {
    "customers",
    "departments",
    "complaints",
    "decision_context",
    "predictions",
    "decisions",
    "capacity",
    "complaint_events",
    "outcomes"
}

actual_tables = {
    row[0]
    for row in conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table';
        """
    ).fetchall()
}

missing_tables = (
    required_tables
    - actual_tables
)

print("\n2. Missing required tables:")
print(missing_tables)


# --------------------------------------------------
# 3. COMPLAINT-LEVEL TABLE COUNTS
# --------------------------------------------------

complaint_count = conn.execute(
    """
    SELECT COUNT(*)
    FROM complaints;
    """
).fetchone()[0]

context_count = conn.execute(
    """
    SELECT COUNT(*)
    FROM decision_context;
    """
).fetchone()[0]

decision_count = conn.execute(
    """
    SELECT COUNT(*)
    FROM decisions;
    """
).fetchone()[0]

outcome_count = conn.execute(
    """
    SELECT COUNT(*)
    FROM outcomes;
    """
).fetchone()[0]


print("\n3. Complaint-level table counts:")

print(
    "complaints       :",
    complaint_count
)

print(
    "decision_context :",
    context_count
)

print(
    "decisions        :",
    decision_count
)

print(
    "outcomes         :",
    outcome_count
)


# --------------------------------------------------
# 4. DUPLICATE PRIMARY BUSINESS KEYS
# --------------------------------------------------

duplicate_checks = {}

for table in [
    "complaints",
    "decision_context",
    "decisions",
    "outcomes"
]:

    query = f"""
        SELECT
            COUNT(*) - COUNT(
                DISTINCT complaint_id
            )
        FROM {table};
    """

    duplicate_checks[table] = (
        conn.execute(
            query
        ).fetchone()[0]
    )


print(
    "\n4. Duplicate complaint IDs:"
)

for table, count in (
    duplicate_checks.items()
):

    print(
        f"{table:20s}: {count}"
    )


# --------------------------------------------------
# 5. COMPLAINT TIMESTAMP DATE MATCH
# --------------------------------------------------

date_mismatch = conn.execute(
    """
    SELECT COUNT(*)
    FROM complaints
    WHERE DATE(complaint_timestamp)
          <> date_received;
    """
).fetchone()[0]

print(
    "\n5. Complaint date/timestamp mismatches:"
)

print(date_mismatch)


# --------------------------------------------------
# 6. PREDICTION AFTER INTAKE
# --------------------------------------------------

invalid_prediction_times = conn.execute(
    """
    SELECT COUNT(*)
    FROM decision_context dc
    JOIN complaints c
      ON dc.complaint_id
       = c.complaint_id
    WHERE datetime(dc.prediction_time)
          <= datetime(c.complaint_timestamp);
    """
).fetchone()[0]

print(
    "\n6. Invalid prediction times:"
)

print(invalid_prediction_times)


# --------------------------------------------------
# 7. DECISION AFTER PREDICTION
# --------------------------------------------------

invalid_decision_times = conn.execute(
    """
    SELECT COUNT(*)
    FROM decisions d
    JOIN predictions p
      ON d.prediction_id
       = p.prediction_id
    WHERE datetime(d.decision_time)
          <= datetime(p.prediction_time);
    """
).fetchone()[0]

print(
    "\n7. Invalid decision times:"
)

print(invalid_decision_times)


# --------------------------------------------------
# 8. EVENTS BEFORE DECISION
# --------------------------------------------------

events_before_decision = conn.execute(
    """
    SELECT COUNT(*)
    FROM complaint_events e
    JOIN decisions d
      ON e.complaint_id
       = d.complaint_id
    WHERE datetime(e.event_timestamp)
          < datetime(d.decision_time);
    """
).fetchone()[0]

print(
    "\n8. Events before manager decision:"
)

print(events_before_decision)


# --------------------------------------------------
# 9. EVENT STATUS MAPPING
# --------------------------------------------------

mapping_errors = conn.execute(
    """
    SELECT COUNT(*)
    FROM complaint_events
    WHERE NOT (

       (event_type = 'ASSIGNED'
        AND status = 'Assigned')

       OR

       (event_type = 'REVIEW_STARTED'
        AND status = 'In Review')

       OR

       (event_type = 'INFORMATION_REQUESTED'
        AND status =
            'Waiting for Information')

       OR

       (event_type = 'TRANSFERRED'
        AND status = 'Transferred')

       OR

       (event_type = 'ESCALATED'
        AND status = 'Escalated')

       OR

       (event_type = 'RESOLVED'
        AND status = 'Resolved')

       OR

       (event_type = 'CLOSED'
        AND status = 'Closed')
    );
    """
).fetchone()[0]

print(
    "\n9. Event/status mapping errors:"
)

print(mapping_errors)


# --------------------------------------------------
# 10. EVENT CHRONOLOGY
# Check that event order never goes backwards
# --------------------------------------------------

chronology_errors = conn.execute(
    """
    WITH ordered_events AS (

        SELECT
            complaint_id,
            event_timestamp,

            LAG(event_timestamp)
            OVER (
                PARTITION BY complaint_id
                ORDER BY event_timestamp,
                         event_id
            ) AS previous_timestamp

        FROM complaint_events
    )

    SELECT COUNT(*)
    FROM ordered_events
    WHERE previous_timestamp IS NOT NULL
      AND datetime(event_timestamp)
          < datetime(previous_timestamp);
    """
).fetchone()[0]

print(
    "\n10. Event chronology errors:"
)

print(chronology_errors)


# --------------------------------------------------
# 11. CLOSED MUST HAVE RESOLVED EVENT FIRST
# --------------------------------------------------

closed_without_resolved = conn.execute(
    """
    SELECT COUNT(*)
    FROM outcomes o

    WHERE o.final_status = 'Closed'

    AND NOT EXISTS (

        SELECT 1
        FROM complaint_events e

        WHERE e.complaint_id
              = o.complaint_id

          AND e.event_type
              = 'RESOLVED'
    );
    """
).fetchone()[0]


closed_before_resolved = conn.execute(
    """
    SELECT COUNT(*)

    FROM outcomes o

    JOIN complaint_events r
      ON o.complaint_id
       = r.complaint_id
     AND r.event_type
       = 'RESOLVED'

    JOIN complaint_events c
      ON o.complaint_id
       = c.complaint_id
     AND c.event_type
       = 'CLOSED'

    WHERE datetime(c.event_timestamp)
          < datetime(r.event_timestamp);
    """
).fetchone()[0]


print(
    "\n11. Closed without resolved event:"
)

print(closed_without_resolved)

print(
    "Closed before resolved event:"
)

print(closed_before_resolved)


# --------------------------------------------------
# 12. TRUE ESCALATION MUST HAVE ESCALATED EVENT
# --------------------------------------------------

true_without_event = conn.execute(
    """
    SELECT COUNT(*)

    FROM outcomes o

    WHERE o.actual_escalation = 1

    AND NOT EXISTS (

        SELECT 1

        FROM complaint_events e

        WHERE e.complaint_id
              = o.complaint_id

          AND e.event_type
              = 'ESCALATED'
    );
    """
).fetchone()[0]

print(
    "\n12. True escalation without event:"
)

print(true_without_event)


# --------------------------------------------------
# 13. FALSE FINAL ESCALATION MUST NOT
# HAVE ESCALATED EVENT
# --------------------------------------------------

false_with_event = conn.execute(
    """
    SELECT COUNT(*)

    FROM outcomes o

    WHERE o.actual_escalation = 0

    AND EXISTS (

        SELECT 1

        FROM complaint_events e

        WHERE e.complaint_id
              = o.complaint_id

          AND e.event_type
              = 'ESCALATED'
    );
    """
).fetchone()[0]

print(
    "\n13. False escalation with escalation event:"
)

print(false_with_event)


# --------------------------------------------------
# 14. FINAL DEPARTMENT MATCHES RESOLVED EVENT
# --------------------------------------------------

final_department_errors = conn.execute(
    """
    SELECT COUNT(*)

    FROM outcomes o

    JOIN complaint_events e
      ON o.complaint_id
       = e.complaint_id

    WHERE e.event_type = 'RESOLVED'

      AND o.final_department_id
          <> e.department_id;
    """
).fetchone()[0]

print(
    "\n14. Final department mismatches:"
)

print(final_department_errors)


# --------------------------------------------------
# 15. OUTCOME STATUS CONSISTENCY
# --------------------------------------------------

pending_errors = conn.execute(
    """
    SELECT COUNT(*)
    FROM outcomes

    WHERE final_status = 'Pending'

    AND (
        resolution_time IS NOT NULL
        OR date_closed IS NOT NULL
    );
    """
).fetchone()[0]


resolved_errors = conn.execute(
    """
    SELECT COUNT(*)
    FROM outcomes

    WHERE final_status = 'Resolved'

    AND (
        resolution_time IS NULL
        OR date_closed IS NOT NULL
    );
    """
).fetchone()[0]


closed_errors = conn.execute(
    """
    SELECT COUNT(*)
    FROM outcomes

    WHERE final_status = 'Closed'

    AND (
        resolution_time IS NULL
        OR date_closed IS NULL
    );
    """
).fetchone()[0]


print(
    "\n15. Outcome status errors:"
)

print(
    "Pending errors :",
    pending_errors
)

print(
    "Resolved errors:",
    resolved_errors
)

print(
    "Closed errors  :",
    closed_errors
)


# --------------------------------------------------
# 16. RESOLUTION TIME RECOMPUTATION CHECK
# --------------------------------------------------

resolution_time_errors = conn.execute(
    """
    SELECT COUNT(*)

    FROM outcomes o

    JOIN complaints c
      ON o.complaint_id
       = c.complaint_id

    JOIN complaint_events e
      ON o.complaint_id
       = e.complaint_id
     AND e.event_type
       = 'RESOLVED'

    WHERE ABS(

        o.resolution_time

        -

        (
            julianday(e.event_timestamp)
            -
            julianday(c.complaint_timestamp)
        ) * 24

    ) > 0.02;
    """
).fetchone()[0]


print(
    "\n16. Resolution-time calculation errors:"
)

print(resolution_time_errors)


# --------------------------------------------------
# 17. DECISION CONTEXT CAPACITY LOOKUP
# Verify a historical capacity snapshot exists
# at or before every prediction
# --------------------------------------------------

missing_historical_capacity = conn.execute(
    """
    SELECT COUNT(*)

    FROM decision_context dc

    WHERE NOT EXISTS (

        SELECT 1

        FROM capacity c

        WHERE c.department_id
              = dc.context_department_id

          AND datetime(
                c.capacity_timestamp
              )
              <=
              datetime(
                dc.prediction_time
              )
    );
    """
).fetchone()[0]


print(
    "\n17. Missing historical capacity snapshots:"
)

print(
    missing_historical_capacity
)


# --------------------------------------------------
# 18. RANGE CHECKS
# --------------------------------------------------

invalid_probabilities = conn.execute(
    """
    SELECT COUNT(*)

    FROM predictions

    WHERE urgency_probability
          NOT BETWEEN 0 AND 1

       OR escalation_probability
          NOT BETWEEN 0 AND 1;
    """
).fetchone()[0]


negative_values = conn.execute(
    """
    SELECT

        (
            SELECT COUNT(*)
            FROM decision_context
            WHERE prior_complaint_count < 0
               OR current_workload < 0
               OR available_capacity < 0
               OR known_financial_impact < 0
        )

        +

        (
            SELECT COUNT(*)
            FROM capacity
            WHERE available_staff < 0
               OR current_open_cases < 0
               OR available_capacity < 0
        )

        +

        (
            SELECT COUNT(*)
            FROM outcomes
            WHERE resolution_time < 0
               OR handling_cost < 0
        );
    """
).fetchone()[0]


print(
    "\n18. Invalid probabilities:"
)

print(invalid_probabilities)

print(
    "Negative-value violations:"
)

print(negative_values)


# --------------------------------------------------
# FINAL PASS / FAIL
# --------------------------------------------------

critical_errors = (

    len(fk_errors)

    + len(missing_tables)

    + sum(
        duplicate_checks.values()
    )

    + date_mismatch

    + invalid_prediction_times

    + invalid_decision_times

    + events_before_decision

    + mapping_errors

    + chronology_errors

    + closed_without_resolved

    + closed_before_resolved

    + true_without_event

    + false_with_event

    + final_department_errors

    + pending_errors

    + resolved_errors

    + closed_errors

    + resolution_time_errors

    + missing_historical_capacity

    + invalid_probabilities

    + negative_values
)


print(
    "\n========================================"
)

if critical_errors == 0:

    print(
        "FINAL RESULT: DATABASE PASSED"
    )

    print(
        "All critical validation checks = 0"
    )

else:

    print(
        "FINAL RESULT: REVIEW REQUIRED"
    )

    print(
        "Total critical errors:",
        critical_errors
    )

print(
    "========================================"
)


# --------------------------------------------------
# Useful final summary
# --------------------------------------------------

summary = pd.read_sql_query(
    """
    SELECT

        COUNT(*) AS total_complaints,

        SUM(
            CASE
            WHEN final_status = 'Closed'
            THEN 1 ELSE 0 END
        ) AS closed,

        SUM(
            CASE
            WHEN final_status = 'Resolved'
            THEN 1 ELSE 0 END
        ) AS resolved,

        SUM(
            CASE
            WHEN final_status = 'Pending'
            THEN 1 ELSE 0 END
        ) AS pending,

        SUM(
            CASE
            WHEN actual_escalation = 1
            THEN 1 ELSE 0 END
        ) AS escalated,

        ROUND(
            AVG(resolution_time),
            2
        ) AS avg_resolution_hours,

        ROUND(
            AVG(handling_cost),
            2
        ) AS avg_handling_cost

    FROM outcomes;
    """,
    conn
)

print(
    "\n===== FINAL DATABASE SUMMARY ====="
)

display(summary)

conn.close()# --------------------------------------------------
# STEP 18 — PACKAGE FINAL DATABASE FILES
# --------------------------------------------------

import sqlite3
import os

db_path = "customer_complaints.db"


# --------------------------------------------------
# 1. Export database schema
# --------------------------------------------------

conn = sqlite3.connect(db_path)

schema_rows = conn.execute(
    """
    SELECT sql
    FROM sqlite_master
    WHERE sql IS NOT NULL
    ORDER BY
        CASE type
            WHEN 'table' THEN 1
            WHEN 'index' THEN 2
            ELSE 3
        END,
        name;
    """
).fetchall()


# sqlite_master stores CREATE statements without trailing semicolons.
# Add a terminator to each statement so this export can be executed as a
# standalone SQL script against a fresh SQLite database.
schema_text = "\n\n".join(
    row[0].rstrip().rstrip(";") + ";"
    for row in schema_rows
    if row[0] is not None
)

with open(
    "database_schema.sql",
    "w",
    encoding="utf-8"
) as file:

    file.write(schema_text)

    file.write(
        "\n"
    )


# --------------------------------------------------
# 2. Create validation summary
# --------------------------------------------------

# Build the summary from the actual database and the validation results above.
# Keep grouped checks explicit while ensuring every displayed value is dynamic.
summary_conn = sqlite3.connect(db_path)
summary_conn.execute("PRAGMA foreign_keys = ON;")

row_counts = {
    "Customers": summary_conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0],
    "Departments": summary_conn.execute("SELECT COUNT(*) FROM departments").fetchone()[0],
    "Complaints": summary_conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0],
    "Decision Context Records": summary_conn.execute("SELECT COUNT(*) FROM decision_context").fetchone()[0],
    "Predictions": summary_conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0],
    "Manager Decisions": summary_conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0],
    "Capacity Records": summary_conn.execute("SELECT COUNT(*) FROM capacity").fetchone()[0],
    "Complaint Events": summary_conn.execute("SELECT COUNT(*) FROM complaint_events").fetchone()[0],
    "Outcomes": summary_conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0],
}

outcome_summary = summary_conn.execute(
    """
    SELECT
        SUM(CASE WHEN final_status = 'Closed' THEN 1 ELSE 0 END),
        SUM(CASE WHEN final_status = 'Resolved' THEN 1 ELSE 0 END),
        SUM(CASE WHEN final_status = 'Pending' THEN 1 ELSE 0 END),
        SUM(CASE WHEN actual_escalation = 1 THEN 1 ELSE 0 END),
        ROUND(AVG(resolution_time), 2),
        ROUND(AVG(handling_cost), 2)
    FROM outcomes
    """
).fetchone()

validation_checks = {
    "Foreign key errors": len(fk_errors),
    "Missing required tables": len(missing_tables),
    "Duplicate complaint IDs": sum(duplicate_checks.values()),
    "Complaint date/timestamp mismatches": date_mismatch,
    "Invalid prediction times": invalid_prediction_times,
    "Invalid decision times": invalid_decision_times,
    "Events before manager decision": events_before_decision,
    "Event/status mapping errors": mapping_errors,
    "Event chronology errors": chronology_errors,
    "Closed without resolved event": closed_without_resolved,
    "Closed before resolved event": closed_before_resolved,
    "True escalation without event": true_without_event,
    "False escalation with escalation event": false_with_event,
    "Final department mismatches": final_department_errors,
    "Outcome status errors": pending_errors + resolved_errors + closed_errors,
    "Resolution-time calculation errors": resolution_time_errors,
    "Missing historical capacity snapshots": missing_historical_capacity,
    "Invalid probabilities": invalid_probabilities,
    "Negative-value violations": negative_values,
}

validation_result = "DATABASE PASSED" if critical_errors == 0 else "REVIEW REQUIRED"

summary_lines = [
    "CUSTOMER COMPLAINT DECISION-SUPPORT DATABASE",
    "FINAL VALIDATION SUMMARY",
    "",
    "Database file:",
    os.path.abspath(db_path),
    "",
    "DATASET SUMMARY",
    "---------------",
]
summary_lines.extend(f"{label}: {value:,}" for label, value in row_counts.items())
summary_lines.extend([
    "",
    "OUTCOME SUMMARY",
    "---------------",
    f"Closed: {outcome_summary[0]:,}",
    f"Resolved: {outcome_summary[1]:,}",
    f"Pending: {outcome_summary[2]:,}",
    f"Escalated: {outcome_summary[3]:,}",
    "",
    f"Average Resolution Time: {outcome_summary[4]} hours",
    f"Average Handling Cost: ${outcome_summary[5]:,.2f}",
    "",
    "DATABASE VALIDATION",
    "-------------------",
])
summary_lines.extend(f"{label}: {value:,}" for label, value in validation_checks.items())
summary_lines.extend([
    "",
    "FINAL RESULT",
    "------------",
    validation_result,
    f"Total critical errors: {critical_errors:,}",
    f"Named validation checks reported: {len(validation_checks)}",
    "",
    "DATA ORIGIN",
    "-----------",
    "Core complaint information was sourced from the CFPB consumer complaint dataset.",
    "Synthetic operational variables were generated for the academic decision-support simulation, including:",
    "- synthetic customer identifiers and customer types",
    "- complaint intake timestamps",
    "- financial impact",
    "- department staffing and workload",
    "- model predictions",
    "- manager decisions",
    "- complaint complexity and hidden risk",
    "- workflow events",
    "- escalation outcomes",
    "- resolution durations",
    "- internal handling costs",
    "",
    "The oracle ground-truth labels are generated in memory for simulation and evaluation;",
    "they are not persisted as a separate database table.",
    "Missing CFPB complaint narratives were preserved as NULL rather than artificially generated.",
])
validation_summary = "\n".join(summary_lines)
summary_conn.close()


with open(
    "database_validation_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        validation_summary.strip()
    )


conn.close()


# --------------------------------------------------
# 3. Confirm files
# --------------------------------------------------

print("===== FINAL FILES CREATED =====")

final_files = [
    "customer_complaints.db",
    "database_schema.sql",
    "database_validation_summary.txt"
]

for filename in final_files:

    size_mb = (
        os.path.getsize(filename)
        / (1024 * 1024)
    )

    print(
        f"{filename:35s} "
        f"{size_mb:.2f} MB"
    )# --------------------------------------------------
# EXPORT ALL DATABASE TABLES + FINAL FILES
# INTO ONE ZIP FILE
# --------------------------------------------------

import sqlite3
import pandas as pd
import os
import zipfile

db_path = "customer_complaints.db"

# Create folder
export_folder = "customer_complaints_project"

os.makedirs(
    export_folder,
    exist_ok=True
)

# --------------------------------------------------
# Export all SQLite tables as CSV
# --------------------------------------------------

conn = sqlite3.connect(db_path)

tables = [
    "customers",
    "departments",
    "complaints",
    "decision_context",
    "predictions",
    "decisions",
    "capacity",
    "complaint_events",
    "outcomes"
]

for table in tables:

    df_export = pd.read_sql_query(
        f"SELECT * FROM {table}",
        conn
    )

    csv_path = os.path.join(
        export_folder,
        f"{table}.csv"
    )

    df_export.to_csv(
        csv_path,
        index=False
    )

    print(
        f"Exported {table}:",
        len(df_export),
        "rows"
    )

conn.close()


# --------------------------------------------------
# Add main database files
# --------------------------------------------------

main_files = [
    "customer_complaints.db",
    "database_schema.sql",
    "database_validation_summary.txt"
]

for filename in main_files:

    if os.path.exists(filename):

        destination = os.path.join(
            export_folder,
            filename
        )

        with open(
            filename,
            "rb"
        ) as source:

            with open(
                destination,
                "wb"
            ) as target:

                target.write(
                    source.read()
                )


# --------------------------------------------------
# Create ZIP
# --------------------------------------------------

zip_filename = (
    "customer_complaints_project_complete.zip"
)

with zipfile.ZipFile(
    zip_filename,
    "w",
    zipfile.ZIP_DEFLATED
) as zip_file:

    for root, dirs, files_in_folder in os.walk(
        export_folder
    ):

        for file in files_in_folder:

            file_path = os.path.join(
                root,
                file
            )

            zip_file.write(
                file_path,
                arcname=os.path.relpath(
                    file_path,
                    export_folder
                )
            )


print("\n===== COMPLETE EXPORT CREATED =====")

print(zip_filename)

print(
    "Size:",
    round(
        os.path.getsize(zip_filename)
        / (1024 * 1024),
        2
    ),
    "MB"
)