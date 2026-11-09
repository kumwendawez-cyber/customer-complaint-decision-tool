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

CREATE TABLE departments (

    department_id TEXT PRIMARY KEY,

    department_name TEXT
        NOT NULL
        UNIQUE
);

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

CREATE INDEX idx_capacity_department_time
ON capacity (
    department_id,
    capacity_timestamp
);

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

CREATE INDEX idx_decisions_department
ON decisions (
    selected_department_id
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

CREATE INDEX idx_predictions_complaint_time
ON predictions (
    complaint_id,
    prediction_time
);

CREATE INDEX idx_predictions_department
ON predictions (
    recommended_department_id
);

CREATE INDEX idx_predictions_model
ON predictions (
    model_version
);
