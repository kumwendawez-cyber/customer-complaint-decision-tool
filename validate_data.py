"""
Independent validation script for the Customer Complaint Decision-Support Tool.

This script reads an existing SQLite database. It does not generate, modify,
or overwrite the database.

Run from the project root:
    python validate_data.py
Or specify a database path:
    python validate_data.py path/to/customer_complaints.db
"""

import argparse
import sqlite3
import sys
from pathlib import Path


REQUIRED_TABLES = {
    "customers",
    "departments",
    "complaints",
    "decision_context",
    "predictions",
    "decisions",
    "capacity",
    "complaint_events",
    "outcomes",
}

COMPLAINT_LEVEL_TABLES = [
    "decision_context",
    "predictions",
    "decisions",
    "outcomes",
]

EXPECTED_COMPLAINT_EVENT_STATUS = {
    "ASSIGNED": "Assigned",
    "REVIEW_STARTED": "In Review",
    "INFORMATION_REQUESTED": "Waiting for Information",
    "TRANSFERRED": "Transferred",
    "ESCALATED": "Escalated",
    "RESOLVED": "Resolved",
    "CLOSED": "Closed",
}


def table_names(conn):
    return {
        row[0]
        for row in conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table';
            """
        ).fetchall()
    }


def run_check(results, name, issue_count, detail=""):
    passed = issue_count == 0
    results.append((name, passed, int(issue_count), detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}: {issue_count} issue(s)")
    if detail:
        print(f"       {detail}")


def count_query(conn, sql):
    return int(conn.execute(sql).fetchone()[0])


def validate_database(db_path):
    if not db_path.exists():
        print(f"ERROR: Database file not found: {db_path}")
        print("Place customer_complaints.db beside validate_data.py,")
        print("or provide its path as a command-line argument.")
        return 2

    results = []

    # Open in read-only mode so validation cannot change the database.
    connection_uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(connection_uri, uri=True)

    try:
        print("=" * 68)
        print("CUSTOMER COMPLAINT DATABASE VALIDATION")
        print("=" * 68)
        print(f"Database: {db_path.resolve()}")
        print("Mode: read-only")
        print()

        actual_tables = table_names(conn)
        missing_tables = REQUIRED_TABLES - actual_tables

        run_check(
            results,
            "Required database tables",
            len(missing_tables),
            f"Missing tables: {sorted(missing_tables)}"
            if missing_tables else "All 9 required tables are present.",
        )

        if missing_tables:
            print("\nValidation stopped because required tables are missing.")
            return 1

        # 1. SQLite integrity and foreign-key integrity.
        integrity_rows = conn.execute("PRAGMA integrity_check;").fetchall()
        integrity_errors = [
            row[0] for row in integrity_rows if row[0] != "ok"
        ]
        run_check(
            results,
            "SQLite database integrity",
            len(integrity_errors),
            "; ".join(integrity_errors[:5]) if integrity_errors else "SQLite integrity check returned ok.",
        )

        fk_errors = conn.execute("PRAGMA foreign_key_check;").fetchall()
        run_check(
            results,
            "Foreign-key integrity",
            len(fk_errors),
            f"First violations: {fk_errors[:5]}" if fk_errors else "No orphan foreign-key records found.",
        )

        # 2. Primary-key uniqueness and null checks for every table.
        pk_issues = 0
        pk_details = []
        for table in sorted(REQUIRED_TABLES):
            info = conn.execute(f'PRAGMA table_info("{table}");').fetchall()
            pk_columns = [row[1] for row in info if row[5] > 0]

            if not pk_columns:
                pk_issues += 1
                pk_details.append(f"{table}: no declared primary key")
                continue

            # Supports single-column and composite primary keys.
            columns_sql = ", ".join(f'"{col}"' for col in pk_columns)
            duplicate_count = count_query(
                conn,
                f"""
                SELECT COUNT(*) FROM (
                    SELECT {columns_sql}
                    FROM "{table}"
                    GROUP BY {columns_sql}
                    HAVING COUNT(*) > 1
                );
                """,
            )
            null_condition = " OR ".join(
                f'"{col}" IS NULL' for col in pk_columns
            )
            null_count = count_query(
                conn,
                f'SELECT COUNT(*) FROM "{table}" WHERE {null_condition};',
            )
            if duplicate_count or null_count:
                pk_issues += duplicate_count + null_count
                pk_details.append(
                    f"{table}: {duplicate_count} duplicate key groups, "
                    f"{null_count} null key rows"
                )

        run_check(
            results,
            "Primary-key uniqueness and completeness",
            pk_issues,
            "; ".join(pk_details) if pk_details else "All declared primary keys are unique and non-null.",
        )

        # 3. Complaint IDs must be unique in complaint-level tables.
        duplicate_complaint_ids = 0
        duplicate_details = []
        for table in ["complaints"] + COMPLAINT_LEVEL_TABLES:
            duplicates = count_query(
                conn,
                f"""
                SELECT COUNT(*) FROM (
                    SELECT complaint_id
                    FROM "{table}"
                    GROUP BY complaint_id
                    HAVING complaint_id IS NULL OR COUNT(*) > 1
                );
                """,
            )
            duplicate_complaint_ids += duplicates
            if duplicates:
                duplicate_details.append(f"{table}: {duplicates}")

        run_check(
            results,
            "One record per complaint in complaint-level tables",
            duplicate_complaint_ids,
            "; ".join(duplicate_details) if duplicate_details else "No null or repeated complaint IDs found.",
        )

        # 4. Reconcile row counts across complaint-level tables.
        complaint_count = count_query(conn, "SELECT COUNT(*) FROM complaints;")
        count_mismatches = 0
        count_details = []
        for table in COMPLAINT_LEVEL_TABLES:
            table_count = count_query(conn, f'SELECT COUNT(*) FROM "{table}";')
            if table_count != complaint_count:
                count_mismatches += 1
                count_details.append(f"{table}={table_count}, complaints={complaint_count}")

        run_check(
            results,
            "Complaint-level table row-count reconciliation",
            count_mismatches,
            "; ".join(count_details) if count_details else f"All complaint-level tables contain {complaint_count:,} rows.",
        )

        # 5. Every complaint must have at least one event.
        complaints_without_events = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM complaints c
            WHERE NOT EXISTS (
                SELECT 1
                FROM complaint_events e
                WHERE e.complaint_id = c.complaint_id
            );
            """,
        )
        run_check(
            results,
            "Event-log coverage",
            complaints_without_events,
            "Every complaint has at least one event." if not complaints_without_events
            else "Some complaints have no event record.",
        )

        # 6. Intake date and timestamp must agree.
        date_mismatches = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM complaints
            WHERE DATE(complaint_timestamp) IS NULL
               OR DATE(complaint_timestamp) <> date_received;
            """,
        )
        run_check(results, "Complaint date/timestamp consistency", date_mismatches)

        # 7. Prediction must occur after complaint intake.
        invalid_prediction_times = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM decision_context dc
            JOIN complaints c ON c.complaint_id = dc.complaint_id
            WHERE datetime(dc.prediction_time) IS NULL
               OR datetime(c.complaint_timestamp) IS NULL
               OR datetime(dc.prediction_time) <= datetime(c.complaint_timestamp);
            """,
        )
        run_check(results, "Prediction occurs after complaint intake", invalid_prediction_times)

        # 8. Prediction timestamps in context and prediction tables must agree.
        prediction_time_mismatches = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM decision_context dc
            JOIN predictions p ON p.complaint_id = dc.complaint_id
            WHERE datetime(dc.prediction_time) <> datetime(p.prediction_time);
            """,
        )
        run_check(results, "Prediction-time reconciliation", prediction_time_mismatches)

        # 9. Manager decision must occur after prediction.
        invalid_decision_times = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM decisions d
            JOIN predictions p ON p.prediction_id = d.prediction_id
            WHERE datetime(d.decision_time) IS NULL
               OR datetime(p.prediction_time) IS NULL
               OR datetime(d.decision_time) <= datetime(p.prediction_time);
            """,
        )
        run_check(results, "Manager decision occurs after prediction", invalid_decision_times)

        # 10. Events must not occur before the manager decision.
        events_before_decision = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM complaint_events e
            JOIN decisions d ON d.complaint_id = e.complaint_id
            WHERE datetime(e.event_timestamp) < datetime(d.decision_time);
            """,
        )
        run_check(results, "Events occur after manager decision", events_before_decision)

        # 11. Event type/status pairs must follow the defined mapping.
        mapping_cases = " OR ".join(
            f"(event_type = '{event}' AND status <> '{status}')"
            for event, status in EXPECTED_COMPLAINT_EVENT_STATUS.items()
        )
        allowed_events = ", ".join(
            f"'{event}'" for event in EXPECTED_COMPLAINT_EVENT_STATUS
        )
        mapping_errors = count_query(
            conn,
            f"""
            SELECT COUNT(*)
            FROM complaint_events
            WHERE event_type NOT IN ({allowed_events})
               OR ({mapping_cases})
            """,
        )
        run_check(results, "Event type and status mapping", mapping_errors)

        # 12. Event timestamps must be chronological within each complaint.
        chronology_errors = count_query(
            conn,
            """
            WITH ordered_events AS (
                SELECT
                    complaint_id,
                    event_timestamp,
                    LAG(event_timestamp) OVER (
                        PARTITION BY complaint_id
                        ORDER BY datetime(event_timestamp), event_id
                    ) AS previous_timestamp
                FROM complaint_events
            )
            SELECT COUNT(*)
            FROM ordered_events
            WHERE previous_timestamp IS NOT NULL
              AND datetime(event_timestamp) < datetime(previous_timestamp);
            """,
        )
        run_check(results, "Event chronology", chronology_errors)

        # 13. Closed complaints must have a resolved event first.
        closed_without_resolved = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM outcomes o
            WHERE o.final_status = 'Closed'
              AND NOT EXISTS (
                  SELECT 1
                  FROM complaint_events e
                  WHERE e.complaint_id = o.complaint_id
                    AND e.event_type = 'RESOLVED'
              );
            """,
        )
        closed_before_resolved = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM complaint_events r
            JOIN complaint_events c
              ON c.complaint_id = r.complaint_id
            JOIN outcomes o
              ON o.complaint_id = r.complaint_id
            WHERE r.event_type = 'RESOLVED'
              AND c.event_type = 'CLOSED'
              AND o.final_status = 'Closed'
              AND datetime(c.event_timestamp) < datetime(r.event_timestamp);
            """,
        )
        run_check(
            results,
            "Closed outcome has a prior resolved event",
            closed_without_resolved + closed_before_resolved,
            f"Closed without resolved event: {closed_without_resolved}; closed before resolved: {closed_before_resolved}.",
        )

        # 14. Escalation outcomes must agree with escalation events.
        true_without_event = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM outcomes o
            WHERE o.actual_escalation = 1
              AND NOT EXISTS (
                  SELECT 1
                  FROM complaint_events e
                  WHERE e.complaint_id = o.complaint_id
                    AND e.event_type = 'ESCALATED'
              );
            """,
        )
        false_with_event = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM outcomes o
            WHERE o.actual_escalation = 0
              AND EXISTS (
                  SELECT 1
                  FROM complaint_events e
                  WHERE e.complaint_id = o.complaint_id
                    AND e.event_type = 'ESCALATED'
              );
            """,
        )
        run_check(
            results,
            "Escalation outcome and event consistency",
            true_without_event + false_with_event,
            f"Escalated outcome without event: {true_without_event}; non-escalated outcome with event: {false_with_event}.",
        )

        # 15. Final department must agree with the department on RESOLVED event.
        final_department_errors = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM outcomes o
            JOIN complaint_events e
              ON e.complaint_id = o.complaint_id
             AND e.event_type = 'RESOLVED'
            WHERE o.final_department_id IS NOT e.department_id;
            """,
        )
        run_check(results, "Final department and resolved-event consistency", final_department_errors)

        # 16. Outcome status and missing-value rules.
        outcome_status_errors = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM outcomes
            WHERE
                (final_status = 'Pending'
                    AND (resolution_time IS NOT NULL OR date_closed IS NOT NULL))
                OR
                (final_status = 'Resolved'
                    AND (resolution_time IS NULL OR date_closed IS NOT NULL))
                OR
                (final_status = 'Closed'
                    AND (resolution_time IS NULL OR date_closed IS NULL))
                OR
                final_status NOT IN ('Pending', 'Resolved', 'Closed');
            """,
        )
        run_check(results, "Outcome status and null-value consistency", outcome_status_errors)

        # 17. Recalculate resolution time from intake to RESOLVED event.
        resolution_time_errors = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM outcomes o
            JOIN complaints c ON c.complaint_id = o.complaint_id
            JOIN complaint_events e
              ON e.complaint_id = o.complaint_id
             AND e.event_type = 'RESOLVED'
            WHERE o.resolution_time IS NULL
               OR ABS(
                    o.resolution_time -
                    ((julianday(e.event_timestamp) -
                      julianday(c.complaint_timestamp)) * 24)
                  ) > 0.02;
            """,
        )
        run_check(results, "Resolution-time recalculation", resolution_time_errors)

        # 18. A historical capacity snapshot must exist at or before prediction.
        missing_capacity_snapshots = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM decision_context dc
            WHERE NOT EXISTS (
                SELECT 1
                FROM capacity c
                WHERE c.department_id = dc.context_department_id
                  AND datetime(c.capacity_timestamp) <= datetime(dc.prediction_time)
            );
            """,
        )
        run_check(results, "Historical capacity snapshot availability", missing_capacity_snapshots)

        # 19. Available capacity formula:
        # max(0, (available_staff * 8) - current_open_cases).
        capacity_formula_errors = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM capacity
            WHERE available_capacity <>
                MAX(0, (available_staff * 8) - current_open_cases);
            """,
        )
        run_check(results, "Available-capacity formula", capacity_formula_errors)

        # 20. Current workload formula, using the latest capacity snapshot
        # for the context department at or before prediction time.
        workload_formula_errors = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM decision_context dc
            JOIN capacity c
              ON c.capacity_id = (
                  SELECT c2.capacity_id
                  FROM capacity c2
                  WHERE c2.department_id = dc.context_department_id
                    AND datetime(c2.capacity_timestamp) <= datetime(dc.prediction_time)
                  ORDER BY datetime(c2.capacity_timestamp) DESC
                  LIMIT 1
              )
            WHERE ABS(
                dc.current_workload -
                ROUND(
                    CAST(c.current_open_cases AS REAL) /
                    MAX(c.available_staff, 1),
                    2
                )
            ) > 0.01;
            """,
        )
        run_check(results, "Current-workload formula", workload_formula_errors)

        # 21. Prediction probabilities must be between zero and one.
        invalid_probabilities = count_query(
            conn,
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE urgency_probability IS NULL
               OR escalation_probability IS NULL
               OR urgency_probability NOT BETWEEN 0 AND 1
               OR escalation_probability NOT BETWEEN 0 AND 1;
            """,
        )
        run_check(results, "Prediction probability ranges", invalid_probabilities)

        # 22. Nonnegative numeric values. Null financial impact is allowed.
        negative_values = count_query(
            conn,
            """
            SELECT
                (SELECT COUNT(*) FROM decision_context
                 WHERE prior_complaint_count < 0
                    OR current_workload < 0
                    OR available_capacity < 0
                    OR known_financial_impact < 0)
              +
                (SELECT COUNT(*) FROM capacity
                 WHERE available_staff < 0
                    OR current_open_cases < 0
                    OR available_capacity < 0)
              +
                (SELECT COUNT(*) FROM outcomes
                 WHERE resolution_time < 0
                    OR handling_cost < 0);
            """,
        )
        run_check(results, "Nonnegative operational and outcome values", negative_values)

        # 23. Report known missingness without treating documented nulls as errors.
        narrative_nulls = count_query(
            conn,
            "SELECT COUNT(*) FROM complaints WHERE complaint_narrative IS NULL;",
        )
        sub_issue_nulls = count_query(
            conn,
            "SELECT COUNT(*) FROM complaints WHERE sub_issue IS NULL;",
        )
        financial_impact_nulls = count_query(
            conn,
            "SELECT COUNT(*) FROM decision_context WHERE known_financial_impact IS NULL;",
        )
        pending_outcomes = count_query(
            conn,
            "SELECT COUNT(*) FROM outcomes WHERE final_status = 'Pending';",
        )

        print("\nDOCUMENTED MISSINGNESS (reported, not automatically failed)")
        print(f"Complaint narratives missing: {narrative_nulls:,}")
        print(f"Sub-issues missing: {sub_issue_nulls:,}")
        print(f"Known financial impact missing: {financial_impact_nulls:,}")
        print(f"Pending outcomes: {pending_outcomes:,}")

        # Final summary.
        passed_count = sum(1 for _, passed, _, _ in results if passed)
        failed_count = len(results) - passed_count

        print("\n" + "=" * 68)
        print("VALIDATION SUMMARY")
        print("=" * 68)
        print(f"Complaints in database: {complaint_count:,}")
        print(f"Checks passed: {passed_count}")
        print(f"Checks failed: {failed_count}")

        if failed_count == 0:
            print("FINAL RESULT: PASS")
            print("All implemented structural and logical checks passed.")
            return 0

        print("FINAL RESULT: REVIEW REQUIRED")
        print("Please review the failed checks before using this database.")
        return 1

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run independent quality checks on the synthetic complaints SQLite database."
    )
    parser.add_argument(
        "database",
        nargs="?",
        default="customer_complaints.db",
        help="Path to customer_complaints.db (default: beside the script/current folder).",
    )
    args = parser.parse_args()

    return validate_database(Path(args.database))


if __name__ == "__main__":
    sys.exit(main())
