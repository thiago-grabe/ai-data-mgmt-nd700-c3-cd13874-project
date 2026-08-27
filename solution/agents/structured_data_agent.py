from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    selection_rate,
)
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine, inspect

# Fairness thresholds, named rather than buried as magic numbers at the use site.
DEMOGRAPHIC_PARITY_THRESHOLD = 0.10
MEAN_RATIO_THRESHOLD = 0.80


class StructuredDataAgent:
    """Natural-language questions against PostgreSQL, audited for bias in flight.

    Connects to exactly one data source: the PostgreSQL demographics and housing
    database. It holds no MongoDB, Blob Storage, or Content Safety credential —
    source isolation is a property of this class's constructor signature, not a
    convention.
    """

    # ============================================
    # INIT
    # ============================================
    def __init__(
        self,
        azure_endpoint,
        api_key,
        deployment,
        db_config,
        api_version="2024-12-01-preview",
        temperature=0.7,
    ):
        encoded_password = quote_plus(db_config["password"])
        self.db_uri = (
            # +psycopg selects the psycopg 3 driver. Bare postgresql:// would make
            # SQLAlchemy reach for psycopg2, which requirements.txt does not install.
            # sslmode=require is mandatory: Azure Flexible Server runs with
            # require_secure_transport ON and rejects plaintext connections.
            f"postgresql+psycopg://{db_config['user']}:{encoded_password}"
            f"@{db_config['host']}:{db_config.get('port', 5432)}"
            f"/{db_config['dbname']}?sslmode=require"
        )

        self.model = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            azure_deployment=deployment,
            api_version=api_version,
            temperature=temperature,
        )

        self.db = SQLDatabase.from_uri(self.db_uri)
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.model)
        tools = toolkit.get_tools()

        self.agent_executor = create_react_agent(self.model, tools)

        # Populated by the most recent __auto_bias_check so ask() can act on the
        # findings rather than merely print them.
        self.last_bias_findings = []

    # ============================================
    # PUBLIC METHOD
    # ============================================
    def ask(self, question, verbose=False, run_bias_audit=True):

        response_text = ""

        for chunk in self.agent_executor.stream(
            {"messages": [("user", question)]},
            stream_mode="values",
        ):
            msg = chunk["messages"][-1]
            response_text = msg.content or ""
            if verbose:
                print(f"[StructuredDataAgent] {msg.type}: {str(response_text)[:200]}")

        self.last_bias_findings = []

        if run_bias_audit:
            print("\n" + "=" * 80)
            print("Using Strucutred Data stored in PostgresQL database to answer question")
            print("Performing a Fairness audit before providing answer")
            print("=" * 80)
            self.__auto_bias_check(self.db_uri)

        # Actionable safeguard: a bias audit that only prints is a log line, not a
        # control. Where the audit found a disparity, the caveat travels with the
        # answer to whoever reads it.
        answer = response_text
        if self.last_bias_findings:
            caveat = self.__format_bias_caveat(self.last_bias_findings)
            answer = f"{caveat}\n\n{response_text}"

        return {
            "question": question,
            "response": answer,
            "raw_response": response_text,
            "bias_findings": self.last_bias_findings,
        }

    # ============================================
    # TABLE DISCOVERY
    # ============================================
    def __discover_tables(self, db_uri):

        engine = create_engine(db_uri)
        inspector = inspect(engine)

        tables = []
        for schema in inspector.get_schema_names():
            if schema in ["information_schema", "pg_catalog"]:
                continue

            for table in inspector.get_table_names(schema=schema):
                full_name = f"{schema}.{table}" if schema != "public" else table
                tables.append(full_name)

        return tables, engine

    # ============================================
    # DEMOGRAPHIC DETECTION
    # ============================================
    def __find_demographic_columns(self, columns):

        # These keywords are deliberately narrow. The schools and hospitals tables
        # carry columns named male_students, white_patients, hispanic_students and
        # so on, which are aggregate COUNTS, not per-row group labels. Matching on
        # "male" or "white" would feed a continuous integer column into groupby and
        # emit statistically meaningless output under a fairness heading — worse
        # than no audit, because it looks like one.
        #
        # Verified against all 62 real column names across the four tables: this
        # list returns exactly identified_race (neighborhood_houses) and gender
        # (education_survey). test_s4_demographic_detection_precision pins that.
        # Note "male" is also a substring of "female", so it double-matches.
        keywords = [
            "race",
            "ethnic",
            "gender",
            "sex",
            "religio",
            "disab",
            "marital",
            "veteran",
            "nationality",
            "citizenship",
            "orientation",
        ]

        return [
            col for col in columns
            if any(k in col.lower() for k in keywords)
        ]

    # ============================================
    # BINARY ENCODING
    # ============================================
    def __to_binary_indicator(self, series):
        """Map a two-valued column onto 0/1 and report which value counts as positive.

        Fairlearn's selection_rate defaults to pos_label=1. Handed a TEXT binary
        such as ownership ('Own'/'Rent') it evaluates 'Own' == 1, which is never
        true, so every group scores 0.0000 and the demographic parity difference
        comes back 0.0000 — printed as "Within fairness threshold" for a column
        that was never actually measured. A fairness metric's silence is not
        evidence of fairness, so encode explicitly and say what was assumed.
        """
        values = pd.unique(series.dropna())
        if len(values) != 2:
            return series, None

        # Already a usable indicator: booleans, or ints that are exactly {0, 1}.
        if pd.api.types.is_bool_dtype(series):
            return series.astype(int), True
        if pd.api.types.is_numeric_dtype(series) and set(np.unique(values)) <= {0, 1}:
            return series.astype(int), 1

        # Otherwise pick a positive label deterministically: prefer an
        # affirmative-looking value, else the lexicographically greater one so the
        # choice is stable across runs rather than dependent on row order.
        affirmative = {"yes", "y", "true", "t", "own", "owned", "approved", "1"}
        ordered = sorted(values, key=lambda v: str(v))
        positive = next(
            (v for v in ordered if str(v).strip().lower() in affirmative),
            ordered[-1],
        )
        return (series == positive).astype(int), positive

    def __format_bias_caveat(self, findings):
        lines = ["[Fairness notice] The bias audit flagged the following before this answer was returned:"]
        for f in findings:
            lines.append(
                f"  - {f['table']}.{f['column']} across {f['demographic_column']}: "
                f"{f['detail']}"
            )
        lines.append(
            "  Treat comparisons across these groups with care; the disparity may reflect "
            "the underlying data rather than the question asked."
        )
        return "\n".join(lines)

    # ============================================
    # FAIRNESS ENGINE
    # ============================================
    def __run_fairness_analysis(self, df, table_name, demographic_column):

        print("\n" + "-" * 70)
        print(f"TABLE: {table_name}")
        print(f"DEMOGRAPHIC COLUMN: {demographic_column}")
        print("-" * 70)

        for col in df.columns:

            if col == demographic_column:
                continue

            temp = df[[demographic_column, col]].dropna()
            if temp.empty:
                continue

            print(f"\nAnalyzing column: {col}")

            unique_count = temp[col].nunique()

            # ============================================
            # BINARY → Fairlearn
            # ============================================
            if unique_count == 2:

                print("Type: Binary (classification fairness)")

                y, positive_label = self.__to_binary_indicator(temp[col])
                sensitive = temp[demographic_column]

                if positive_label is not None:
                    print(f"Positive label: {positive_label!r}")

                metric_frame = MetricFrame(
                    metrics={"selection_rate": selection_rate},
                    y_true=y,
                    y_pred=y,
                    sensitive_features=sensitive,
                )

                print("\nSelection Rate by Group:")
                print(metric_frame.by_group)

                dp_diff = demographic_parity_difference(
                    y_true=y,
                    y_pred=y,
                    sensitive_features=sensitive,
                )

                print(f"Demographic Parity Difference: {dp_diff:.4f}")

                if abs(dp_diff) > DEMOGRAPHIC_PARITY_THRESHOLD:
                    print("Potential bias detected")
                    self.last_bias_findings.append({
                        "table": table_name,
                        "column": col,
                        "demographic_column": demographic_column,
                        "metric": "demographic_parity_difference",
                        "value": float(dp_diff),
                        "detail": (
                            f"demographic parity difference {dp_diff:.4f} exceeds "
                            f"{DEMOGRAPHIC_PARITY_THRESHOLD:.2f} "
                            f"(positive label {positive_label!r})"
                        ),
                    })
                else:
                    print("Within fairness threshold")

            # ============================================
            # NUMERIC → Mean Comparison
            # ============================================
            elif pd.api.types.is_numeric_dtype(temp[col]):

                print("Type: Numeric (distribution fairness)")

                means = temp.groupby(demographic_column)[col].mean()
                print("\nMean by group:")
                print(means)

                min_val = means.min()
                max_val = means.max()
                ratio = min_val / max_val if max_val > 0 else 0

                print(f"Mean ratio (min/max): {ratio:.3f}")

                if ratio < MEAN_RATIO_THRESHOLD:
                    print("Large disparity detected")
                    self.last_bias_findings.append({
                        "table": table_name,
                        "column": col,
                        "demographic_column": demographic_column,
                        "metric": "mean_ratio",
                        "value": float(ratio),
                        "detail": (
                            f"group mean ratio {ratio:.3f} is below "
                            f"{MEAN_RATIO_THRESHOLD:.2f} "
                            f"(lowest {min_val:.2f}, highest {max_val:.2f})"
                        ),
                    })
                else:
                    print("No major disparity")

            # ============================================
            # SMALL CATEGORICAL → Distribution Compare
            # ============================================
            elif 2 < unique_count <= 10:

                print("Type: Categorical (distribution comparison)")

                dist = pd.crosstab(
                    temp[demographic_column],
                    temp[col],
                    normalize="index"
                ) * 100

                print("\nPercentage distribution by group:")
                print(dist.round(2))

            else:
                print("Skipping (too many categories)")

    # ============================================
    # AUTO BIAS CHECK
    # ============================================
    def __auto_bias_check(self, db_uri):

        tables, engine = self.__discover_tables(db_uri)

        for table_name in tables:

            df = pd.read_sql(f"SELECT * FROM {table_name}", engine)

            demographic_cols = self.__find_demographic_columns(df.columns)

            if not demographic_cols:
                continue

            for demo_col in demographic_cols:
                # Defence in depth behind the keyword list: a sensitive feature is
                # a group label, so it must be low-cardinality and non-numeric. A
                # continuous column that slipped through the names would produce
                # thousands of one-row "groups".
                series = df[demo_col]
                if pd.api.types.is_numeric_dtype(series) or series.nunique() > 20:
                    print(
                        f"\nSkipping {table_name}.{demo_col}: "
                        f"{series.nunique()} distinct values of dtype {series.dtype} "
                        "is a measurement, not a group label."
                    )
                    continue

                self.__run_fairness_analysis(df, table_name, demo_col)
