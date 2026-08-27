"""Fourth agent: environmental sensor time-series over Azure Table Storage.

Extends the three required agents to a fourth data modality. It follows the same
contract as its siblings — one data source, one credential, one ethical check run
inline during query processing, a dict carrying "response".

The safeguard here is a reliability guardrail rather than a fairness or privacy
one. Sensor feeds drift, spike, and drop out; presenting a statistic computed over
a contaminated or too-short window as though it were fact is the ethical failure
this agent is built to avoid. Anomalies are surfaced explicitly, and where the
evidence is too thin to support a conclusion the agent says so instead of
answering.
"""

import re
import statistics
from collections import defaultdict

from azure.data.tables import TableServiceClient
from langchain_openai import AzureChatOpenAI

NEIGHBORHOODS = ["Ashford", "Huntington", "Kingsley", "Maplewood", "Rosedale"]
METRICS = {
    "pm25": ("air quality (PM2.5)", "µg/m³"),
    "noise_db": ("noise level", "dB"),
    "water_gal": ("water usage", "gallons"),
}

# A reading this many standard deviations from its neighborhood-and-metric mean is
# reported as anomalous. 3.0 is the conventional threshold and keeps false
# positives low on 336 readings per series.
ZSCORE_THRESHOLD = 3.0

# Below this many readings a mean is not worth stating. One day of hourly data.
MIN_READINGS_FOR_CONCLUSION = 24


class TimeSeriesDataAgent:
    def __init__(
        self,
        connection_string,
        table_name,
        azure_endpoint,
        api_key,
        deployment,
        api_version="2024-12-01-preview",
        temperature=0.0,
    ):
        self.table_service = TableServiceClient.from_connection_string(connection_string)
        self.table_name = table_name
        self.table_client = self.table_service.get_table_client(table_name)

        self.model = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            azure_deployment=deployment,
            api_version=api_version,
            temperature=temperature,
        )

        self.last_anomalies = []
        self.last_refused = False

    # ============================================
    # PUBLIC METHOD
    # ============================================
    def ask(self, question, run_anomaly_audit=True):
        neighborhood = self.__extract_neighborhood(question)
        metric = self.__extract_metric(question)

        rows = self.__fetch(neighborhood)
        self.last_anomalies = []
        self.last_refused = False

        if not rows:
            message = (
                "No sensor readings are available for that query. "
                "The time-series store returned nothing to analyse."
            )
            return {
                "question": question,
                "response": message,
                "anomalies": [],
                "refused": True,
            }

        summary = self.__summarize(rows, metric)

        if run_anomaly_audit:
            print("\n" + "=" * 80)
            print("Using Time-Series Data stored in Azure Table Storage to answer question")
            print("Running anomaly detection before providing answer")
            print("=" * 80)
            self.last_anomalies = self.__detect_anomalies(rows, metric)
            self.__report_anomalies(self.last_anomalies, rows, metric)

        # Reliability guardrail: too little data to state a conclusion from.
        insufficient = [
            f"{name} ({stats['count']} readings)"
            for name, stats in summary.items()
            if stats["count"] < MIN_READINGS_FOR_CONCLUSION
        ]
        if insufficient and len(insufficient) == len(summary):
            self.last_refused = True
            message = (
                "Withholding a conclusion: every series matching this query has fewer "
                f"than {MIN_READINGS_FOR_CONCLUSION} readings "
                f"({'; '.join(insufficient)}). That is too little data to state an "
                "average or a trend responsibly."
            )
            print(message)
            return {
                "question": question,
                "response": message,
                "anomalies": self.last_anomalies,
                "refused": True,
            }

        prompt = self.__build_prompt(question, summary, metric, self.last_anomalies)
        response = self.model.invoke(prompt)
        answer = response.content

        if self.last_anomalies:
            answer = f"{self.__format_caveat(self.last_anomalies)}\n\n{answer}"

        return {
            "question": question,
            "response": answer,
            "anomalies": self.last_anomalies,
            "refused": False,
        }

    # ============================================
    # PRIVATE METHODS
    # ============================================
    def __extract_neighborhood(self, question):
        lowered = question.lower()
        for name in NEIGHBORHOODS:
            if name.lower() in lowered:
                return name
        return None

    def __extract_metric(self, question):
        lowered = question.lower()
        if re.search(r"\b(air|pm2\.?5|pollution|particulate|quality)\b", lowered):
            return "pm25"
        if re.search(r"\b(noise|sound|loud|decibel|db)\b", lowered):
            return "noise_db"
        if re.search(r"\b(water|usage|consumption|gallon)\b", lowered):
            return "water_gal"
        return None

    def __fetch(self, neighborhood=None):
        if neighborhood:
            entities = self.table_client.query_entities(
                query_filter="PartitionKey eq @nb",
                parameters={"nb": neighborhood},
            )
        else:
            entities = self.table_client.list_entities()

        rows = []
        for e in entities:
            rows.append({
                "neighborhood": e["PartitionKey"],
                "timestamp": e["RowKey"],
                "sensor_id": e.get("sensor_id"),
                "pm25": float(e["pm25"]),
                "noise_db": float(e["noise_db"]),
                "water_gal": float(e["water_gal"]),
            })
        return rows

    def __series(self, rows, metric):
        """Group values by neighborhood for one metric, or for all metrics."""
        metrics = [metric] if metric else list(METRICS)
        grouped = defaultdict(list)
        for r in rows:
            for m in metrics:
                grouped[(r["neighborhood"], m)].append(r)
        return grouped

    def __summarize(self, rows, metric):
        summary = {}
        for (neighborhood, m), items in self.__series(rows, metric).items():
            values = [i[m] for i in items]
            summary[f"{neighborhood} {m}"] = {
                "count": len(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "unit": METRICS[m][1],
            }
        return summary

    def __detect_anomalies(self, rows, metric):
        anomalies = []
        for (neighborhood, m), items in self.__series(rows, metric).items():
            values = [i[m] for i in items]
            if len(values) < 2:
                continue
            mean = statistics.fmean(values)
            stdev = statistics.pstdev(values)
            if stdev == 0:
                continue
            for item in items:
                z = (item[m] - mean) / stdev
                if abs(z) >= ZSCORE_THRESHOLD:
                    anomalies.append({
                        "neighborhood": neighborhood,
                        "metric": m,
                        "timestamp": item["timestamp"],
                        "sensor_id": item["sensor_id"],
                        "value": item[m],
                        "zscore": z,
                        "series_mean": mean,
                    })
        anomalies.sort(key=lambda a: abs(a["zscore"]), reverse=True)
        return anomalies

    def __report_anomalies(self, anomalies, rows, metric):
        series_count = len(self.__series(rows, metric))
        print(f"Readings analysed: {len(rows)} across {series_count} series")
        print(f"Anomaly rule: |z| >= {ZSCORE_THRESHOLD} against the series mean")

        if not anomalies:
            print("\nNo anomalous readings detected.\n")
            return

        print(f"\nWARNING: {len(anomalies)} anomalous reading(s) detected:")
        for a in anomalies:
            label, unit = METRICS[a["metric"]]
            print(
                f"- {a['neighborhood']} {label}: {a['value']:.2f} {unit} "
                f"at {a['timestamp']} (sensor {a['sensor_id']}, "
                f"z={a['zscore']:+.2f}, series mean {a['series_mean']:.2f})"
            )
        print()

    def __format_caveat(self, anomalies):
        lines = [
            (
                "[Reliability notice] Anomaly detection flagged readings that may "
                "distort any statistic computed over this window:"
            )
        ]
        for a in anomalies[:5]:
            label, unit = METRICS[a["metric"]]
            lines.append(
                f"  - {a['neighborhood']} {label} {a['value']:.2f} {unit} "
                f"at {a['timestamp']} (z={a['zscore']:+.2f})"
            )
        if len(anomalies) > 5:
            lines.append(f"  - ...and {len(anomalies) - 5} more")
        lines.append(
            "  These are included in the figures below. Treat means over the affected "
            "series as provisional until the sensors are checked."
        )
        return "\n".join(lines)

    def __build_prompt(self, question, summary, metric, anomalies):
        lines = ["Environmental sensor statistics for the five neighborhoods:", ""]
        for name, s in sorted(summary.items()):
            lines.append(
                f"{name}: n={s['count']}, mean={s['mean']:.2f} {s['unit']}, "
                f"median={s['median']:.2f}, sd={s['stdev']:.2f}, "
                f"range {s['min']:.2f}-{s['max']:.2f}"
            )

        if anomalies:
            lines += ["", f"Anomalous readings detected ({len(anomalies)}):"]
            for a in anomalies[:10]:
                lines.append(
                    f"- {a['neighborhood']} {a['metric']}={a['value']:.2f} "
                    f"at {a['timestamp']} (z={a['zscore']:+.2f})"
                )

        lines += [
            "",
            f"Question: {question}",
            "",
            (
                "Answer using only the statistics above. Cite the specific numbers "
                "you use. If anomalies were detected, say plainly that they may be "
                "affecting the figures. Do not speculate beyond the data."
            ),
        ]
        return "\n".join(lines)
