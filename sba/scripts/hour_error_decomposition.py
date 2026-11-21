"""Arithmetic decomposition of aggregate amount error, not causal attribution."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/hour'
def decompose(name,n,events,probability,amount):
 expected_events=n*probability['predicted_rate'];observed_severity=amount['actual_total']/events;expected_severity=amount['predicted_total']/expected_events;frequency_ratio=expected_events/events;severity_ratio=expected_severity/observed_severity;assert abs(frequency_ratio*severity_ratio-amount['predicted_actual_ratio'])<1e-10
 return {'cohort':name,'loans':n,'actual_events':events,'expected_events':expected_events,'event_count_ratio':frequency_ratio,'actual_average_amount_per_event':observed_severity,'predicted_probability_weighted_amount_per_event':expected_severity,'severity_ratio':severity_ratio,'amount_ratio':amount['predicted_actual_ratio'],'identity_verified':True}
def main():
 long=json.loads((OUT/'long_robustness.json').read_text());medium=json.loads((OUT/'medium_results.json').read_text());r=medium['results'][medium['selected_group']];rows=[decompose(f"{v['test_year']}Q3 / 72 months",v['n'],v['events'],v['probability'],v['amount']) for v in long['rows']];rows.append(decompose('2024Q3 / 24 months',medium['cohort']['n'],medium['cohort']['events'],r['probability'],r['amount']));out={'arithmetic_identity_not_causal_explanation':True,'not_a_new_model_or_independent_test':True,'rows':rows};(OUT/'error_decomposition.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
