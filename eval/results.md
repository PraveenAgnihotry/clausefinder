# Evaluation Results

Model: `gemini-2.5-flash`  
Run UTC: `2026-06-25T12:53:02+00:00`

## Headline Metrics

| Metric | Value | n |
| --- | --- | --- |
| Retrieval hit-rate (document-level) | 100.0% (19/19) | 19 |
| Retrieval hit-rate (section-level) | 100.0% (6/6) | 6 |
| Refusal accuracy | 100.0% (4/4) | 4 |
| Faithfulness (verified) | 100.0% (3/3) | 3 |
| Faithfulness (all annotated; includes unverified expectations) | 77.8% (7/9) | 9 |
| Known failures - failed as expected | 100.0% (2/2) | 2 |
| Known failures - unexpectedly passed | 0.0% (0/2) | 2 |
| Errors | 2 | 25 |

## Per-Case Results

| id | type | doc hit | section hit | behavior/faithfulness | known_fail | error |
| --- | --- | --- | --- | --- | --- | --- |
| b_firefighting_stair_width | answer | yes | yes | faithful | — | — |
| m_wheelchair_wc | answer | yes | yes | faithful | — | — |
| b_cavity_barriers | answer | yes | yes | faithful | — | — |
| refuse_part_l_windows | refuse | — | — | behavior ok | — | — |
| k_domestic_stair_width_KNOWNFAIL | answer | yes | yes | not faithful | expected fail (documented) | — |
| k_stair_headroom_KNOWNFAIL | answer | yes | yes | not faithful | expected fail (documented) | — |
| k_stair_max_pitch | answer | yes | no | faithful | — | — |
| k_stair_rise_going | answer | yes | no | not faithful | — | — |
| k_guarding_drop_dwelling | answer | yes | no | not faithful | — | — |
| k_open_riser_sphere | answer | yes | no | faithful | — | — |
| k_stair_max_risers | answer | yes | no | faithful | — | — |
| k_stair_guarding_height | answer | yes | no | refused | — | — |
| b_protected_stairway | answer | yes | yes | answered | — | — |
| b_smoke_alarms_dwelling | answer | yes | no | answered | — | — |
| b_compartment_wall_between_dwellings | answer | yes | no | answered | — | — |
| b_travel_distance_to_exit | answer | yes | no | refused | — | — |
| m_accessible_doorway_width | answer | yes | no | answered | — | — |
| m_ramp_gradient | answer | yes | no | faithful | — | — |
| m_entrance_storey_wc | answer | yes | no | answered | — | — |
| m_step_free_approach | answer | yes | no | answered | — | — |
| refuse_part_f_ventilation | refuse | — | — | behavior ok | — | — |
| refuse_part_e_sound | refuse | — | — | behavior ok | — | — |
| refuse_part_h_drainage | refuse | — | — | behavior ok | — | — |
| refuse_part_a_foundation | refuse | — | — | — | — | 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash\nPlease retry in 51.699967964s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '51s'}]}} |
| refuse_part_p_electrical | refuse | — | — | — | — | 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash\nPlease retry in 10.818099724s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '10s'}]}} |
