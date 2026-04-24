# KQL Queries — insult-bot prod

Workspace customer ID: `a07bf4c8-22ff-455a-b7bd-91055da53b28`
Table: `ContainerAppConsoleLogs_CL`
Filter column: `ContainerAppName_s == "insult-bot"`
Log string column: `Log_s`

Run via `scripts/kql.sh 'QUERY'` or paste into Azure Portal → Log Analytics workspace → Logs.

## 1. Event rate (pipeline sanity check)

```kql
ContainerAppConsoleLogs_CL
| where TimeGenerated > ago(1h)
| where ContainerAppName_s == "insult-bot"
| summarize count() by bin(TimeGenerated, 5m)
| order by TimeGenerated desc
```

## 2. Reconstruct a single turn by request_id

```kql
ContainerAppConsoleLogs_CL
| where TimeGenerated > ago(6h)
| where ContainerAppName_s == "insult-bot"
| where Log_s contains "request_id=<PASTE_ID>"
| project TimeGenerated, Log_s
| order by TimeGenerated asc
```

## 3. All LLM timeouts in the last 2 hours

```kql
ContainerAppConsoleLogs_CL
| where TimeGenerated > ago(2h)
| where ContainerAppName_s == "insult-bot"
| where Log_s contains "llm_timeout" or Log_s contains "llm_failed"
| project TimeGenerated, Log_s
| order by TimeGenerated desc
```

## 4. Turns that ended with non-ok outcome

```kql
ContainerAppConsoleLogs_CL
| where TimeGenerated > ago(6h)
| where ContainerAppName_s == "insult-bot"
| where Log_s contains "chat_turn_end"
| where Log_s !contains "outcome=ok"
| project TimeGenerated, Log_s
| order by TimeGenerated desc
```

## 5. Character break events (what pattern, what text)

```kql
ContainerAppConsoleLogs_CL
| where TimeGenerated > ago(24h)
| where ContainerAppName_s == "insult-bot"
| where Log_s contains "character_break_detected"
| project TimeGenerated, Log_s
| order by TimeGenerated desc
```

## 6. Preset distribution (drift check)

```kql
ContainerAppConsoleLogs_CL
| where TimeGenerated > ago(24h)
| where ContainerAppName_s == "insult-bot"
| where Log_s contains "preset_classified"
| extend mode = extract(@"mode=(\w+)", 1, Log_s)
| summarize count() by mode
| order by count_ desc
```

## 7. Cache read/create per turn (is prompt caching working?)

```kql
ContainerAppConsoleLogs_CL
| where TimeGenerated > ago(2h)
| where ContainerAppName_s == "insult-bot"
| where Log_s contains "llm_response"
| extend cache_read = toint(extract(@"cache_read=(\d+)", 1, Log_s))
| extend cache_create = toint(extract(@"cache_create=(\d+)", 1, Log_s))
| extend input_tokens = toint(extract(@"input_tokens=(\d+)", 1, Log_s))
| project TimeGenerated, cache_read, cache_create, input_tokens
| order by TimeGenerated desc
```

## 8. Delivery failures (Discord HTTP errors)

```kql
ContainerAppConsoleLogs_CL
| where TimeGenerated > ago(24h)
| where ContainerAppName_s == "insult-bot"
| where Log_s contains "chat_delivery_failed" or Log_s contains "delivery_chunk_failed"
| project TimeGenerated, Log_s
| order by TimeGenerated desc
```
