# Robustness Rules

## Error Handling
- Every command must have try/except with user-facing error message
- Error messages are ALWAYS in-character (via `get_error_response()` in `core/errors.py`)
- NEVER expose "Claude", "Anthropic", "API", or internal error types to users
- DB write failures are logged but don't crash the command (user message still gets a response)
- Never let exceptions propagate silently — always log with structlog

## Rate Limiting
- !chat: 1 use / 15s per user (protects Claude API tokens)
- !buscar: 1 use / 10s per user
- !memoria: 1 use / 5s per user
- !perfil: 1 use / 10s per user
- !ping: 1 use / 3s per user
- Cooldown errors are in-character (not generic Discord messages)

## LLM Resilience
- Timeout: 30s (configurable via LLM_TIMEOUT)
- Max retries: 5 (configurable via LLM_MAX_RETRIES)
- RateLimitError: exponential backoff (2^attempt seconds)
- AuthenticationError: fail immediately, no retry
- Timeout/ConnectionError: retry with 1s delay
- Other APIError: fail immediately
- Character break detected: auto-retry with reinforced system prompt, then sanitize

## Lifecycle
- Signal handling: SIGTERM/SIGINT → graceful shutdown (close DB, close bot)
- Health check task: every 60s, logs latency + guilds + memory stats
- on_disconnect / on_resumed events logged for connection monitoring
- DB auto-reconnect via _ensure_connection() before every operation

## Logging
- Use structlog everywhere (never print())
- Log events: bot_ready, bot_disconnected, bot_resumed, health_check
- Log LLM: llm_request, llm_response (with token counts), llm_*_error
- Log character: character_break_detected, character_break_fixed_on_retry
- Log style: style_adapted (with user profile metrics)
- Log memory: memory_connected, memory_closed, memory_store_failed
- Log attachments: attachment_processed, attachment_rejected
- Log commands: command_error with user context
