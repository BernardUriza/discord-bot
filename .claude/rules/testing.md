# Testing Rules

## Pre-Push Verification — MANDATORY

Before EVERY push, verify that code actually works at the Python import level, not just at the lint/test level:

1. **SDK class existence**: If you reference a new exception class, enum, or attribute from an external SDK (e.g., `anthropic.OverloadedError`), ALWAYS verify it exists first:
   ```bash
   python3 -c "import anthropic; print(hasattr(anthropic, 'OverloadedError'))"
   ```
   `ruff check` and `pytest` may pass even when the class doesn't exist (if the import is inside a try/except or conditional path). Only a live import test catches this.

2. **Real import smoke test**: After adding imports from external packages, verify the module loads:
   ```bash
   python3 -c "from insult.core.llm import LLMClient; print('OK')"
   ```

3. **Never assume SDK APIs exist**: Always check `dir(module)` or `hasattr(module, 'ClassName')` before using a class you haven't used before in this codebase.

4. **If CI fails, don't just re-push with a guess**: Read the CI error, reproduce it locally, then fix with verification.

## Browser Testing
- Always test the bot's web-facing features using Chrome DevTools (via MCP chrome-devtools)
- Use DevTools to verify Discord interactions when possible: navigate to Discord web, inspect network requests, check console for errors
- Prefer automated browser verification over manual "go check it" instructions

## Debug HTTP Endpoint (read-only introspection)

The bot exposes a read-only HTTP server (`insult/core/debug_server.py`) that Claude Code can hit directly to inspect state without needing the Discord MCP server.

### Setup
- Requires `DEBUG_TOKEN` in `.env` — fail-closed if unset (server does NOT start)
- Generate: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`
- Defaults: `DEBUG_HOST=0.0.0.0`, `DEBUG_PORT=8787`
- Auth: `Authorization: Bearer <token>` header on every endpoint except `/debug/health`

### Endpoints
| Method | Path | Query params | Returns |
|--------|------|--------------|---------|
| GET | `/debug/health` | — | `{"status": "ok"}` (no auth) |
| GET | `/debug/messages` | `channel_id` (req), `limit` (1-500, default 15) | Last N messages for channel |
| GET | `/debug/channels` | `guild_id` (req), `since_hours` (default 24) | Channel activity counts |
| GET | `/debug/stats` | — | Total messages / users / channels |

### How to use from Claude Code
```bash
# Load token from .env
TOKEN=$(grep DEBUG_TOKEN .env | cut -d= -f2)

# Read last 15 messages in a channel
curl -s "http://localhost:8787/debug/messages?channel_id=123456&limit=15" \
  -H "Authorization: Bearer $TOKEN" | jq .

# List active channels in a guild in last 24h
curl -s "http://localhost:8787/debug/channels?guild_id=99999&since_hours=24" \
  -H "Authorization: Bearer $TOKEN" | jq .
```

### Notes
- Read-only — no write paths
- In Azure Container Apps: private by default (no ingress configured)
- Uses `hmac.compare_digest` for timing-safe token comparison
- For local testing, bot must be running (`python -m insult run`)

## E2E Testing with Discord MCP
When you need to verify that the bot actually works end-to-end (not just unit tests), use the Discord MCP server to interact with a real Discord server. This Mac is the server.

### Setup
- The MCP server `barryyip0625/mcp-discord` (or `SaseQ/discord-mcp`) provides tools like `discord_send`, `discord_read_messages`, `discord_search_messages`
- The bot must be running locally on this Mac before E2E tests
- Use a dedicated testing Discord server/channel (not production)
- The bot's `DISCORD_TOKEN` from `.env` is the same token the MCP uses

### How to Run E2E Tests
1. Start the bot locally: `python -m insult run`
2. Use MCP Discord tools to send a command: `discord_send` with `!chat <test message>`
3. Wait briefly, then `discord_read_messages` to read the bot's response
4. Verify the response: no character breaks, correct language, in-character tone
5. Use `discord_search_messages` to verify memory persistence across messages

### E2E Test Mode (App-Level Bypass)
- The app should support a `E2E_TEST_MODE=true` environment variable
- When enabled: reduces cooldowns to 0, disables rate limiting, enables a `!e2e_reset` command to clear test data
- This mode should ONLY be active while the bot is running locally for testing — never in production
- The bypass is controlled via `.env`, not hardcoded

### When to Run E2E Tests
- After implementing new commands or changing chat flow
- After modifying persona.md or character break detection
- After changing memory/context building logic
- When unit tests pass but you want to verify the full integration
- Do this naturally — don't wait for the user to ask. If you changed bot behavior, verify it works

### E2E Test Checklist
- [ ] Bot responds to `!chat` in the correct language
- [ ] Bot stays in character (no "I'm an AI", no "Claude", no apologies)
- [ ] Bot remembers context from previous messages in the same channel
- [ ] Error responses are in-character (not exposing internals)
- [ ] `!perfil` shows style profile after 5+ messages
- [ ] Long messages are chunked correctly (< 2000 chars per message)

## E2E via CI/CD Pipeline (Production Verification)

E2E testing means pushing code, triggering the full CI/CD pipeline, and monitoring the deployment to Azure. This is the real verification flow:

### Steps
1. Commit and push to `main`
2. CI runs (GitHub Actions: lint, tests, audit, security) — monitor with `gh run watch --exit-status`
3. On CI success, CD triggers automatically (workflow_run on CI completion)
4. CD builds Docker image, pushes to Azure Container Registry (insultacr), deploys to Azure Container Apps (insult-bot in insult-rg)
5. Monitor CD deployment via `az` CLI:
   - `gh run list --workflow=cd.yml --limit=3` to check CD run status
   - `az containerapp show --name insult-bot --resource-group insult-rg --query "properties.latestRevisionName"` to verify new revision
   - `az containerapp logs show --name insult-bot --resource-group insult-rg --follow` to tail logs and confirm bot starts healthy
6. Once deployed, verify the bot responds in Discord (either manually or via MCP)

### Key Resources
- **ACR**: insultacr.azurecr.io
- **Container App**: insult-bot (resource group: insult-rg)
- **CD workflow**: `.github/workflows/cd.yml` — triggers on CI success on main
