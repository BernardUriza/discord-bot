# Testing Rules

## Browser Testing
- Always test the bot's web-facing features using Chrome DevTools (via MCP chrome-devtools)
- Use DevTools to verify Discord interactions when possible: navigate to Discord web, inspect network requests, check console for errors
- Prefer automated browser verification over manual "go check it" instructions

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
