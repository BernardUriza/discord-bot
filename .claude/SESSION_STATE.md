# SESSION STATE - 2026-03-31 (buenas noches)

## Quick Summary
Massive session: modularized bot.py monolith into `insult/` package with DI, fixed .env priority with `settings_customise_sources()`, added SecretStr, deployed to Azure Container Apps, cleaned Azure billing ($122 paid, all old resources deleted), created CI/CD workflows, ran /cruel-critic audit (3 fixes applied, 73 tests passing). Bot is LIVE on Azure responding in Discord.

## Active Work

### discord-bot — Main Bot (Azure Deployed)
- **Branch:** `main`
- **Repo:** github.com/BernardUriza/discord-bot
- **Status:** Deployed to Azure, needs commit of remaining local changes
- **PR:** none
- **Last change:** /cruel-critic fixes (delete_before method, on_ready guard, Dockerfile EXPOSE removed), CD workflow
- **Next step:** Commit uncommitted files, configure GitHub AZURE_CREDENTIALS secret for CD pipeline

## Azure Infrastructure (LIVE)
- **Resource Group:** `insult-rg` (East US)
- **Container Registry:** `insultacr` (Basic, insultacr.azurecr.io)
- **Container Apps Env:** `insult-env`
- **Container App:** `insult-bot` (0.25 vCPU, 0.5 GiB, min=1, max=1)
- **Image:** `insultacr.azurecr.io/insult-bot:latest`
- **Status:** Running, responding to !chat in Discord
- **Estimated cost:** ~$5-7/mes (idle rate) + ACR $5/mes = ~$12/mes
- **Old resources:** ALL DELETED (aurity-prod, rg-vhouse-prod, sentient-friend-rg, aurity-ci-cd)

## Uncommitted Changes
Local files not yet committed:
- `.claude/rules/workflow.md` — CI pipeline + git workflow rules
- `.claude/rules/architecture.md` — updated (by linter)
- `.claude/rules/robustness.md` — updated with character break, style logging
- `.dockerignore` — excludes .venv, .env, .git, __pycache__
- `Dockerfile` — Python 3.14-slim, no EXPOSE (not a web server)
- `.github/workflows/cd.yml` — CD pipeline: CI pass → ACR build → Container Apps deploy
- `.coverage` — test coverage data (gitignore candidate)
- `CLAUDE.md` — project-level Claude instructions

## Commits Made This Session (by other bot)
- `844dd5f` — Refactor to insult/ package with DI, style adaptation, character guard, and CI
- `34c2d72` — Pin Python 3.14
- `eeb7cd1` — Add 4-layer CI pipeline
- `2abe454` — Add multimodal attachment support

## Key Improvements Made
1. **Modularized bot.py** → insult/ package (config, app, bot, cogs/chat, cogs/utility, core/llm, core/memory, core/character, core/errors, core/style)
2. **DI Container** — `Container` dataclass, `create_app()` factory, cogs receive deps via constructor
3. **CLI** — `python -m insult run`, `python -m insult db-stats`, `python -m insult db-clean`
4. **.env priority fix** — `settings_customise_sources()` (dotenv > shell env), no more os.environ mutation
5. **SecretStr** — discord_token and anthropic_api_key are SecretStr, never logged
6. **Style profiling** — EMA-based user style detection (language, formality, technical level, verbosity, emoji)
7. **Character break detection** — regex patterns + auto-retry with reinforced prompt + sanitization
8. **Adaptive prompts** — system prompt adapts to user style while preserving base persona
9. **Tests** — 73 tests (character, errors, style, chat cog), 80% coverage gate
10. **CI** — 4-layer pipeline (ruff, pytest, pip-audit, bandit)
11. **Azure deploy** — Container Apps, ACR cloud build, secrets via secretref

## Known Issues
1. **CD needs AZURE_CREDENTIALS** — GitHub secret not yet configured for CD pipeline
2. **Memory is ephemeral on Azure** — SQLite in container, data lost on restart. Need persistent volume or switch to Azure PostgreSQL (future)
3. **Pyright warnings** — `_db` Optional access warnings (safe at runtime via _ensure_connection guard)
4. **Mock SecretStr mismatch** — test mocks use plain strings for tokens instead of SecretStr (MENOR)
5. **pip-audit not installed in CI** — ci.yml runs pip-audit without installing it first
6. **Test deps in requirements.txt** — pytest/pytest-asyncio should be in dev deps only

## Discord Setup
- App ID: 1488415576551325906
- Bot: Insult#2662
- Server: "Insult" (ID: 1488419218302042223)
- Channels: #general, #testing
- Intents: Presence, Server Members, Message Content
- Discord account: bernard2389

## Important Context
- **Two users**: Bernardo (builder) and Ale Nava (co-user, non-technical)
- **Purpose**: Therapeutic/conversational AI companion with longitudinal memory
- **Ale Nava** is under psychiatric care (sertralina + quetiapina), bot is supplementary
- **"Insult"** is aggressive but warm therapeutic style personality
- **Cost target**: ~$20 USD/month for both users (Claude API pay-per-token)
- **Azure subscription**: buo45_12@hotmail.com, paid $122.05 on 2026-03-31, now Enabled
- **Bernard works at**: visa/migration documents company that uses AI (happy there)

## Roadmap
- ~~Fix API key~~ ✅
- ~~Insult personality~~ ✅
- ~~Modularize to package~~ ✅
- ~~Deploy to Azure~~ ✅
- **Invite Ale Nava** ← next
- **Configure CD (AZURE_CREDENTIALS)** ← next
- **Persistent storage** — Azure Files or PostgreSQL for memory
- Multi-model (Qwen) → Fine-tuning

## To Resume
1. `cd ~/Documents/discord-bot`
2. Commit all uncommitted files (cd.yml, Dockerfile, .dockerignore, rules updates)
3. Configure `AZURE_CREDENTIALS` GitHub secret for CD pipeline:
   ```bash
   az ad sp create-for-rbac --name insult-cd --role contributor --scopes /subscriptions/d61ba6bc-eda9-4327-a264-5cfddef30bc8/resourceGroups/insult-rg --json-auth
   ```
   Then add output as GitHub secret `AZURE_CREDENTIALS`
4. Invite Ale Nava to Discord server "Insult"
5. Consider persistent storage for memory (SQLite in container = ephemeral)
6. Bot is running on Azure — check with: `az containerapp logs show --name insult-bot --resource-group insult-rg --tail 10`
