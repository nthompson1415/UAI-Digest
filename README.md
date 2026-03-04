# UAI Weekly AI Digest
### Brought to you by The Foundry

Automated weekly AI news digest powered by Google Gemini (free tier) with live Google Search grounding. Runs every Monday via GitHub Actions at zero cost.

## How It Works

1. GitHub Actions triggers every Monday at 9:00 AM EST
2. Python script calls Gemini 2.0 Flash with Google Search grounding across 8 news categories
3. Gemini searches the live web, finds recent AI news, and returns structured results
4. Script compiles everything into a formatted Markdown digest
5. Digest is committed to the `digests/` folder and optionally posted to Discord

### Categories Scanned
- ⚡ Top Stories
- 🧠 Models & Benchmarks
- 📄 Research Spotlight
- 💰 Industry & Funding
- ⚖️ Policy & Regulation
- 🔧 Open Source & Tools
- 🛡️ Safety & Alignment
- 🚀 Product Updates

### What's NOT Hardcoded
- No source URLs or blog lists — search queries are open-ended questions
- New sources, companies, or topics get picked up automatically
- Paywalled sources are explicitly excluded so every link is freely readable

---

## Setup (5 minutes)

### 1. Get a free Gemini API key

- Go to [Google AI Studio](https://aistudio.google.com/apikey)
- Sign in with any Google account
- Click "Create API Key"
- Copy the key

### 2. Fork or clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/uai-weekly-digest.git
cd uai-weekly-digest
```

### 3. Add your API key as a GitHub Secret

- Go to your repo → **Settings** → **Secrets and variables** → **Actions**
- Click **New repository secret**
- Name: `GEMINI_API_KEY`
- Value: paste your API key

### 4. (Optional) Add Discord webhook

If you want the digest auto-posted to a Discord channel:

- In Discord: channel settings → Integrations → Webhooks → New Webhook
- Copy the webhook URL
- Add another GitHub Secret:
  - Name: `DISCORD_WEBHOOK_URL`
  - Value: paste the webhook URL

### 5. Run it

The workflow runs automatically every Monday at 9 AM EST. To test immediately:

- Go to repo → **Actions** → **Generate Weekly AI Digest** → **Run workflow**

Or run locally:

```bash
pip install -r requirements.txt
export GEMINI_API_KEY="your-key-here"
python generate_digest.py
```

---

## Output

Digests are saved to `digests/` as timestamped Markdown files:

```
digests/
├── digest-2026-03-03.md
├── digest-2026-03-10.md
├── latest.md          ← always the most recent
└── ...
```

Each digest includes:
- Categorized news items with 2-3 sentence summaries
- Source links for every item (all freely accessible)
- Full source list at the bottom
- Generation timestamp

---

## Customization

### Change the schedule

Edit `.github/workflows/weekly-digest.yml`:

```yaml
schedule:
  - cron: '0 14 * * 1'  # Monday 9AM EST (14:00 UTC)
```

Use [crontab.guru](https://crontab.guru) to set your preferred time.

### Add or remove categories

Edit the `CATEGORIES` list in `generate_digest.py`. Each category is just a prompt — no URLs or source lists to maintain.

### Change the output format

The `format_digest()` function generates Markdown. Modify it to output HTML, plain text, or any format you need.

### Adjust the lookback window

Search prompts default to "past 14 days". Change this in each category's `prompt` field.

---

## Cost

**$0.** The Gemini 2.0 Flash free tier includes:
- 15 requests per minute
- 1 million tokens per day
- Google Search grounding included

The weekly digest uses ~8 API calls. You'd need to run it 120+ times per day to hit limits.

---

## Paywalled Sources (Auto-Excluded)

The script instructs Gemini to skip these publications since their full articles aren't freely readable:

- The Information, MIT Technology Review, Financial Times
- Wall Street Journal, Bloomberg, New York Times
- The Atlantic, Wired, The Economist

Every link in the digest points to a freely accessible source.

---

## Reporting Errors

Found a wrong URL, bad summary, or missing story? [Open an error report](https://github.com/nthompson1415/UAI-Digest/issues/new?template=error_report.md) — the template will ask for the digest date, category, and what went wrong.

---

## License

MIT — do whatever you want with it.
