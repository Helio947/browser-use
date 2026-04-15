"""
LinkedIn AI Marketing & Sales Agent
====================================

A fully autonomous agent that handles all your LinkedIn marketing and sales activities.

Capabilities:
  - post_content       Post updates, articles, or carousels to your feed
  - find_leads         Search LinkedIn for target prospects and save them to CSV
  - send_connections   Send personalised connection requests to a prospect list
  - send_messages      Send personalised DMs to existing connections
  - engage_content     Like, comment, and share relevant posts in your niche
  - monitor_analytics  Scrape impressions, reactions and comments for recent posts
  - profile_update     Update your headline, about section, or featured section
  - full_campaign      Run a complete outreach campaign end-to-end

Setup:
  1. Copy .env.example → .env and fill in your values.
  2. Point CHROME_PATH at your installed Chrome / Chromium binary.
  3. Log in to LinkedIn in that browser profile manually (one-time).
  4. Run:  python linkedin_agent.py --mode find_leads

Requirements:
  pip install browser-use langchain-openai python-dotenv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from browser_use import ActionResult, Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LinkedInConfig:
    """All tuneable settings for the LinkedIn agent."""

    # --- LLM ---
    llm_provider: str = "openai"          # "openai" | "anthropic"
    model: str = "gpt-4o"
    temperature: float = 0.3

    # --- Browser ---
    chrome_path: str = os.getenv(
        "CHROME_PATH",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",  # macOS default
    )
    headless: bool = False                 # Keep False to avoid bot-detection
    cookies_file: Optional[str] = os.getenv("LINKEDIN_COOKIES_FILE")  # alternative auth

    # --- LinkedIn identity ---
    your_name: str = os.getenv("LINKEDIN_NAME", "Your Name")
    your_headline: str = os.getenv("LINKEDIN_HEADLINE", "")
    your_profile_url: str = os.getenv("LINKEDIN_PROFILE_URL", "https://www.linkedin.com/in/yourhandle")

    # --- Targeting ---
    target_keywords: List[str] = field(default_factory=lambda: ["SaaS founder", "startup CEO"])
    target_location: str = "United States"
    target_industry: str = ""              # e.g. "Software Development"
    daily_connection_limit: int = 20       # Stay well below LinkedIn's 100/week cap
    daily_message_limit: int = 15

    # --- Content ---
    post_topic: str = ""                   # Topic / draft for post_content mode
    post_hashtags: List[str] = field(default_factory=lambda: ["#LinkedIn", "#Marketing"])

    # --- Outreach templates ---
    connection_note_template: str = (
        "Hi {first_name}, I came across your profile and loved your work in {industry}. "
        "Would love to connect and exchange ideas!"
    )
    message_template: str = (
        "Hi {first_name}, thanks for connecting! "
        "I help {target_persona} achieve {value_prop}. "
        "Would you be open to a quick 15-min call? No pitch — just value."
    )
    value_prop: str = os.getenv("LINKEDIN_VALUE_PROP", "grow their business with AI automation")
    target_persona: str = os.getenv("LINKEDIN_TARGET_PERSONA", "founders and sales leaders")

    # --- Output ---
    data_dir: Path = Path("linkedin_data")
    leads_file: str = "leads.csv"
    analytics_file: str = "analytics.json"
    max_steps: int = 80


# ---------------------------------------------------------------------------
# Pydantic output models (structured data from the agent)
# ---------------------------------------------------------------------------

class LeadProfile(BaseModel):
    name: str
    headline: Optional[str] = None
    location: Optional[str] = None
    profile_url: str
    connection_degree: Optional[str] = None   # "1st", "2nd", "3rd"
    company: Optional[str] = None
    mutual_connections: Optional[int] = None
    notes: Optional[str] = None


class LeadList(BaseModel):
    leads: List[LeadProfile]


class PostAnalytics(BaseModel):
    post_url: str
    impressions: Optional[int] = None
    reactions: Optional[int] = None
    comments: Optional[int] = None
    reposts: Optional[int] = None
    published_at: Optional[str] = None


class AnalyticsReport(BaseModel):
    scraped_at: str
    posts: List[PostAnalytics]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_data_dir(cfg: LinkedInConfig) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)


def _save_leads(leads: List[LeadProfile], cfg: LinkedInConfig) -> None:
    _ensure_data_dir(cfg)
    path = cfg.data_dir / cfg.leads_file
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "headline", "location", "company",
                        "profile_url", "connection_degree", "mutual_connections", "notes"],
        )
        if not file_exists:
            writer.writeheader()
        for lead in leads:
            writer.writerow(lead.model_dump())
    logger.info("Saved %d leads → %s", len(leads), path)


def _save_analytics(report: AnalyticsReport, cfg: LinkedInConfig) -> None:
    _ensure_data_dir(cfg)
    path = cfg.data_dir / cfg.analytics_file
    with path.open("w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
    logger.info("Analytics saved → %s", path)


def _build_llm(cfg: LinkedInConfig):
    if cfg.llm_provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set.")
        return ChatAnthropic(model=cfg.model, temperature=cfg.temperature, api_key=key)
    else:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set.")
        return ChatOpenAI(model=cfg.model, temperature=cfg.temperature, api_key=key)


def _build_browser(cfg: LinkedInConfig) -> Browser:
    ctx_cfg = BrowserContextConfig(
        wait_between_actions=1.5,          # polite pacing
        minimum_wait_page_load_time=1.0,
        maximum_wait_page_load_time=10.0,
        cookies_file=cfg.cookies_file,
        allowed_domains=["linkedin.com", "www.linkedin.com"],
    )
    return Browser(
        config=BrowserConfig(
            headless=cfg.headless,
            chrome_instance_path=cfg.chrome_path if not cfg.cookies_file else None,
            disable_security=True,
            new_context_config=ctx_cfg,
        )
    )


# ---------------------------------------------------------------------------
# Mode: post_content
# ---------------------------------------------------------------------------

async def run_post_content(cfg: LinkedInConfig) -> None:
    """Draft and publish a LinkedIn post on a given topic."""
    if not cfg.post_topic:
        raise ValueError("Set cfg.post_topic before using post_content mode.")

    hashtags = " ".join(cfg.post_hashtags)
    task = f"""
You are a LinkedIn content expert managing the account of {cfg.your_name}.

Your job:
1. Go to https://www.linkedin.com/feed/
2. Click the "Start a post" box at the top of the feed.
3. Write an engaging, value-driven LinkedIn post on this topic:

   TOPIC: {cfg.post_topic}

   Writing guidelines:
   - Open with a powerful hook (1-2 lines, no fluff)
   - Use short paragraphs (1-2 sentences each)
   - Include 3-5 actionable insights or a personal story
   - End with a clear call-to-action (comment, share, or DM)
   - Append these hashtags at the very end: {hashtags}
   - Keep total length between 800-1500 characters

4. Review the drafted post to make sure it looks correct.
5. Click the "Post" button to publish.
6. Confirm the post appeared in the feed and report the post URL if visible.

Important:
- Do NOT click any "+ Add" or scheduling buttons.
- Do NOT attach images unless the topic specifically requires one.
- If a modal or overlay appears, close it before proceeding.
"""
    browser = _build_browser(cfg)
    agent = Agent(
        task=task,
        llm=_build_llm(cfg),
        browser=browser,
        use_vision=True,
        max_failures=3,
        extend_system_message=(
            "You are operating a real LinkedIn account. Be professional and human-like. "
            "Never spam or violate LinkedIn's terms of service."
        ),
    )
    history = await agent.run(max_steps=cfg.max_steps)
    logger.info("post_content result: %s", history.final_result())
    await browser.close()


# ---------------------------------------------------------------------------
# Mode: find_leads
# ---------------------------------------------------------------------------

async def run_find_leads(cfg: LinkedInConfig) -> LeadList:
    """Search LinkedIn for prospects and return structured lead data."""
    keywords = ", ".join(cfg.target_keywords)
    controller = Controller(output_model=LeadList)

    task = f"""
You are a LinkedIn lead-generation specialist.

Search LinkedIn for potential prospects matching this profile:
  Keywords : {keywords}
  Location : {cfg.target_location}
  Industry : {cfg.target_industry or "Any"}

Steps:
1. Go to https://www.linkedin.com/search/results/people/
2. Use the search filters to enter the keywords and location.
   - Click "All filters" to access advanced filters if needed.
   - Apply Industry filter if "{cfg.target_industry}" is not empty.
3. Scroll through at least 2 pages of results.
4. For each person you find, collect:
   - Full name
   - Headline / job title
   - Location
   - Company (current)
   - Profile URL (click their name to open profile, copy URL, then go back)
   - Connection degree (1st / 2nd / 3rd+)
   - Number of mutual connections (if shown)
5. Aim to collect at least 15 leads.
6. Return ALL collected leads in the structured output format.

Important:
- Skip sponsored / ad results.
- Do NOT send any connection requests in this mode — only observe and collect.
- If LinkedIn asks you to verify, stop and report "verification required".
"""
    browser = _build_browser(cfg)
    agent = Agent(
        task=task,
        llm=_build_llm(cfg),
        browser=browser,
        controller=controller,
        use_vision=True,
        max_failures=3,
    )
    history = await agent.run(max_steps=cfg.max_steps)
    await browser.close()

    raw = history.final_result()
    if not raw:
        logger.warning("find_leads returned no results.")
        return LeadList(leads=[])

    lead_list = LeadList.model_validate_json(raw)
    _save_leads(lead_list.leads, cfg)
    return lead_list


# ---------------------------------------------------------------------------
# Mode: send_connections
# ---------------------------------------------------------------------------

async def run_send_connections(cfg: LinkedInConfig) -> None:
    """Read leads from CSV and send personalised connection requests."""
    leads_path = cfg.data_dir / cfg.leads_file
    if not leads_path.exists():
        raise FileNotFoundError(
            f"Leads file not found at {leads_path}. Run --mode find_leads first."
        )

    # Load 2nd-degree leads that haven't been contacted yet
    leads_to_connect: List[dict] = []
    with leads_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("connection_degree") in ("2nd", "3rd+", "3rd", "") and \
               row.get("notes", "").lower() not in ("connected", "request sent"):
                leads_to_connect.append(row)
                if len(leads_to_connect) >= cfg.daily_connection_limit:
                    break

    if not leads_to_connect:
        logger.info("No new leads to connect with today.")
        return

    profiles_json = json.dumps(
        [{"name": l["name"], "url": l["profile_url"], "industry": l.get("headline", "")}
         for l in leads_to_connect],
        indent=2,
    )

    task = f"""
You are a LinkedIn outreach specialist managing the account of {cfg.your_name}.

Your job is to send personalised connection requests to the following prospects.
Do NOT exceed {cfg.daily_connection_limit} requests in this session.

Prospect list (JSON):
{profiles_json}

Connection note template (personalise for each person):
"{cfg.connection_note_template}"

Steps for EACH prospect:
1. Navigate to their profile URL.
2. Click the "Connect" button (it may be inside a "More" dropdown).
3. Click "Add a note".
4. Write a personalised note using the template above — replace {{first_name}} with
   their first name and {{industry}} with their current role/industry from their profile.
   Keep the note under 300 characters.
5. Click "Send" to submit the request.
6. Wait 3 seconds before moving to the next prospect.

Important:
- If the "Connect" button is not visible, skip that person.
- If LinkedIn shows a "You've reached the weekly invitation limit" warning, STOP immediately.
- Never send a blank note.
- Report how many requests were sent successfully.
"""
    browser = _build_browser(cfg)
    agent = Agent(
        task=task,
        llm=_build_llm(cfg),
        browser=browser,
        use_vision=True,
        max_failures=4,
        extend_system_message=(
            "Be respectful, human-paced, and professional. "
            "Strictly respect LinkedIn rate limits — stop if warned."
        ),
    )
    history = await agent.run(max_steps=cfg.max_steps)
    logger.info("send_connections result: %s", history.final_result())
    await browser.close()


# ---------------------------------------------------------------------------
# Mode: send_messages
# ---------------------------------------------------------------------------

async def run_send_messages(cfg: LinkedInConfig) -> None:
    """Send personalised DMs to 1st-degree connections."""
    task = f"""
You are a LinkedIn sales outreach specialist for {cfg.your_name}.

Your goal: Find relevant 1st-degree connections who have NOT been messaged recently,
and send them a personalised outreach message.

Steps:
1. Go to https://www.linkedin.com/mynetwork/invite-connect/connections/
2. Browse through your connections list.
3. For each connection that looks like a {cfg.target_persona}:
   a. Click their profile to confirm they match the target persona.
   b. Click the "Message" button on their profile.
   c. Write a personalised message using this template (fill in the blanks):
      "{cfg.message_template}"
      Replace {{first_name}} with their first name.
      Replace {{target_persona}} with: {cfg.target_persona}
      Replace {{value_prop}} with: {cfg.value_prop}
   d. Click "Send".
   e. Wait 5 seconds before moving on.
4. Stop after sending {cfg.daily_message_limit} messages.

Important:
- Never copy-paste the exact same message — vary the wording slightly each time.
- Skip connections who are recruiters, competitors, or clearly irrelevant.
- If a message fails to send, skip and continue.
- Report the total number of messages sent.
"""
    browser = _build_browser(cfg)
    agent = Agent(
        task=task,
        llm=_build_llm(cfg),
        browser=browser,
        use_vision=True,
        max_failures=4,
    )
    history = await agent.run(max_steps=cfg.max_steps)
    logger.info("send_messages result: %s", history.final_result())
    await browser.close()


# ---------------------------------------------------------------------------
# Mode: engage_content
# ---------------------------------------------------------------------------

async def run_engage_content(cfg: LinkedInConfig) -> None:
    """Like, thoughtfully comment on, and optionally repost relevant content."""
    keywords = ", ".join(cfg.target_keywords)
    task = f"""
You are a LinkedIn engagement specialist for {cfg.your_name} ({cfg.your_headline}).

Your job is to engage authentically with relevant content in your niche.

Steps:
1. Go to https://www.linkedin.com/feed/
2. Scroll through the feed slowly, reading each post.
3. For posts related to these topics: {keywords}
   a. Click the Like (or "Insightful"/"Celebrate") reaction — choose the most fitting one.
   b. If the post has valuable content, leave a thoughtful comment (2-4 sentences).
      - Add genuine value: ask a question, share a related insight, or affirm with specifics.
      - Never write generic comments like "Great post!" or "Thanks for sharing!"
      - Keep comments under 200 characters.
   c. If the post is outstanding and relevant to your audience, click "Repost".
4. Engage with at least 10 posts.
5. Also search for posts using hashtags relevant to your niche:
   - Go to https://www.linkedin.com/search/results/content/ and search for one of: {keywords}
   - Engage with 5 more posts from the search results.

Important:
- Space out interactions — wait 3-5 seconds between each action.
- Do NOT like your own posts.
- Do NOT leave promotional comments about your product/service.
- Be genuinely helpful and professional in all comments.
"""
    browser = _build_browser(cfg)
    agent = Agent(
        task=task,
        llm=_build_llm(cfg),
        browser=browser,
        use_vision=True,
        max_failures=3,
    )
    history = await agent.run(max_steps=cfg.max_steps)
    logger.info("engage_content result: %s", history.final_result())
    await browser.close()


# ---------------------------------------------------------------------------
# Mode: monitor_analytics
# ---------------------------------------------------------------------------

async def run_monitor_analytics(cfg: LinkedInConfig) -> AnalyticsReport:
    """Scrape post analytics (impressions, reactions, comments) from your recent posts."""
    controller = Controller(output_model=AnalyticsReport)

    task = f"""
You are a LinkedIn analytics analyst reviewing the account of {cfg.your_name}.

Steps:
1. Go to {cfg.your_profile_url}
2. Scroll down to the "Activity" section and click "Show all activity".
3. Filter by "Posts" to see only your published posts.
4. For each of the 5 most recent posts:
   a. Note the post URL.
   b. Look for the impressions / views count shown below the post.
   c. Note the number of reactions (likes, celebrates, etc.).
   d. Note the number of comments.
   e. Note the number of reposts.
   f. Note the publish date/time if visible.
5. Also check https://www.linkedin.com/analytics/creator/ if accessible for
   more detailed dashboard numbers — record them if found.
6. Return all data in the structured AnalyticsReport format.
   Set scraped_at to today's date/time: {datetime.utcnow().isoformat()}
"""
    browser = _build_browser(cfg)
    agent = Agent(
        task=task,
        llm=_build_llm(cfg),
        browser=browser,
        controller=controller,
        use_vision=True,
        max_failures=3,
    )
    history = await agent.run(max_steps=cfg.max_steps)
    await browser.close()

    raw = history.final_result()
    if not raw:
        logger.warning("monitor_analytics returned no data.")
        return AnalyticsReport(scraped_at=datetime.utcnow().isoformat(), posts=[])

    report = AnalyticsReport.model_validate_json(raw)
    _save_analytics(report, cfg)
    return report


# ---------------------------------------------------------------------------
# Mode: profile_update
# ---------------------------------------------------------------------------

async def run_profile_update(cfg: LinkedInConfig) -> None:
    """Update your LinkedIn headline, about section, and featured section."""
    if not cfg.your_headline:
        raise ValueError("Set cfg.your_headline before using profile_update mode.")

    task = f"""
You are managing the LinkedIn profile of {cfg.your_name}.

Your tasks:
1. Go to {cfg.your_profile_url} and click "Edit profile" (pencil icon).
2. Update the HEADLINE field to:
   "{cfg.your_headline}"
   Click Save.

3. Scroll down to the "About" section and click its edit (pencil) icon.
   Rewrite it to be:
   - 3-4 short paragraphs
   - Focus on who you help, the problem you solve, and your unique approach
   - End with a clear CTA: "DM me if you want to explore how I can help."
   - Incorporate these keywords naturally: {", ".join(cfg.target_keywords)}
   Click Save.

4. Scroll to the "Featured" section. If it exists:
   - Ensure your most important post or link is the first featured item.
   - Remove any outdated featured items.
   If it does not exist, skip this step.

5. Confirm all changes are saved and visible on your public profile.

Important:
- Do NOT change your profile photo or banner.
- Do NOT alter your work experience entries.
- Save each section individually before moving to the next.
"""
    browser = _build_browser(cfg)
    agent = Agent(
        task=task,
        llm=_build_llm(cfg),
        browser=browser,
        use_vision=True,
        max_failures=3,
    )
    history = await agent.run(max_steps=cfg.max_steps)
    logger.info("profile_update result: %s", history.final_result())
    await browser.close()


# ---------------------------------------------------------------------------
# Mode: full_campaign  (orchestrates multiple modes in sequence)
# ---------------------------------------------------------------------------

async def run_full_campaign(cfg: LinkedInConfig) -> None:
    """
    End-to-end marketing & sales campaign:
      1. Post content to establish authority
      2. Find new leads
      3. Send connection requests
      4. Engage with community content
      5. Report analytics
    """
    logger.info("=== Starting Full LinkedIn Campaign ===")

    if cfg.post_topic:
        logger.info("Step 1/5 — Posting content...")
        await run_post_content(cfg)
    else:
        logger.info("Step 1/5 — Skipped (post_topic not set)")

    logger.info("Step 2/5 — Finding leads...")
    lead_list = await run_find_leads(cfg)
    logger.info("Found %d leads.", len(lead_list.leads))

    logger.info("Step 3/5 — Sending connection requests...")
    await run_send_connections(cfg)

    logger.info("Step 4/5 — Engaging with content...")
    await run_engage_content(cfg)

    logger.info("Step 5/5 — Monitoring analytics...")
    report = await run_monitor_analytics(cfg)
    logger.info("Analytics: %d posts analysed.", len(report.posts))

    logger.info("=== Campaign Complete ===")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

MODES = {
    "post_content": run_post_content,
    "find_leads": run_find_leads,
    "send_connections": run_send_connections,
    "send_messages": run_send_messages,
    "engage_content": run_engage_content,
    "monitor_analytics": run_monitor_analytics,
    "profile_update": run_profile_update,
    "full_campaign": run_full_campaign,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous LinkedIn AI Marketing & Sales Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {m}" for m in MODES),
    )
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default="find_leads",
        help="Which agent mode to run (default: find_leads)",
    )
    parser.add_argument("--topic", help="Post topic for post_content mode")
    parser.add_argument("--keywords", help="Comma-separated target keywords")
    parser.add_argument("--location", help="Target location for lead search")
    parser.add_argument("--model", default="gpt-4o", help="LLM model name")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--headless", action="store_true", help="Run browser headlessly")
    parser.add_argument("--max-steps", type=int, default=80)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    cfg = LinkedInConfig(
        llm_provider=args.provider,
        model=args.model,
        headless=args.headless,
        max_steps=args.max_steps,
    )

    if args.topic:
        cfg.post_topic = args.topic
    if args.keywords:
        cfg.target_keywords = [k.strip() for k in args.keywords.split(",")]
    if args.location:
        cfg.target_location = args.location

    mode_fn = MODES[args.mode]
    result = await mode_fn(cfg)

    # Pretty-print structured results
    if isinstance(result, LeadList):
        print(f"\n{'='*50}")
        print(f"Found {len(result.leads)} leads:")
        for lead in result.leads:
            print(f"  • {lead.name} — {lead.headline} | {lead.profile_url}")
    elif isinstance(result, AnalyticsReport):
        print(f"\n{'='*50}")
        print(f"Analytics report ({result.scraped_at}):")
        for post in result.posts:
            print(f"  • {post.post_url}")
            print(f"    Impressions: {post.impressions}  Reactions: {post.reactions}  "
                  f"Comments: {post.comments}")


if __name__ == "__main__":
    asyncio.run(main())
