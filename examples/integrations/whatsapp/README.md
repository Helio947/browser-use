# WhatsApp Agent

A browser-use agent that operates [WhatsApp Web](https://web.whatsapp.com) for you, with a set of
WhatsApp-specific skills so the agent doesn't have to rediscover the UI on every step.

## Skills

| Skill | What it does |
| --- | --- |
| `whatsapp_open` | Opens WhatsApp Web and reports whether it is logged in or showing a QR code |
| `whatsapp_wait_for_login` | Waits until you scan the QR code with your phone |
| `whatsapp_list_chats` | Lists chats with last message and unread count (`unread_only` to filter) |
| `whatsapp_open_chat` | Opens a chat by exact name or by phone number with country code |
| `whatsapp_read_messages` | Returns the last N messages of a chat (`→` outgoing, `←` incoming) |
| `whatsapp_send_message` | Sends a text message (multi-line supported) |
| `whatsapp_send_file` | Sends an image, video or document, with an optional caption |
| `whatsapp_search_messages` | Searches all chats and messages for a text |
| `whatsapp_get_chat_info` | Returns a chat's name and status line (online, last seen, group members) |

Use the skills in your own agent:

```python
from browser_use import Agent, Controller
from examples.integrations.whatsapp.whatsapp_skills import register_whatsapp_skills

controller = register_whatsapp_skills(Controller())
agent = Agent(task='Tell me what Mom wrote today', llm=llm, controller=controller)
```

`register_whatsapp_skills` takes two optional safety settings:

- `confirm_send`: async callback `(chat, description) -> bool`, called before every send. Return `False` to cancel.
- `allowed_chats`: list of chat names or phone numbers the agent may send to. Everything else is refused.

## Running the agent

1. Install the project: `pip install -e .` and `playwright install chromium`.
2. Put `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) in `.env`.
3. Optional but recommended: set `CHROME_PATH` to your Chrome executable, for example
   `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`. The agent then starts Chrome with a
   dedicated profile (`~/.config/browseruse/whatsapp-profile`, override with `WHATSAPP_PROFILE_DIR`) so you
   scan the QR code once and stay logged in. Without it you scan the QR code on every run.
   Close other Chrome windows that use remote debugging port 9222 first.

Run a single task:

```bash
python examples/integrations/whatsapp/whatsapp_agent.py "Send 'running 10 min late' to Mom"
python examples/integrations/whatsapp/whatsapp_agent.py "Summarize today's messages in Work Team"
python examples/integrations/whatsapp/whatsapp_agent.py --file ./invoice.pdf "Send the invoice to +4915112345678"
python examples/integrations/whatsapp/whatsapp_agent.py --model claude-sonnet-5-5 "Who has unread messages for me?"
```

Watch for unread messages and handle them:

```bash
python examples/integrations/whatsapp/whatsapp_agent.py --watch \
  --instructions "Reply that I'm on vacation until Monday and will answer then" \
  --allow "Mom" --allow "Work Team"
```

The default `--watch` instructions only summarize unread conversations and send nothing.

| Flag | Meaning |
| --- | --- |
| `--model` | `gpt-4o` by default; any `claude-*` model uses Anthropic (`WHATSAPP_AGENT_MODEL` env var also works) |
| `--watch`, `--interval` | Keep polling for unread chats every N seconds (default 60) |
| `--instructions` | What to do with unread messages in `--watch` mode |
| `--allow` | Only allow sending to these chats (repeatable) |
| `--file` | Files the agent may send (repeatable); any other path is refused |
| `--no-confirm` | Send without asking. By default every send waits for `y` in the terminal |
| `--max-steps` | Step limit per agent run (default 25) |

## Notes

- WhatsApp Web changes its markup often. All selectors are in `SELECTORS` at the top of
  `whatsapp_skills.py`, each with fallbacks. If a skill stops finding something, update it there.
- Automating a personal WhatsApp account is against WhatsApp's Terms of Service if used for bulk or
  unsolicited messaging and can get the number banned. Keep it to your own conversations, keep
  confirmations on, and use the [WhatsApp Business Platform](https://developers.facebook.com/docs/whatsapp)
  for anything at scale.
- `tests/test_whatsapp_skills.py` runs every skill against `mock_whatsapp.html`, a small stand-in for
  WhatsApp Web's DOM: `pytest tests/test_whatsapp_skills.py`.
