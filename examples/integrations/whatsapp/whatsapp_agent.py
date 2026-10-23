"""
WhatsApp agent: drive WhatsApp Web with browser-use and the WhatsApp skills.

Run a single task:
	python examples/integrations/whatsapp/whatsapp_agent.py "Send 'running 10 min late' to Mom"

Watch for unread messages and answer them following your instructions:
	python examples/integrations/whatsapp/whatsapp_agent.py --watch --instructions "Say I am away until Monday"

@dev Set OPENAI_API_KEY (or ANTHROPIC_API_KEY with --model claude-...) in your .env file.
Set CHROME_PATH to your Chrome executable to keep the WhatsApp login between runs.
"""

import argparse
import asyncio
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

from browser_use import Agent, Browser, BrowserConfig, Controller
from browser_use.browser.context import BrowserContext
from examples.integrations.whatsapp.whatsapp_skills import register_whatsapp_skills

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_PROFILE_DIR = os.path.expanduser('~/.config/browseruse/whatsapp-profile')

WHATSAPP_SYSTEM_MESSAGE = """
You operate WhatsApp Web for the user. Always prefer the whatsapp_* actions over clicking around:
- Start with whatsapp_open. If it reports a QR code, call whatsapp_wait_for_login.
- Use whatsapp_read_messages before replying so your answer fits the conversation.
- Use whatsapp_send_message / whatsapp_send_file to send; never type into the message box by hand.
- Use exact chat names. If you are unsure which chat is meant, use whatsapp_list_chats or whatsapp_search_messages first.
- Never send anything the task did not ask for. If a send is declined, do not retry it.
"""


def get_llm(model: str) -> BaseChatModel:
	if model.startswith('claude'):
		from langchain_anthropic import ChatAnthropic

		return ChatAnthropic(model_name=model, temperature=0.0, timeout=60, stop=None)
	from langchain_openai import ChatOpenAI

	return ChatOpenAI(model=model, temperature=0.0)


def get_browser(headless: bool) -> Browser:
	chrome_path = os.getenv('CHROME_PATH')
	if chrome_path:
		# A real Chrome with its own profile keeps the WhatsApp session, so the QR code is only scanned once.
		profile_dir = os.getenv('WHATSAPP_PROFILE_DIR', DEFAULT_PROFILE_DIR)
		os.makedirs(profile_dir, exist_ok=True)
		return Browser(
			config=BrowserConfig(
				chrome_instance_path=chrome_path,
				extra_chromium_args=[f'--user-data-dir={profile_dir}'],
			)
		)
	if headless:
		logger.warning('Headless mode without CHROME_PATH: you cannot scan the QR code unless already logged in.')
	return Browser(config=BrowserConfig(headless=headless))


async def ask_confirmation(chat: str, description: str) -> bool:
	answer = await asyncio.to_thread(input, f'\n❓ Send to "{chat}" → {description}\n   Send? [y/N] ')
	return answer.strip().lower() in ('y', 'yes')


def build_controller(confirm: bool, allowed_chats: list[str] | None) -> Controller:
	controller = Controller()
	return register_whatsapp_skills(
		controller,
		confirm_send=ask_confirmation if confirm else None,
		allowed_chats=allowed_chats,
	)


async def run_task(
	task: str, llm: BaseChatModel, context: BrowserContext, controller: Controller, files: list[str], max_steps: int
):
	agent = Agent(
		task=task,
		llm=llm,
		browser_context=context,
		controller=controller,
		extend_system_message=WHATSAPP_SYSTEM_MESSAGE,
		available_file_paths=files or None,
		initial_actions=[{'whatsapp_open': {}}],
	)
	history = await agent.run(max_steps=max_steps)
	return history.final_result()


async def watch(
	instructions: str,
	llm: BaseChatModel,
	context: BrowserContext,
	controller: Controller,
	files: list[str],
	interval: int,
	max_steps: int,
):
	"""Poll for unread chats and start an agent run whenever new messages arrive."""
	result = await controller.registry.execute_action('whatsapp_wait_for_login', {'timeout_seconds': 300}, browser=context)
	if result.error:
		raise RuntimeError(result.error)

	logger.info(f'👀 Watching WhatsApp for unread messages every {interval}s (Ctrl+C to stop)')
	while True:
		try:
			result = await controller.registry.execute_action(
				'whatsapp_list_chats', {'unread_only': True, 'limit': 10}, browser=context
			)
		except RuntimeError as e:
			logger.warning(f'Unread check failed, retrying next round: {e}')
			await asyncio.sleep(interval)
			continue

		if result.extracted_content and not result.extracted_content.startswith('No unread'):
			logger.info(result.extracted_content)
			task = (
				'Handle my unread WhatsApp messages. These chats have unread messages:\n'
				f'{result.extracted_content}\n\n'
				'For each chat: read the latest messages, then act according to these instructions:\n'
				f'{instructions}'
			)
			await run_task(task, llm, context, controller, files, max_steps)
		await asyncio.sleep(interval)


async def main():
	parser = argparse.ArgumentParser(description='WhatsApp agent powered by browser-use')
	parser.add_argument('task', nargs='?', help='What the agent should do on WhatsApp')
	parser.add_argument('--model', default=os.getenv('WHATSAPP_AGENT_MODEL', 'gpt-4o'))
	parser.add_argument('--watch', action='store_true', help='Keep running and handle unread messages')
	parser.add_argument('--instructions', default='Summarize each unread conversation for me. Do not send anything.')
	parser.add_argument('--interval', type=int, default=60, help='Seconds between unread checks in --watch mode')
	parser.add_argument('--allow', action='append', help='Only allow sending to this chat/phone (repeatable)')
	parser.add_argument('--file', action='append', default=[], help='File the agent may send (repeatable)')
	parser.add_argument('--no-confirm', action='store_true', help='Send without asking for confirmation')
	parser.add_argument('--headless', action='store_true')
	parser.add_argument('--max-steps', type=int, default=25)
	args = parser.parse_args()

	if not args.task and not args.watch:
		parser.error('pass a task or --watch')

	llm = get_llm(args.model)
	browser = get_browser(args.headless)
	controller = build_controller(confirm=not args.no_confirm, allowed_chats=args.allow)
	files = [os.path.abspath(f) for f in args.file]
	context = await browser.new_context()

	try:
		if args.watch:
			await watch(args.instructions, llm, context, controller, files, args.interval, args.max_steps)
		else:
			result = await run_task(args.task, llm, context, controller, files, args.max_steps)
			print(f'\n✅ Result:\n{result}')
	finally:
		await context.close()
		await browser.close()


if __name__ == '__main__':
	asyncio.run(main())
