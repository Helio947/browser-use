"""
WhatsApp Web skills for browser-use.

Registers deterministic WhatsApp Web actions on a browser-use Controller, so the agent
does not have to rediscover the WhatsApp UI on every step:

- whatsapp_open: open web.whatsapp.com and report whether the session is logged in
- whatsapp_wait_for_login: wait until the QR code has been scanned
- whatsapp_list_chats: list chats from the sidebar (optionally only unread ones)
- whatsapp_open_chat: open a chat by contact/group name or phone number
- whatsapp_read_messages: read the latest messages of a chat
- whatsapp_send_message: send a text message to a chat
- whatsapp_send_file: send a file (image, video, document) with an optional caption
- whatsapp_search_messages: search all chats for a text
- whatsapp_get_chat_info: get the header details (name, status) of a chat

WhatsApp Web changes its markup often. All selectors live in SELECTORS so they can be
updated in one place; each entry is a list of fallbacks tried in order.
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from playwright.async_api import Locator, Page
from pydantic import BaseModel, Field

from browser_use.agent.views import ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.service import Controller

logger = logging.getLogger(__name__)

WHATSAPP_URL = 'https://web.whatsapp.com/'

SELECTORS: dict[str, list[str]] = {
	'chat_list': ['#pane-side', '[aria-label="Chat list"]'],
	'qr_code': ['canvas[aria-label*="Scan"]', 'div[data-ref] canvas', '[data-ref]'],
	'search_box': [
		'#side div[contenteditable="true"][role="textbox"]',
		'#side div[contenteditable="true"]',
		'div[contenteditable="true"][data-tab="3"]',
	],
	'chat_rows': ['#pane-side [role="listitem"]', '#pane-side [role="row"]'],
	'chat_title': ['span[title]'],
	'unread_badge': ['span[aria-label*="unread" i]'],
	'chat_header': ['#main header'],
	'composer': [
		'#main footer div[contenteditable="true"][role="textbox"]',
		'#main footer div[contenteditable="true"]',
		'div[contenteditable="true"][data-tab="10"]',
	],
	'message_rows': ['#main div.copyable-text[data-pre-plain-text]'],
	'message_text': ['span.selectable-text', 'span[dir]'],
	'attach_button': [
		'#main footer [aria-label="Attach"]',
		'#main footer [title="Attach"]',
		'#main footer span[data-icon="plus"]',
		'#main footer span[data-icon="clip"]',
	],
	'file_input_media': ['input[type="file"][accept*="image"]'],
	'file_input_document': ['input[type="file"][accept="*"]', 'input[type="file"]:not([accept])', 'input[type="file"]'],
	'media_caption': [
		'div[contenteditable="true"][aria-label*="caption" i]',
		'div[aria-label*="caption" i] div[contenteditable="true"]',
	],
	'send_button': ['[aria-label="Send"]', 'span[data-icon="send"]', 'span[data-icon="wds-ic-send-filled"]'],
	'search_results': ['#pane-side [role="listitem"]', '#pane-side [role="row"]'],
	'invalid_phone_popup': ['[data-animate-modal-popup="true"]', 'div[role="dialog"]'],
}

# Format of the data-pre-plain-text attribute: "[10:21, 23/10/2026] John Doe: "
PRE_PLAIN_TEXT_RE = re.compile(r'^\[(?P<time>[^,\]]+),\s*(?P<date>[^\]]+)\]\s*(?P<sender>.*?):\s*$')
PHONE_RE = re.compile(r'^\+?[\d\s\-().]{7,}$')


# --- Parameter models -------------------------------------------------------------------


class OpenWhatsAppAction(BaseModel):
	pass


class WaitForLoginAction(BaseModel):
	timeout_seconds: int = Field(default=120, description='How long to wait for the QR code to be scanned')


class ListChatsAction(BaseModel):
	unread_only: bool = Field(default=False, description='Only return chats with unread messages')
	limit: int = Field(default=20, description='Maximum number of chats to return')


class OpenChatAction(BaseModel):
	chat: str = Field(description='Exact contact/group name as shown in WhatsApp, or a phone number with country code')


class ReadMessagesAction(BaseModel):
	chat: Optional[str] = Field(default=None, description='Chat to read; omit to read the currently open chat')
	limit: int = Field(default=20, description='Number of most recent messages to return')


class SendMessageAction(BaseModel):
	chat: str = Field(description='Exact contact/group name, or a phone number with country code')
	text: str = Field(description='Message text; newlines are kept')


class SendFileAction(BaseModel):
	chat: str = Field(description='Exact contact/group name, or a phone number with country code')
	file_path: str = Field(description='Path of the file to send; must be one of the available file paths')
	caption: Optional[str] = Field(default=None, description='Optional caption for images and videos')
	as_document: bool = Field(default=False, description='Send images/videos as a document instead of media')


class SearchMessagesAction(BaseModel):
	query: str = Field(description='Text to search for across all chats')
	limit: int = Field(default=10, description='Maximum number of results')


class ChatInfoAction(BaseModel):
	chat: Optional[str] = Field(default=None, description='Chat to inspect; omit for the currently open chat')


# --- Helpers ----------------------------------------------------------------------------


@dataclass
class WhatsAppMessage:
	sender: str
	time: str
	date: str
	text: str
	outgoing: bool

	def format(self) -> str:
		direction = '→' if self.outgoing else '←'
		return f'{direction} [{self.date} {self.time}] {self.sender}: {self.text}'


def parse_pre_plain_text(value: str) -> tuple[str, str, str]:
	"""Split WhatsApp's data-pre-plain-text attribute into (time, date, sender)."""
	match = PRE_PLAIN_TEXT_RE.match(value.strip() + ' ')
	if not match:
		return '', '', value.strip().rstrip(':')
	return match['time'].strip(), match['date'].strip(), match['sender'].strip()


def is_phone_number(value: str) -> bool:
	return bool(PHONE_RE.match(value.strip())) and sum(c.isdigit() for c in value) >= 7


def normalize_phone(value: str) -> str:
	return ''.join(c for c in value if c.isdigit())


async def _first(page: Page | Locator, key: str, timeout: float = 0) -> Optional[Locator]:
	"""Return the first visible locator among the fallbacks for `key`, waiting up to `timeout` ms."""
	deadline = asyncio.get_event_loop().time() + timeout / 1000
	while True:
		for selector in SELECTORS[key]:
			locator = page.locator(selector).first
			try:
				if await locator.count() and await locator.is_visible():
					return locator
			except Exception:
				continue
		if asyncio.get_event_loop().time() >= deadline:
			return None
		await asyncio.sleep(0.25)


async def _all(page: Page | Locator, key: str) -> list[Locator]:
	for selector in SELECTORS[key]:
		items = await page.locator(selector).all()
		if items:
			return items
	return []


async def _attached(page: Page, key: str) -> Optional[Locator]:
	"""Like _first, but accepts hidden elements (file inputs are never visible)."""
	for selector in SELECTORS[key]:
		locator = page.locator(selector)
		if await locator.count():
			return locator.first
	return None


async def _type_multiline(page: Page, box: Locator, text: str) -> None:
	await box.click()
	lines = text.split('\n')
	for i, line in enumerate(lines):
		if line:
			await page.keyboard.insert_text(line)
		if i < len(lines) - 1:
			await page.keyboard.press('Shift+Enter')


async def _get_whatsapp_page(browser: BrowserContext) -> Page:
	"""Return the WhatsApp tab, opening WhatsApp Web in the current tab if none is open."""
	session = await browser.get_session()
	for page in session.context.pages:
		if page.url.startswith(WHATSAPP_URL):
			if page != await browser.get_current_page():
				await page.bring_to_front()
			return page
	page = await browser.get_current_page()
	await page.goto(WHATSAPP_URL)
	return page


async def _login_state(page: Page, timeout: float = 30000) -> str:
	"""Return 'logged_in', 'qr' or 'loading'."""
	deadline = asyncio.get_event_loop().time() + timeout / 1000
	while True:
		if await _first(page, 'chat_list'):
			return 'logged_in'
		if await _first(page, 'qr_code'):
			return 'qr'
		if asyncio.get_event_loop().time() >= deadline:
			return 'loading'
		await asyncio.sleep(0.5)


async def _ensure_logged_in(page: Page) -> None:
	state = await _login_state(page)
	if state != 'logged_in':
		raise RuntimeError('WhatsApp Web is not logged in. Call whatsapp_wait_for_login and ask the user to scan the QR code.')


async def _open_chat(page: Page, chat: str) -> str:
	"""Open a chat by name or phone number. Returns the chat title shown in the header."""
	await _ensure_logged_in(page)

	if is_phone_number(chat):
		await page.goto(f'{WHATSAPP_URL}send?phone={normalize_phone(chat)}')
		composer = await _first(page, 'composer', timeout=30000)
		if not composer:
			raise RuntimeError(f'Could not open a chat with phone number {chat}; it may not be on WhatsApp.')
		return await _chat_title(page) or chat

	# Already open?
	if (await _chat_title(page) or '').strip().lower() == chat.strip().lower():
		return chat

	search = await _first(page, 'search_box', timeout=10000)
	if not search:
		raise RuntimeError('Could not find the WhatsApp search box.')
	await search.click()
	await page.keyboard.press('Control+A')
	await page.keyboard.press('Backspace')
	await page.keyboard.insert_text(chat)

	# Prefer an exact title match, fall back to the first result.
	target: Optional[Locator] = None
	for _ in range(20):
		exact = page.locator(f'#pane-side span[title="{_css_escape(chat)}"]')
		if await exact.count():
			target = exact.first
			break
		await asyncio.sleep(0.25)
	if target is None:
		rows = await _all(page, 'search_results')
		if not rows:
			raise RuntimeError(f'No chat found matching "{chat}".')
		target = rows[0]
	await target.click()

	if not await _first(page, 'composer', timeout=10000):
		raise RuntimeError(f'Opened "{chat}" but could not find the message box (is it a read-only group?).')
	return await _chat_title(page) or chat


def _css_escape(value: str) -> str:
	return value.replace('\\', '\\\\').replace('"', '\\"')


async def _chat_title(page: Page) -> Optional[str]:
	header = await _first(page, 'chat_header')
	if not header:
		return None
	title = header.locator('span[title]').first
	if await title.count():
		return await title.get_attribute('title') or await title.inner_text()
	text = (await header.inner_text()).strip()
	return text.split('\n')[0] if text else None


async def _read_messages(page: Page, limit: int) -> list[WhatsAppMessage]:
	rows = await _all(page, 'message_rows')
	messages = []
	for row in rows[-limit:]:
		pre = await row.get_attribute('data-pre-plain-text') or ''
		time, date, sender = parse_pre_plain_text(pre)
		text = ''
		for selector in SELECTORS['message_text']:
			span = row.locator(selector).first
			if await span.count():
				text = (await span.inner_text()).strip()
				break
		outgoing = await row.evaluate('el => !!el.closest(".message-out")')
		messages.append(WhatsAppMessage(sender=sender, time=time, date=date, text=text, outgoing=outgoing))
	return messages


# --- Registration -----------------------------------------------------------------------

ConfirmCallback = Callable[[str, str], Awaitable[bool]]


def register_whatsapp_skills(
	controller: Controller,
	confirm_send: Optional[ConfirmCallback] = None,
	allowed_chats: Optional[list[str]] = None,
) -> Controller:
	"""
	Register all WhatsApp skills on `controller`.

	confirm_send: optional async callback (chat, description) -> bool, called before anything is
	sent. Return False to cancel. Use it to keep a human in the loop.
	allowed_chats: optional allowlist of chat names / phone numbers the agent may send to.
	"""

	allowed = {c.strip().lower() for c in allowed_chats} if allowed_chats else None
	allowed_phones = {normalize_phone(c) for c in allowed_chats if is_phone_number(c)} if allowed_chats else set()

	def _check_allowed(chat: str) -> None:
		if allowed is None:
			return
		if chat.strip().lower() in allowed or (is_phone_number(chat) and normalize_phone(chat) in allowed_phones):
			return
		raise PermissionError(f'Sending to "{chat}" is not allowed. Allowed chats: {", ".join(sorted(allowed))}')

	async def _confirm(chat: str, description: str) -> bool:
		if confirm_send is None:
			return True
		return await confirm_send(chat, description)

	@controller.registry.action(
		'Open WhatsApp Web and report whether it is logged in. Use this first for any WhatsApp task.',
		param_model=OpenWhatsAppAction,
	)
	async def whatsapp_open(params: OpenWhatsAppAction, browser: BrowserContext):
		page = await _get_whatsapp_page(browser)
		state = await _login_state(page)
		messages = {
			'logged_in': 'WhatsApp Web is open and logged in.',
			'qr': 'WhatsApp Web shows a QR code. Call whatsapp_wait_for_login so the user can scan it with their phone.',
			'loading': 'WhatsApp Web is still loading. Try whatsapp_open again.',
		}
		return ActionResult(extracted_content=messages[state], include_in_memory=True)

	@controller.registry.action(
		'Wait for the user to scan the WhatsApp Web QR code with their phone',
		param_model=WaitForLoginAction,
	)
	async def whatsapp_wait_for_login(params: WaitForLoginAction, browser: BrowserContext):
		page = await _get_whatsapp_page(browser)
		logger.info('📱 Scan the QR code in the browser window with WhatsApp on your phone (Settings → Linked devices)')
		if not await _first(page, 'chat_list', timeout=params.timeout_seconds * 1000):
			return ActionResult(error=f'Not logged in after {params.timeout_seconds}s.', include_in_memory=True)
		return ActionResult(extracted_content='Logged in to WhatsApp Web.', include_in_memory=True)

	@controller.registry.action(
		'List WhatsApp chats from the sidebar with their last message and unread count',
		param_model=ListChatsAction,
	)
	async def whatsapp_list_chats(params: ListChatsAction, browser: BrowserContext):
		page = await _get_whatsapp_page(browser)
		await _ensure_logged_in(page)
		lines = []
		for row in await _all(page, 'chat_rows'):
			title_el = row.locator('span[title]').first
			if not await title_el.count():
				continue
			title = await title_el.get_attribute('title') or ''
			unread = 0
			badge = row.locator(SELECTORS['unread_badge'][0]).first
			if await badge.count():
				digits = normalize_phone(await badge.get_attribute('aria-label') or await badge.inner_text())
				unread = int(digits) if digits else 1
			if params.unread_only and not unread:
				continue
			preview_el = row.locator('span[title]').nth(1)
			preview = (await preview_el.get_attribute('title') or '') if await preview_el.count() else ''
			lines.append(f'- {title}' + (f' ({unread} unread)' if unread else '') + (f': {preview}' if preview else ''))
			if len(lines) >= params.limit:
				break
		if not lines:
			content = 'No unread chats.' if params.unread_only else 'No chats found.'
		else:
			content = 'WhatsApp chats:\n' + '\n'.join(lines)
		return ActionResult(extracted_content=content, include_in_memory=True)

	@controller.registry.action(
		'Open a WhatsApp chat by exact contact/group name or by phone number with country code',
		param_model=OpenChatAction,
	)
	async def whatsapp_open_chat(params: OpenChatAction, browser: BrowserContext):
		page = await _get_whatsapp_page(browser)
		title = await _open_chat(page, params.chat)
		return ActionResult(extracted_content=f'Opened WhatsApp chat "{title}".', include_in_memory=True)

	@controller.registry.action(
		'Read the most recent messages of a WhatsApp chat (→ outgoing, ← incoming)',
		param_model=ReadMessagesAction,
	)
	async def whatsapp_read_messages(params: ReadMessagesAction, browser: BrowserContext):
		page = await _get_whatsapp_page(browser)
		await _ensure_logged_in(page)
		title = await _open_chat(page, params.chat) if params.chat else await _chat_title(page)
		if not title:
			return ActionResult(error='No chat is open. Pass the chat name.', include_in_memory=True)
		messages = await _read_messages(page, params.limit)
		if not messages:
			content = f'No messages visible in "{title}".'
		else:
			content = f'Last {len(messages)} messages in "{title}":\n' + '\n'.join(m.format() for m in messages)
		return ActionResult(extracted_content=content, include_in_memory=True)

	@controller.registry.action(
		'Send a WhatsApp text message to a contact/group name or phone number',
		param_model=SendMessageAction,
	)
	async def whatsapp_send_message(params: SendMessageAction, browser: BrowserContext):
		_check_allowed(params.chat)
		if not params.text.strip():
			return ActionResult(error='Message text is empty.', include_in_memory=True)
		if not await _confirm(params.chat, f'message: {params.text}'):
			return ActionResult(error=f'User declined sending the message to "{params.chat}".', include_in_memory=True)

		page = await _get_whatsapp_page(browser)
		title = await _open_chat(page, params.chat)
		composer = await _first(page, 'composer', timeout=10000)
		if not composer:
			return ActionResult(error='Could not find the message box.', include_in_memory=True)
		await _type_multiline(page, composer, params.text)
		await page.keyboard.press('Enter')
		msg = f'📤 Sent WhatsApp message to "{title}": {params.text}'
		logger.info(msg)
		return ActionResult(extracted_content=msg, include_in_memory=True)

	@controller.registry.action(
		'Send a file (image, video or document) on WhatsApp with an optional caption',
		param_model=SendFileAction,
	)
	async def whatsapp_send_file(params: SendFileAction, browser: BrowserContext, available_file_paths: list[str]):
		_check_allowed(params.chat)
		if params.file_path not in available_file_paths:
			return ActionResult(
				error=f'File {params.file_path} is not in the available file paths: {available_file_paths}',
				include_in_memory=True,
			)
		if not os.path.isfile(params.file_path):
			return ActionResult(error=f'File {params.file_path} does not exist.', include_in_memory=True)
		if not await _confirm(params.chat, f'file: {params.file_path}'):
			return ActionResult(error=f'User declined sending the file to "{params.chat}".', include_in_memory=True)

		page = await _get_whatsapp_page(browser)
		title = await _open_chat(page, params.chat)

		attach = await _first(page, 'attach_button', timeout=5000)
		if not attach:
			return ActionResult(error='Could not find the attach button.', include_in_memory=True)
		await attach.click()

		is_media = params.file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.mp4', '.3gp', '.mov'))
		key = 'file_input_media' if is_media and not params.as_document else 'file_input_document'
		file_input = None
		for _ in range(20):
			file_input = await _attached(page, key)
			if file_input:
				break
			await asyncio.sleep(0.25)
		if not file_input:
			return ActionResult(error='Could not find the file input.', include_in_memory=True)
		await file_input.set_input_files(params.file_path)

		if params.caption:
			caption_box = await _first(page, 'media_caption', timeout=10000)
			if caption_box:
				await _type_multiline(page, caption_box, params.caption)
			else:
				logger.warning('Caption box not found, sending without caption')

		button = await _first(page, 'send_button', timeout=15000)
		if not button:
			return ActionResult(error='Could not find the send button after attaching the file.', include_in_memory=True)
		await button.click()
		msg = f'📎 Sent {os.path.basename(params.file_path)} to "{title}"'
		logger.info(msg)
		return ActionResult(extracted_content=msg, include_in_memory=True)

	@controller.registry.action(
		'Search all WhatsApp chats and messages for a text',
		param_model=SearchMessagesAction,
	)
	async def whatsapp_search_messages(params: SearchMessagesAction, browser: BrowserContext):
		page = await _get_whatsapp_page(browser)
		await _ensure_logged_in(page)
		search = await _first(page, 'search_box', timeout=10000)
		if not search:
			return ActionResult(error='Could not find the WhatsApp search box.', include_in_memory=True)
		await search.click()
		await page.keyboard.press('Control+A')
		await page.keyboard.press('Backspace')
		await page.keyboard.insert_text(params.query)
		await asyncio.sleep(1.5)
		results = []
		for row in (await _all(page, 'search_results'))[: params.limit]:
			text = ' '.join((await row.inner_text()).split())
			if text:
				results.append(f'- {text}')
		content = (
			f'Search results for "{params.query}":\n' + '\n'.join(results) if results else f'No results for "{params.query}".'
		)
		return ActionResult(extracted_content=content, include_in_memory=True)

	@controller.registry.action(
		'Get the name and status line (online, last seen, group members) of a WhatsApp chat',
		param_model=ChatInfoAction,
	)
	async def whatsapp_get_chat_info(params: ChatInfoAction, browser: BrowserContext):
		page = await _get_whatsapp_page(browser)
		await _ensure_logged_in(page)
		if params.chat:
			await _open_chat(page, params.chat)
		header = await _first(page, 'chat_header')
		if not header:
			return ActionResult(error='No chat is open. Pass the chat name.', include_in_memory=True)
		lines = [line.strip() for line in (await header.inner_text()).split('\n') if line.strip()]
		content = 'Chat info: ' + ' | '.join(lines) if lines else 'No chat info available.'
		return ActionResult(extracted_content=content, include_in_memory=True)

	return controller


WHATSAPP_SKILLS = [
	'whatsapp_open',
	'whatsapp_wait_for_login',
	'whatsapp_list_chats',
	'whatsapp_open_chat',
	'whatsapp_read_messages',
	'whatsapp_send_message',
	'whatsapp_send_file',
	'whatsapp_search_messages',
	'whatsapp_get_chat_info',
]
