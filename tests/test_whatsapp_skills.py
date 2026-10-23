"""
Tests for the WhatsApp skills in examples/integrations/whatsapp, run against a local mock of
WhatsApp Web (mock_whatsapp.html) served in place of https://web.whatsapp.com.
"""

import os
from types import SimpleNamespace

import pytest
from playwright.async_api import async_playwright

from browser_use.controller.service import Controller
from examples.integrations.whatsapp.whatsapp_skills import (
	WHATSAPP_SKILLS,
	is_phone_number,
	normalize_phone,
	parse_pre_plain_text,
	register_whatsapp_skills,
)

MOCK_HTML = os.path.join(os.path.dirname(__file__), '..', 'examples', 'integrations', 'whatsapp', 'mock_whatsapp.html')
CHROMIUM = os.getenv('CHROMIUM_EXECUTABLE', '/opt/pw-browsers/chromium')


class FakeBrowserContext:
	"""The subset of browser_use's BrowserContext that the WhatsApp skills use."""

	def __init__(self, context, page):
		self.session = SimpleNamespace(context=context)
		self.page = page

	async def get_session(self):
		return self.session

	async def get_current_page(self):
		return self.page


@pytest.fixture
async def whatsapp(request):
	qr = getattr(request, 'param', False)
	with open(MOCK_HTML) as f:
		html = f.read()
	async with async_playwright() as p:
		launch_args = {'executable_path': CHROMIUM} if os.path.exists(CHROMIUM) else {}
		browser = await p.chromium.launch(headless=True, **launch_args)
		context = await browser.new_context()

		async def serve(route):
			await route.fulfill(status=200, content_type='text/html', body=html)

		await context.route('https://web.whatsapp.com/**', serve)
		page = await context.new_page()
		if qr:
			await page.goto('https://web.whatsapp.com/?qr=1')
		yield FakeBrowserContext(context, page)
		await browser.close()


@pytest.fixture
def controller():
	return register_whatsapp_skills(Controller())


async def run(controller, name, params=None, browser=None, files=None):
	return await controller.registry.execute_action(name, params or {}, browser=browser, available_file_paths=files)


def test_all_skills_registered(controller):
	for name in WHATSAPP_SKILLS:
		assert name in controller.registry.registry.actions
	description = controller.registry.get_prompt_description()
	assert 'whatsapp_send_message' in description


def test_parse_pre_plain_text():
	assert parse_pre_plain_text('[10:21, 23/10/2026] John Doe: ') == ('10:21', '23/10/2026', 'John Doe')
	assert parse_pre_plain_text('[9:05 PM, 10/23/2026] +1 555 123: ') == ('9:05 PM', '10/23/2026', '+1 555 123')
	assert parse_pre_plain_text('garbage') == ('', '', 'garbage')


def test_phone_helpers():
	assert is_phone_number('+1 555 123 4567')
	assert is_phone_number('4915112345678')
	assert not is_phone_number('Mom')
	assert not is_phone_number('Team 2024')
	assert normalize_phone('+1 (555) 123-4567') == '15551234567'


async def test_open_logged_in(controller, whatsapp):
	result = await run(controller, 'whatsapp_open', browser=whatsapp)
	assert 'logged in' in result.extracted_content


@pytest.mark.parametrize('whatsapp', [True], indirect=True)
async def test_open_shows_qr(controller, whatsapp):
	result = await run(controller, 'whatsapp_open', browser=whatsapp)
	assert 'QR code' in result.extracted_content
	with pytest.raises(RuntimeError, match='not logged in'):
		await run(controller, 'whatsapp_list_chats', browser=whatsapp)


async def test_list_chats(controller, whatsapp):
	result = await run(controller, 'whatsapp_list_chats', browser=whatsapp)
	assert '- Mom (2 unread): Bring bread please' in result.extracted_content
	assert '- Work Team: Thanks!' in result.extracted_content

	unread = await run(controller, 'whatsapp_list_chats', {'unread_only': True}, browser=whatsapp)
	assert 'Mom' in unread.extracted_content
	assert 'Work Team' not in unread.extracted_content


async def test_read_messages(controller, whatsapp):
	result = await run(controller, 'whatsapp_read_messages', {'chat': 'Work Team'}, browser=whatsapp)
	assert '← [23/10/2026 09:00] Alice: Standup moved to 9:30' in result.extracted_content
	assert '→ [23/10/2026 09:05] Me: Thanks!' in result.extracted_content

	last = await run(controller, 'whatsapp_read_messages', {'limit': 1}, browser=whatsapp)
	assert 'Alice' not in last.extracted_content
	assert 'Thanks!' in last.extracted_content


async def test_send_message(controller, whatsapp):
	result = await run(controller, 'whatsapp_send_message', {'chat': 'Mom', 'text': 'Yes!\nBringing bread'}, browser=whatsapp)
	assert 'Sent WhatsApp message to "Mom"' in result.extracted_content
	sent = await whatsapp.page.evaluate('window.sent')
	assert sent == [{'chat': 'Mom', 'text': 'Yes!\nBringing bread', 'kind': 'text'}]


async def test_send_message_by_phone(controller, whatsapp):
	await run(controller, 'whatsapp_open', browser=whatsapp)
	await run(controller, 'whatsapp_send_message', {'chat': '+1 555 123 4567', 'text': 'Hi'}, browser=whatsapp)
	sent = await whatsapp.page.evaluate('window.sent')
	assert sent == [{'chat': '+1 555 123 4567', 'text': 'Hi', 'kind': 'text'}]


async def test_send_file_with_caption(controller, whatsapp, tmp_path):
	photo = tmp_path / 'photo.jpg'
	photo.write_bytes(b'\xff\xd8\xff')
	params = {'chat': 'Mom', 'file_path': str(photo), 'caption': 'Dinner'}
	result = await run(controller, 'whatsapp_send_file', params, browser=whatsapp, files=[str(photo)])
	assert 'Sent photo.jpg to "Mom"' in result.extracted_content
	sent = await whatsapp.page.evaluate('window.sent')
	assert sent == [{'chat': 'Mom', 'text': 'photo.jpg | Dinner', 'kind': 'media'}]


async def test_send_file_rejects_unlisted_path(controller, whatsapp, tmp_path):
	doc = tmp_path / 'secret.pdf'
	doc.write_bytes(b'%PDF')
	result = await run(
		controller, 'whatsapp_send_file', {'chat': 'Mom', 'file_path': str(doc)}, browser=whatsapp, files=['other']
	)
	assert 'not in the available file paths' in result.error


async def test_search_messages(controller, whatsapp):
	result = await run(controller, 'whatsapp_search_messages', {'query': 'standup'}, browser=whatsapp)
	assert 'Work Team' in result.extracted_content
	assert 'Mom' not in result.extracted_content


async def test_chat_info(controller, whatsapp):
	result = await run(controller, 'whatsapp_get_chat_info', {'chat': 'Work Team'}, browser=whatsapp)
	assert 'Work Team' in result.extracted_content
	assert 'Alice, Bob, You' in result.extracted_content


async def test_allowlist_blocks_other_chats(whatsapp):
	controller = register_whatsapp_skills(Controller(), allowed_chats=['Mom'])
	with pytest.raises(RuntimeError, match='not allowed'):
		await run(controller, 'whatsapp_send_message', {'chat': 'Work Team', 'text': 'hi'}, browser=whatsapp)
	await run(controller, 'whatsapp_send_message', {'chat': 'mom', 'text': 'hi'}, browser=whatsapp)
	assert len(await whatsapp.page.evaluate('window.sent')) == 1


async def test_confirmation_can_decline(whatsapp):
	asked = []

	async def decline(chat, description):
		asked.append((chat, description))
		return False

	controller = register_whatsapp_skills(Controller(), confirm_send=decline)
	result = await run(controller, 'whatsapp_send_message', {'chat': 'Mom', 'text': 'hi'}, browser=whatsapp)
	assert 'declined' in result.error
	assert asked == [('Mom', 'message: hi')]
	assert not await whatsapp.page.evaluate('window.sent')
