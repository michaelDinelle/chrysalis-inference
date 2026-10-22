# <img src="web/logo.svg" alt="" height="32"> Chrysalis Inference

An inference engine written for exactly one model on exactly one GPU: **Qwen3.8-27B** on a
**Radeon RX 7900 XTX** (gfx1100, 24 GB). It serves an OpenAI-compatible API (and Anthropic's and OpenAI's newer
ones) built for coding agents such as OpenCode: full 262K context, prompt caching that makes every agent turn an
append, MTP speculative decoding, and pictures (the model's vision tower, without giving up any of
that VRAM; see [Pictures](#pictures)). A second engine runs **Qwen3.8-Flash-Next** (a 48-layer,
512-expert MoE) split between the GPU and system RAM; requests pick the model and Chrysalis switches
(see [Qwen3.8-Flash-Next](#qwen38-flash-next)).

About 6,100 lines: C++/HIP kernels and model code in `engine/` (3,500), the Python server in `chrysalis/` (2,600).

## Use it

```sh
./chrysalis.sh install        # once: systemd user services + ~/.local/bin/chrysalis (does not enable autostart)

chrysalis start               # loads the model (~1.5 s), starts the web chat, prints both URLs:
                              #   API http://127.0.0.1:8080/v1, web chat https://localhost:8081
chrysalis start --lan         # the same, also on the local network: prints its URLs and a QR code that connects a phone
chrysalis key                 # (with --lan) a key for an app on another computer to use the API with
chrysalis stop                # stops both and frees the GPU (also: restart, status)
chrysalis logs                # one line per request: prompt/cached tokens, TTFT, tok/s, draft acceptance
```

The web chat is HTTPS always, on this machine and on the network. `chrysalis start` makes this computer's own browsers
trust its authority, as mkcert does (Chromium's store and each Firefox profile's, by NSS's `certutil`; no password; a
browser open since before then trusts it once it restarts). `--lan` puts the web chat and the API on the local network
too, for the devices it lets in, and asks for the password once to let that network (and nothing outside it) through
ufw, when ufw is on (TCP 8081-8083; another firewall needs the same); a plain `chrysalis start` or `chrysalis restart`
goes back to this machine only. Chrysalis itself (`127.0.0.1:8080`) listens on this machine only: the network reaches
its API through the web chat, over HTTPS at `https://<address>:8081/v1`, behind the same keys as the chat. A browser let
in has its own; an app on another computer sends a key made here (Connect a device, or `chrysalis key`, shown once) as
OpenAI's SDKs send theirs (`Authorization: Bearer`) or Anthropic's (`x-api-key`), and trusts this machine's authority
(its connect page, or `chrysalis-ca.crt` as `NODE_EXTRA_CA_CERTS` or `SSL_CERT_FILE`). A reply streams through as it
comes, and an app that hangs up stops it. The web chat is `https://localhost:8081` here and `https://<address>:8081` on
the network (browsers give the microphone, voice mode, only to secure pages, which `http://<address>` is not). Its
certificate comes from an authority of this machine's own (`~/.config/chrysalis/tls`, made by `web/tls.ts`) whose name
constraints let it vouch only for private addresses and local names, so trusting it trusts nothing else. Only the
devices this computer let in get past the chat's pages to its replies, tools and voice (`web/devices.ts`), and a browser
only while it shows it trusts that authority, this computer's own too. No server can see whether a browser checked the
certificate it was shown, and a page loads all the same past a warning clicked through. So a trust check listens on a
port of its own (`https://<address>:8083` and here, `CHRYSALIS_CHECK_PORT`) with a certificate of its own, from the same
authority and made anew at each start: a browser's request reaches it only where the browser trusts the authority
itself, since a warning clicked past covers the one certificate it was shown (per certificate in Chromium, per port too
in Firefox), and an image cannot show a warning. A browser has to come through it to ask to join, and again every few
minutes and after each restart (its chat does so by itself); past the warning, Chromium and Firefox get the connect page
at the chat's address and none of the chat's own files (`browser.test.ts`, which drives both). Connect a device (the
phone at the top of the chat here) and `chrysalis status` show a QR code of the chat's address, with no secret in it. A
new device's browser warns there once, as it does not trust the authority yet; past the warning, the connect page (a
card on the chat's sea, the butterfly above it and the aurora's night sky over the page) offers the authority the way
the device's system trusts one (a configuration profile on iPhone, iPad and Mac, the certificate on Android, Windows,
Linux and ChromeOS, Firefox's own prompt in Firefox) and leads through the few taps that trust it there, one at a time
(`web/setup.ts`). It is offered over that HTTPS, as Chromium holds back every download a plain-HTTP page makes. Once the
trust check's image loads, the same page asks to join and shows a code of its own, one per device; Connect a device
shows the same code beside Allow and Deny (`chrysalis allow <code>` in a terminal), and an ask lasts 10 minutes.
Allowed, the device gets a 256-bit key of its own as a cookie (Secure, HttpOnly, SameSite=Strict, 400 days from its last
visit), kept here as its SHA-256 only (`devices.json` beside the certificates), and the chat opens in the page's place;
Connect a device lists the devices and removes one. This computer, at `localhost` and at its own network address, needs
no code, but its browser too gets the chat only once it trusts the authority (`chrysalis start` sets it up; a browser
open since before then trusts it once it restarts). What a browser gets before it trusts the authority comes over
connections that close after each answer: a browser keeps using one it went past a warning on, and marks the page not
secure on it even after it trusts the authority, so the chat opens on a new one, verified. Until a device trusts the
authority, someone on the network at that moment could answer with a page and an authority of their own: to be sure, a
device's certificate details show the SHA-256 fingerprint that Connect a device and `chrysalis status` show. A new
address (another network) gets a new certificate at the next start, from the same authority, which devices keep
trusting. Either way both servers answer only requests addressed to an IP address or to this machine's name, which keeps
web pages from other sites out (DNS rebinding). `systemctl --user enable chrysalis chrysalis-web` starts both at login.
`chrysalis serve` runs the server in the foreground instead of as a service (it takes `--port`, `--draft`, `--use`, ...:
`chrysalis serve -h`). Models: `qwen3.8-27b` (thinking, reasoning effort `xhigh` by default; OpenCode variants pick
`medium`/`low`) and `qwen3.8-27b-fast` (no thinking; also used for session titles), and the same pair for
`qwen3.8-flash-next`. One model is on the GPU at a time: a request naming the other one switches (titles never do);
`chrysalis status` says which is loaded (and the PCIe link, when it came up slow: see Notes). OpenCode reaches it as an
OpenAI-compatible provider at `http://127.0.0.1:8080/v1`, set up in `~/.config/opencode/opencode.json`.

`/metrics` on the API port is Prometheus's text format (`chrysalis/metrics.py`): counters of the replies by model and
finish reason, their prompt, cached and written tokens, prefill and decode seconds and decode steps (tok/s and tokens
per step are ratios of their rates), a time-to-first-token histogram, model loads and their seconds, error statuses; and
gauges read at the scrape: the model loaded, replies being written, free KV tokens, snapshots, the PCIe link, and on
Flash-Next the expert pool's decode picks (all, and in the pool), the file's experts out of it (in the spill mirror)
and the DMA rate.

The same server also speaks Anthropic's Messages API (`/v1/messages`, with `count_tokens`; Claude Code with
`ANTHROPIC_BASE_URL=http://127.0.0.1:8080`) and OpenAI's Responses API (`/v1/responses`; Codex), translated to the
chat completions above (`chrysalis/apis.py`): text, pictures and PDFs, tool calls and results (`tool_choice`, `strict`),
thinking (`thinking` blocks; `reasoning_text` items) and a budget as the reasoning effort and a hard limit, stop sequences,
streaming. A PDF (a chat `file` part, Anthropic's `document` block, the Responses API's `input_file`) reaches the model as
Anthropic's API gives it one: each page as its text and as a picture (1,024 pixels on the long side, ~800 tokens each;
`chrysalis/documents.py`, poppler's pdftotext and pdftoppm), the pictures going where the text cannot stand in for them:
to every page without a text layer (a scan, a drawing), up to 120 wherever they are, then to the first 20 of the others.
`page_pictures` on the file part (or the document block, or `input_file`) sets that 20, 0 for text alone; another file
reaches the model as its text. A final assistant message is a
prefill, as Anthropic's are. Only the client's own tools run: Anthropic's server tools and OpenAI's built-in ones are
refused. `previous_response_id` continues one of the last 256 responses (kept until a restart).

## Web UI

`web/` is a small Bun app: `server.ts` serves the page, proxies Chrysalis's API on the same origin and runs the tools
(`tools.ts`, with `tool-fetch.ts`, `tool-agent.ts` and `tool-mcp.ts`); `app.ts` is plain TypeScript (marked, highlight.js, DOMPurify,
KaTeX; Mermaid's own build loads with the first diagram). The chat lies on a sea of characters that fills everything
under the top bar, the message box floating on it (`ocean.ts`, two WebGL2 passes: one fragment per character cell works
out what it shows, then each screen pixel draws its cell's glyph from an atlas at the screen's own pixel size; the grid
is fixed to the window, so a side panel slides over it without redrawing it at a new size). Without WebGL2 (no GPU, or
WebGL turned off) the CPU runs the same two passes into a 2D canvas, a quarter of the time at most, and draws the same
picture: the noise's permutations and gradient choices are integers in the shader, so every GPU and the CPU make the
same ones, and the browser test holds the two drawings to each other. A small butterfly (a phone's) is drawn bolder,
its strokes three glyphs wide. A new chat opens on the
butterfly in it: the current drifts in ribbons, a click sends soft ripples across it in a shape (a heart, a ring, the
butterfly, a star, a flower) that bend the current and the butterfly as they pass, and keeps the message box's focus,
and a drag leaves a wake. Once the chat begins the butterfly sinks into the water, where its shape stays faintly, and
the sea stays behind the messages, dimmed and dimmest under their column, at 12 frames a second; a new chat raises the
butterfly again. Settings, Appearance turns the sea behind chats off ("ASCII background behind chats") and picks the
theme (the system's, light or dark; `theme.ts`, the stylesheet's colors being light-dark() pairs), both kept on the
device. With reduced motion it holds still. Both side panels' heads
have an aurora behind their names (one static SVG after T3 Code's nightly sidebar art), a night sky in either theme. Conversations live in the tab's sessionStorage: a reload keeps
them, closing the tab ends them, and nothing is stored on the server's disk. A question is kept as it is sent, before its
reply begins (with tools on, or a model loading, that takes seconds): a reload then says the reply never began. Export saves the open chat (Export all:
every chat of the tab) as a JSON file, every version, tool round and attached file included, which Import adds back
under new ids. A reply belongs to the web server while it is written (in RAM, until 15 minutes after it ends): opening
another chat, reloading, closing the tab, a laptop asleep or a dropped connection leave it running, and the page follows
it again from the last event it saw; only Stop (the button, or Esc when nothing else is open) or deleting the chat ends
it early. Replies stream with their reasoning (collapsible), each one shows time to first token, tok/s, how much of the
prompt came from cache, the sampling values that differed from the recommended ones and its seed. In the message box, an
energy bar sets the thinking: Off (the no-thinking mode), Low, Medium or Max. It grows more vivid to the right, in the
logo's pink (Medium flows; Max is brightest, with a sweeping shine and rising sparks), and dragging takes a stop a third
of the way to it. The model's name beside the send button opens the models; it is the only place a model is chosen. One
picked while a reply is being written is what the next question asks for: Chrysalis switches once that reply ends.
Beside it are the engine's state (a dot: ready, loading with its progress, or stopped), a power button (unload, or load)
and the usage: this chat against the context, and how full Chrysalis's cache of conversations is. Message actions keep
fixed places (from the right: delete, edit, copy on every message, then a reply's token odds, regenerate and continue),
so none moves between messages. Every message can be edited in place (Save; a reply's thinking too) or deleted (with
Undo). A conversation is a tree: sending an edited question (Send, Ctrl+Enter), regenerating any reply, or picking
another token makes a new version beside the old one, and "‹ 2 / 3 ›" under a message with versions (buttons, keyboard
too) switches between them, each with all that followed it. Deleting a message that has versions deletes that version.
Only the path on screen goes to Chrysalis, so switching back is a cache hit. A reply being written goes on in its
version while another is shown, and shows where it got to when its version is back; a chat writes one reply at a time.
The last reply can be continued (Continue is spelled out when the reply was cut short); it grows in place. Token odds
shows a reply as its tokens, tinted by how unsure the model was of each; a click on one lists what else it could have
written there, and picking one rewrites the reply from that token on. The wand in the message box drafts the user's next
message (or finishes the one being typed) to edit and send: Chrysalis is asked, as the assistant and without thinking,
to write it (Qwen will not write a user turn: at one it ends it). Settings (Ctrl+,) sits beside the chat: with room it
opens without moving the conversation, otherwise it slides the column over (never re-wrapping it) or, in narrow windows,
slides over it. Every change applies from the next reply and can be undone, for every conversation in the tab (kept with
them). It shows exactly what Chrysalis is sent:
- The system prompt, by default "There are no tools or files in this chat. Answer directly.", written on Chrysalis's
  card (Characters, below), which Settings opens. It is an ordinary prompt: edit or delete it. Without it, Flash-Next with thinking off takes asks like "design a cute SVG" for a coding-agent
  task ("let me first explore the project structure", then calls to tools that do not exist), as the Q8_0 original does
  too. In a chat with tools on, the default is sent as "The tools offered here are the only ones in this chat, and there
  are no files or projects to explore. Use a tool when it helps; otherwise answer directly." (a prompt of your own goes
  as written).
- Sampling: top-k, temperature, top-p and min-p, one row each in the order the sampler applies them (drag a row by its
  grip to reorder), one set with thinking and one without, starting from Qwen's recommended values. The thinking bar in
  the message box is the one switch: the panel shows (and edits) the set its level selects.
- Repetition: presence penalty and DRY (with its allowed run and growth).
Each group sits under a chart of real odds run through Chrysalis's whole sampler with the settings as they stand.
Sampling: Qwen naming a cat (the first word, 2,000 draws), a row per name with a bar for its odds now and a tick where
the model had them; min-p is a line across the bars, and a cut name is struck through with the stage that cut it.
Repetition: Qwen listing cat names after it started repeating itself ("…Luna, Milo, Shadow, Luna, Milo,", its next
token's 64 candidates): the reply line marks the names presence counts as used and the run DRY sees, and the rows show
"Shadow" (91% from the model) pushed down, tagged used and repeats. Pointing at a row explains it in the line under the
chart; explanations elsewhere are tooltips, so nothing in the panel changes size while it is used.
- The reply's max length and the seed: random for every reply, or fixed. Clicking a reply's seed fixes it.
- Tools: the config file's MCP servers as they stand (state, tools, the command or URL, read-only), Reload config, Fetch
  URL's page length, and the tools allowed to run without asking (a click makes one ask again).
A value that differs from the recommended one is marked; ↺ or a double click on its slider brings it back. The sidebar
holds the conversations (sending a message starts the model too). It and Settings are side panels that work alike: a
toggle fixed in the window's top corner (left: conversations, Ctrl+B; right: Settings, Ctrl+,), which never moves. With
room, a panel sits beside the chat, dragging its inner edge resizes it (a double click gives it its default width back;
the chat keeps at least 640 px), and it stays as it was left, width and all; in a narrower window it slides over the
chat and a click outside it or Esc closes it. Assistant turns go back to Chrysalis exactly as received, so every turn of
a conversation is a cache hit. While a reply streams the page follows it, until you scroll up (wheel, touch, keys or the
scrollbar); back at the end, or with the ↓ over the message box, it follows again, and it never scrolls up on its own. A
figure being drawn is shown from its top, as large as what can be seen at once (the column's width or the visible
height), and the page holds still while it draws. The thinking box follows its newest line the same way. Replies render
block by block: while one streams, finished paragraphs, code blocks and tables keep their nodes and only the growing
tail is rebuilt (on the next frame), and the tail shows settled text only: whole words, whole code lines and table rows,
links once their address is in, no stray `**`. A code block holding an SVG document (whatever its label) becomes a
figure that draws itself while it streams: it takes its final size from the `<svg>` tag, then each element fades in once
its tag is complete. It sits on a light page sized to the drawing, with Code, Copy, Download and Expand; the markup is
sanitized (no scripts, handlers, `foreignObject` or outside references) into a shadow root, and the page's
Content-Security-Policy lets nothing load from anywhere else. A ```mermaid block becomes a diagram in the same figure;
while it streams, each version that parses replaces the picture. TeX (`$…$`, `\(…\)`, `$$…$$`, `\[…\]`) is typeset by
KaTeX; a `$` opens math only before a non-space and closes it only after one, not before a digit or across a backtick,
so prices and shell variables stay text. A ```html block holding a whole document runs as a page once it is written (a
fragment when its Preview is clicked), in an iframe on `/sandbox` behind two walls, each enough alone: the iframe's
sandbox (an opaque origin: no cookies, storage or chat) and that route's policy, which lets the page's own code run but
nothing load or connect anywhere, Chrysalis included. Pages still get three.js and A-Frame from their usual CDN
addresses, of any version: the chat hands the page the code (the server bundles three, each addon and the classic
builds older pages load, `web/server.ts`), which the page loads from blob: addresses it makes itself (`web/sandbox.js`).
Artifacts (`web/artifacts.ts`): after the system prompt (unless
Settings leaves it out) the model is asked to name what it makes, ```` ```svg panda ```` (it also writes ```` ```svg id="panda" ````,
or the id alone on the block's first line: asked for a 3D page with thinking off, Flash-Next did so in 9 of 12 replies,
and the chat takes those too; a whole page with no id, 2 of the 12, is `page`, the chat's next one its next version),
with an id for each part, and to
change it in place with an edit block, ```` ```edit panda ````, holding only what changes: a whole element by its id
(a new one is added on top, or for a background in place of the old one), a search and replace, or a shape without an
id, which replaces the one at its place; ```` ```edit panda-night from panda ```` makes a changed copy. It is plain text
in the reply, so any backend's model can write it. An artifact has one canvas in the chat, in the reply that changed it
last: an edit moves it there, the same figure, and paints into the picture as it stood, each change landing as it
arrives, while the reply it left keeps a line naming the version made there. The canvas's caption names its version
(`panda · v2 · 2 changes`) and steps through the versions in place (‹ ›, redrawing only what differs); what a version
changed is a click away (Changes: its source against the version before, line by line). Measured with Qwen3.8-Flash-Next: a panda drawn in 1,987 tokens, its coat made red in 325, a copy under
a starry sky in 956, a cat's eyes made green beside another picture in 272. An edit that does not apply in full is
asked for again, once, with the artifact as it is now. The shelf (the shapes button in the top bar, or Open on an
artifact) is a card floating over the chat, moved by its head and sized by its corner, and kept where it is left: the
chat's artifacts as tabs, the one picked large with its versions, and with Compare the version before beside it (or
the original of a copy), each side stepped through its versions; it follows an artifact being drawn. Over a picture the
part under the pointer glows; a click selects it, kept in the message box, and the next question tells the model that
part, quoting it as it is now. In voice mode the part under the pointer as a question is said goes with it ("make this
orange"). What can be done by hand needs no model, in a canvas's edit mode (its Edit, on the shelf too; one canvas at
a time, so a stray click elsewhere only points): a row over it in the chat's own look names its tools and says what a
click does now. On a picture, Select drags the part selected to move it, a corner to resize it, and brings it forward or
sends it backward a step, duplicates or deletes it (the arrow keys nudge it; Ctrl+Z, Ctrl+D, Delete; Esc lets go, then
leaves); Fill paints whatever shape is clicked with its color. Each change is the source edited where that element is, the rest left byte for byte, and is kept at once as the
artifact's next version, the user's own (Undo steps back through it); stepped back to an older version, the canvas
offers Use this version. The next question carries the artifact whole as the user left it, which the model's next
edit starts from. A page artifact (a ```` ```html ```` or ```` ```aframe ```` document) paints as it is written, element by
element, its scripts waiting, and runs once whole; a later version goes into the page as it runs, changing only what
its source changed, so what the page holds stays (a game's score, a form's input), and only a change to its scripts
runs it anew, in a frame loaded behind the old one. Outside edit mode a page is the page (its own buttons and links); in
it, Select (point at a part; move it up or down, copy or delete it), Text (retype it where it is) and Fill (a background,
a shape's fill, an A-Frame entity's color). A page made to its window's size (a 3D scene's canvas) says so and gets a
frame fitted to the column. In an A-Frame scene the part under the pointer is the entity the
camera sees there. LaTeX documents (`web/latex.ts`, numbered and cross-referenced as LaTeX would, with KaTeX for the
math) and Markdown read as documents, block by block; a block points like a part, and in edit mode Text (or a double
click) turns it into its source to retype, and Select moves, copies and deletes blocks. Code is edited where it is shown (Edit).

Characters (`web/cast.ts`): every chat is with someone, as in SillyTavern. The chats without a character are Chrysalis's,
a character of its own: its card holds the system prompt and the default voice, and its name and picture can change
(until a picture is given, its card shows the ASCII background's butterfly, live, ripples and all).
SillyTavern's character cards (Character Card V2 and V3: a PNG with the card in it, or the
JSON), dropped anywhere or added with the + beside New chat, join the conversations as a row each with their portrait,
where their latest chat would be (a card already here, by its name and description, is that character again); a group
(a small JSON of our own: a name, the members, a scenario) is their faces in one circle, and the open chat's row lists
its chats. A portrait, in the list or beside a reply, opens the card: its fields as they are sent, edited where they are
read, the voice it speaks with in voice mode (picked and heard there: the voice service lists its voices and speaks a
sample with no microphone, loading only that voice's model, 230 MB against the whole stack's 566 MB), Export (the card
as a PNG any app reads, every field kept: a lorebook, other apps' extensions) and Delete. The user's persona is the row
at the bottom. Every message has who wrote it beside it: a reply its writer's portrait on the left, the user's own on
the right. A reply's speaker button reads it aloud with voice mode closed, in its writer's voice (a group's paragraphs
each in their speaker's, without their names), through the same service with only that voice's model loaded. A chat
with a character opens with its greeting (the card's others are versions of it). The card is the system prompt (`{{char}}` and `{{user}}` filled in, the
persona's description after it) and stays the same all chat, so every reply reads it from Chrysalis's cache; its
instructions for after the chat go at the end of the last question, since Chrysalis takes a system turn only first. A
group's reply is the scene: each line its speaker's name and a colon (over three turns Flash-Next tagged all 9 of its
paragraphs and kept the three voices apart), each paragraph shown with its speaker's portrait, once for a speaker who
goes on. A card dropped on a chat with characters joins it (as does a character's row dragged there): a group gains
them, and a chat with one character becomes a group of them all. The faces over the message box have the next reply be
one of them's, said in a note after the last message ("[Name next.]") so the model thinks first as it would for any
question (a reply begun with the name for it skips the thinking); Flash-Next began with the one named in 6 of 6 tries,
thinking or not. "Let them talk" plays four such beats, each one of them other than the last to speak. The cast is kept
in this browser's IndexedDB, on this device, never on the server; lorebooks are kept, not used yet.

Tools: the wrench beside the thinking bar (with how many tools are
on) opens them for this chat: Fetch URL and Sub-agents, then each MCP server with its state (not started, starting, running, failed
with the error and Retry, disabled), a switch for the whole server, and under its arrow one per tool, marked read-only
or asks first. The choice is the chat's, kept with it (export too); a new chat starts from the last one. A server starts
the first time a chat turns it on, keeps running, and after a failure starts again 1, 2, 4 ... 60 s later. The servers
come from `~/.config/chrysalis/mcp.json` (`CHRYSALIS_MCP_CONFIG` overrides the path), in Claude Desktop's and Claude
Code's shape, so a config can be pasted:
```json
{ "mcpServers": { "notes":  { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem",
"/home/me/notes"], "env": {}, "cwd": "/home/me" }, "remote": { "url": "https://example.com/mcp", "headers": {
"Authorization": "Bearer ..." }, "timeout": 120 }, "old":    { "command": "old-server", "disabled": true } }, "fetch": {
"allowPrivate": false } }
```
stdio (`command`, `args`, `env`, `cwd`) and Streamable HTTP (`url`, `headers`) servers, through the official SDK; SSE
servers are refused. A remote `url` must be https (its headers often hold a key; plain http only to this machine).
`tools` lists the only tools of a server the chat sees, whatever the server lists: a remote server can change its list
at any time, and its own read-only marks decide what runs without asking. Web search, as OpenCode does it: Exa's hosted
server, no key, search alone (`"exa": { "url": "https://mcp.exa.ai/mcp?tools=web_search_exa", "tools":
["web_search_exa"] }`; its `web_fetch_exa` would read pages through Exa, where Fetch URL reads them from here); the
searches go to Exa. `timeout` is a call's limit in seconds (60 by default), `maxResultChars` a result's size (40,000; a
longer one is cut, and says so). Tool names reach the model as `server__tool` (letters, digits, `_` and `-`, unique,
sorted, so the prompt's tools block stays the same and cached), the page shows the servers' own. The page cannot add,
edit or see a server's command line beyond what Settings shows, its environment or headers: a page that could define
one (here, or on a device let in from the network) would run anything on this machine. It can list, start, retry and reload, and every
request that starts or allows a tool must come as JSON, which another site's form cannot send. Fetch URL (read-only)
GETs http(s) pages (20 s, 5 redirects, 4 MB), gives HTML as compact markdown (title, headings, paragraphs, lists, links,
tables, code; no scripts, styles, menus or footers; `<main>` alone when there is one), text and JSON as they are, and
refuses other types; the result starts with the final URL and the HTTP status, and a long page comes in parts the model
can ask for (`start`; 20,000 characters at a time, Settings changes that). It refuses addresses that are not public
(loopback, private, link-local, shared, documentation, multicast, IPv6 unique-local and link-local, NAT64 and 6to4 of
those), checked in the connection's own DNS lookup at every hop, so neither a redirect nor a name that answers
differently the second time gets through; `"allowPrivate": true` (or a list of addresses and CIDR blocks) in the config
file is the only way to allow them. The reply's tool loop runs in the web server: Chrysalis's answer ends in calls, the
server runs them (side by side), sends Chrysalis the turn with its reasoning, the calls exactly as Chrysalis wrote them
and their results, and asks again, up to 24 rounds (then it stops and says so; Continue carries on). So a reply with
tools goes on with the tab closed, and a page that comes back follows all of its rounds. A tool that is not read-only
(MCP's `readOnlyHint`) asks first: the reply waits at the call, its card shows the arguments with Allow once, Always
allow (that tool, from then on, in every chat of the tab) and Deny, and a denied call tells the model the user declined.
A call to a tool the chat has not turned on gets an error result, not a crash. Stop ends a reply anywhere, a call too
(MCP's cancellation reaches the server). Calls show where they happened among each round's thinking and text, as cards
(the tool and its server, a line of its arguments, running with its time, asks to run, done in N s, failed, declined,
stopped) that open to the arguments and the result (markdown through the same sanitizer; a long one folds). Sub-agents
(`web/tool-agent.ts`, built in) let a reply hand a task to an agent of its own: the model calls `subagent` with a short
label and a self-contained task (OpenCode's shape, without its background runs), and the web server runs a fresh
conversation on it (a system prompt of its own, the chat's other tools, the reply's model and thinking) round after
round as a reply's, its calls asking first as the reply's do. Calls in one message run side by side (Chrysalis writes
three replies at once and queues the rest), and only a sub-agent's final answer goes back, as the call's result, so what
it read stays out of the reply's context. Its card shows the task's label, what it is doing (thinking, the tool it runs,
writing; once done, the start of its answer), its tool calls, tokens and time; open, the task as given, its rounds
(thinking, text, tool cards: one that asks first opens the card), its answer, and Stop for it alone, after which the
reply goes on with what it had written. Voice replies have none. A reply's stats count its rounds as one: time to its first token, tokens over the time spent writing them (each round's first
token to its last), prompts summed; the usage meter shows the context its last round left. Text files, pictures, PDFs
and videos can be attached (the clip, a drop anywhere, a paste): Chrysalis gets each file before the typed text, a text
file as its name and a fenced block, a picture as an image, a PDF as a file (its pages as pictures and text) and a video
as a video. Their contents live in the tab's IndexedDB (one screenshot is most of sessionStorage), and a closed tab's
are deleted when a Chrysalis page next loads. Development: `cd web && bun run dev` (hot reload), `bun run check`.

Replies come from whatever backend's address is in the top bar: it shows the address the web server is connected to
(this machine's Chrysalis, `http://127.0.0.1:8080/v1`, until another is set), and any server with OpenAI's chat API can
go there, as pasted (`localhost:11434`, with or without `http://` and `/v1`), with a key beside it if it wants one
(`web/backend.ts`). The web server checks it (its model list, with the key) and sends it every request from then on, so
no backend needs to let a website in (CORS: LM Studio and vLLM refuse any until they are set to, Ollama any but this
machine's), and tools, sub-agents, voice and replies that outlive their tab work with any of them. Only this computer's
own screen changes it: a device that could point the web server anywhere would read what only this machine reaches. The
key stays in `backend.json` beside the devices (readable by this user only) and goes to the backend alone; a page sees
only that there is one. An address off this machine and its networks takes https:// (typed without a scheme it gets
it; over http:// every chat and the key would cross the internet in the clear), and is kept only once the page has
said where every chat would go. A backend other than Chrysalis gets OpenAI's own fields and the samplers llama.cpp and KoboldCpp take by
those names, only those the user set: what is left unset is the backend's own (Settings says "default"). The model
picker lists its models (not embedding models), the thinking bar sends `reasoning_effort` ("none" for Off, which all
five honor, KoboldCpp after a line of thinking; nothing for Max, as Ollama refuses an effort for a model that cannot think), so at Max a model thinks as
its backend is set up: llama.cpp, LM Studio and Ollama by default, KoboldCpp and vLLM as the model's chat template does
unless started to (`--jinjathink true`, `--default-chat-template-kwargs '{"enable_thinking": true}'`; Qwen3.5-2B's does
not, and vLLM then files its answer as thinking). Thinking Ollama and vLLM send as `reasoning` reaches the page as the
others send it. Chrysalis is told apart by what its model list says of it (`owned_by`), not by its address, so a
Chrysalis anywhere gets its own fields; loading its models, starting it, and the context and cache meter are for this
machine's Chrysalis service, at its address (accepted as it is, stopped or loading too). What needs Chrysalis stays
Chrysalis's: token odds, continuing a reply from its middle and drafting (llama.cpp started over, KoboldCpp wrote into
its thinking), PDFs and videos, and the voice turn held to a tool call (llama.cpp, Ollama, LM Studio and KoboldCpp answered
`tool_choice: "required"` with text from a thinking model). `make test_backends` runs llama.cpp, Ollama, LM Studio, KoboldCpp and vLLM for real
(Rebuilding).

## Voice mode

The wave beside Send talks with the chat's model. The message box stays (typing still works): its row gains a pill whose
dots move with your voice and then Chrysalis's (its mic mutes), the voice beside it opens the voices, and Send becomes
■, which ends voice mode; voice replies think as the chat's thinking bar says (Off speaks at once). Your words show in
the chat as you say them, grey, and turn their own color as they are checked, with the misheard ones fixed and
underlined (what was heard on hover); a pause ends your turn. The pill says it is listening once the microphone gives
audio (a phone's took 2.9 s to open): from then on what you say is heard, while the speech models still load too. The
reply is spoken as it is written, except code blocks and tables, which are shown and not read: its words turn from grey
to their own color as they are said, timed within each piece of speech. Talking over it stops it. Both sides are
ordinary messages (a question said aloud is marked so), and a reply talked over keeps what was heard of it, to the word
it was on, which is also what the model is told it said, with a mark where it was cut off: kept as it was heard, cut
mid-sentence, it taught the model to stop so itself (an essay asked for next came back as a sentence or two).

The model is told about voice mode in a note on the first question said aloud (again when the note changes; another on
the first typed message after it), never in the system prompt, so voice mode does not make a chat read its history
again. What is said is checked as it is said, by the same model in a conversation of its own beside the chat (one per
voice session, so each check is an append to Chrysalis's cache: ~0.3 s), at temperature 0: the check is given the words
so far, the question before them and the start of the reply to it, and writes what speech recognition misheard (everyday
words too: what does not fit what the user asks), as swaps of the transcript's own words (`quinn three point eight
twenty seven B => Qwen3.8-27B`; asked to repeat the whole message instead, Flash-Next dropped sentences), each made only
if it sounds like what was heard (`web/heard.ts`: "tail" => "list" is a meaning, not a mishearing), then with tools
whether answering needs one (`tool: yes`). The chat is sent the question fixed and nothing of the check, so the reply
can neither speak nor copy it. The reply does not wait for its question to stand: it begins at the turn on the fixes
known by then, beside the check of the turn's last words, and when the turn model hears more to come (a turn then waits
up to 1.5 s of silence; 5 of 31 of the user's turns did), it begins at that pause on its words. Chrysalis writes it and
the service speaks it into the page's hold, its tools held by the server (`chrysalis_voice.ahead`, until `/go`): it is
shown and heard once the turn ended with its words and the check found what it was sent with, and is otherwise dropped
unheard (the user went on, or the check fixed more) and the question as it stands answered. Words said while a question
is still being checked join it. A yes, or a user who asks for a search, gets a tool call held to the tools' grammar
(~0.8 s on Flash-Next), after a spoken line the page writes ("Let me look that up."): a model with thinking off, told
its reply is spoken, says "I'm searching for that now" and ends its turn often enough, and then copies that turn. A
reply that announces a tool it did not call is made to call one; a reply that ran tools is kept when talked over, its
results with it. What is said before the speech models have loaded (a first start: ~1.1 s) is kept and heard once they
are in. Settings' "What the model is sent" shows a chat's next request as Chrysalis gets it (built by the same code as a
reply's: the tools as JSON, which Chrysalis's template writes first in the system turn, every message with its note). It
is a cascade, as Claude's voice mode is: speech-to-text, the model Chrysalis already runs (so the reasoning costs no
memory), text-to-speech, all on the CPU (no VRAM) in `voice/`, a Python service the web server starts with the first
voice session and stops 60 s after the last (`CHRYSALIS_VOICE_IDLE_MS`): closed, voice mode costs nothing. Open, it
holds ~0.55 GB of RAM and 4 threads (`CHRYSALIS_VOICE_THREADS`): at all 16 it took Flash-Next's decode from 95 to 3
tok/s, at 4 ~12%. `make voice` makes its venv (`data/voice-venv`) and fetches the default models into `data/voice/`
(~320 MB); `tools/voice_setup.sh NAME...` fetches the others.

What runs is a stack of four parts, each behind one small interface in `voice/parts.py`: the VAD (Silero v6),
speech-to-text (Parakeet 110M), the turn model (Smart Turn v3.2, its Whisper features in numpy) and text-to-speech
(Kokoro 82M int8; also Kitten, Supertonic 3 in 31 languages, Pocket TTS with a cloned voice). Speech-to-text is the one
model the service ships. On 200 Big Bench Audio questions (word error rate against their text) Parakeet 110M read 6.9%
where Kroko, a streaming zipformer, read 7.8% and Parakeet v3 0.6B 7.3% in 1.35 GB; and its search takes the
conversation's names and terms: the page sends the chat's text (`context`), and `terms()` picks from it the words with
capitals inside or digits and the words the system's dictionary lacks, which the search then favors. On 124 sentences in
4 Kokoro voices that say them as people do ("kwen"), it heard "Qwen" in 20 of 20 (0 without the conversation; Kroko's
search, 8 at best) and the terms in 89% (45% without), and never heard a term that was not said beside its sound-alikes
("When will the game", "My friend Jason"; at a higher score it heard "Qwen will the game"); Qwen3-ASR 0.6B, told the
terms, heard 95% of them in 2.8 GB and 2 s chunks. Kokoro has UTMOS 4.48, the best of the small voices. Settings is the
one place the choices live (voice model, voice, speed, the voice prompt), and the voice picker in the dock is a view of
the same values: it lists the service's voice models (those not installed dimmed, with how to fetch them) and their
voices by language and gender, each with a play button for a sample; a pick applies from the next sentence. The service
keeps no preferences: it lists what it has and loads what a session asks for, so a session switches voices mid-call.

The page and the service speak one protocol over a WebSocket (`/api/voice`, which the web server proxies as it is and
refuses from any other origin: WebSockets skip CORS), so any other stack can stand in for this one
(`CHRYSALIS_VOICE_CMD`, a JSON argv of a program that prints `voice: ws://127.0.0.1:PORT`). The page sends 16 kHz int16
audio in 20 ms frames and `hello`/`config` (the stack, voice and speed), `say`/`say_end` (the reply's text as it
streams), `cancel`, `playback` (whether the reply is audible), `context` (the conversation's text, for its names) and
`preview`; the service sends `ready` (the stack, its catalog of models and voices), `speech`, `partial` (`settled` once
speech recognition read the words at a pause), `pause` (the words so far, when the turn waits for more), `turn` (the
transcript, and `wait_ms` from the end of speech), `interrupt`, `said` and audio pieces (a header with the piece's text,
its range in the reply and its speech's length, then int16 PCM).

How it behaves, and why:
- A turn ends at 0.2 s of silence if Smart Turn says the sentence is finished, else at 1.5 s; the wait after speech is
  ~0.26 s (the silence, ~0.03 s for the VAD to let go, Smart Turn 25 ms). Parakeet reads the speech in pieces that
  Silero ends at pauses, each 0.1 s into its silence, so a turn's end has nothing left to read (read then, its last
  piece made the wait 0.35 s).
- While you talk the service reads the piece under way again every 0.3 s of speech (a piece ends at a pause or at 9 s),
  so your first words show ~0.6 s after you start rather than at the first pause.
- The reply is read in pieces that grow: the first ends at the first sentence or clause, or (no comma early) before a
  phrase's first word from the 5th word on ("The average distance from Earth | to the Moon ..."), since Kokoro reads 4
  words in ~0.2 s and a whole first sentence in up to twice that; later pieces are sentences, synthesized while the
  ones before play. Markdown, links and code are not read out; pieces end with the pause their punctuation calls for.
- Speakers bring the reply back into the mic. A turn that starts while a reply plays is presumed to be its echo until
  it holds two whole words the reply did not say (or one like "wait" or "stop"): then it interrupts (~0.9 s after you
  start talking) and the turn is what you said. Once 2 s of a reply have played without the VAD hearing it
  (headphones, or the browser's echo cancellation), the mic is taken not to carry it, and 0.26 s of speech over a
  reply stops it (~0.4 s). Echo cancellation lets a reply through now and then (a Mac's speakers): a leak puts it
  back on words only until 2 s of a reply play unheard again (~0.44 s after a leak, ~0.87 s when a leak was for good).
- Opening voice mode has Chrysalis read the chat with the voice prompt (one token): the prompt replaces the system
  prompt, and Flash-Next's first request after a while idle took 1.7-3.3 s to its first token.

Time from the end of a spoken question to the reply's first sound, six questions Chrysalis had not seen each (their
subjects drawn anew each run: Chrysalis answers a prompt it has seen from its cache), the last of which trails off ("...
a fun fact about owls and"), so the turn model hears more to come and the turn waits out 1.5 s of silence
(`web/bench-voice.ts`: headless Chromium's microphone plays a WAV through the real service and the running Chrysalis;
the page's own figure, then one from outside, the WAV's end of speech to the first audio buffer's start). Two runs each,
alternating with the page and service as they were before replies began ahead (each reply waited for its check, and
Kroko heard the speech):

| Qwen3.8-Flash-Next | median | wait after speech | turn to first sound | the question that trails off |
|---|---|---|---|---|
| replies begun ahead | **1.38 s, 1.35 s** (1.31, 1.33 outside) | 0.22-0.30 s | 0.96-1.43 s | 1.58 s, 1.92 s |
| each reply after its check | 1.50 s, 1.52 s (1.45, 1.46 outside) | 0.23-0.32 s | 1.04-1.30 s | 3.09 s, 2.62 s |

Flash-Next's cold experts and the speech models share the CPU: its replies decode at 35-55 tok/s while speaking
(95 alone) and Kokoro reads a piece ~1.4x slower. The service pinned to two cores of its own (`taskset` in
`CHRYSALIS_VOICE_CMD`) was no faster, 1.65 s and 1.42 s: Chrysalis's eight cold-expert workers are not pinned and still
run on those cores. For scale, Artificial Analysis measures Grok Voice 3.0 at 0.81 s
(without saying whether that includes the end of the user's turn).

## Measured on this machine

| | Chrysalis | llama.cpp (same GGUF, ROCm 10.1) |
|---|---|---|
| Context | **262,144** tokens (KV pool holds 276K) | 262K needs 3-bit weights or 4-bit KV |
| Decode, coding prompt, thinking, Qwen sampling | **100–114 tok/s** (MTP, 3 drafts) | 39.8 tok/s |
| Decode, greedy | **112–116 tok/s** | |
| Decode, rewriting a file the prompt holds | **216 tok/s** (prompt lookup; 144 with MTP alone) | |
| Two replies at once | **141 tok/s** together (107 taking turns), the second from its first token on | |
| Decode at 131K of real code (thinking, MTP) | **80 tok/s** (37 without drafts) | ~17 tok/s at 128K (reported) |
| Decode at 260K of real code | **64-69 tok/s** (31 without drafts) | |
| Prefill, 2K-token chunks | **1,350 tok/s** | 1,015 tok/s |
| Prefill, short appends (one forward pass) | **16 tokens in 43 ms, 64 in 73 ms, 256 in 222 ms** | |
| Prefill at 128K | **~740 tok/s**; filling 260K from empty: 377 s (690 tok/s) | ~390 tok/s (reported) |
| 1,024-token append at 260K | **2.4 s** to the first token | |
| Agent turn TTFT (small tool output) | **0.11–0.18 s**; large outputs add ~1,300 tok/s | full re-prefill whenever history diverges |
| OpenCode compaction | **0.2 s**: served from the cached conversation | full re-read: minutes at 200K |
| New OpenCode session TTFT (warm prefix) | **0.08 s** same project, **0.27 s** another project or day | several seconds |
| First OpenCode request after a restart or a wake | **0.46 s** (its shared prefix read back from disk; 6.5 s without) | |
| Load: start to serving, model file in the page cache | **1.5 s**, first reply at 1.8 s | |
| Load from disk (file not cached) | **3.7–3.9 s**, first reply at 4.3–4.5 s | |
| Unload: stop to process gone / VRAM free | **0.04 s / 0.19 s** | |

Quality:
- Kernels: mean KL divergence vs llama.cpp on identical weights 0.002 (98–100% top-1 agreement),
  far below quantization noise (`tools/compare_ref.py`).
- Weights: mean KL vs the Q8_0 original on 40K tokens of code and docs is 0.0100 (97.99% same top
  token), against 0.0111 (97.74%) for llama.cpp's own IQ4_XS recipe with the same imatrix (`tools/quality.sh`).
- Pictures (`tools/test_vision.py`): the vision tower against transformers' own in fp32 with Qwen's weights, worst token
  cosine 0.9993, relative error 0.45-1.2% (transformers in bf16, as usually served: 0.947 and 7.3%). As the answers see
  it: llama.cpp's next-token distributions after a photo, fed Chrysalis's embeddings rather than the official ones, move
  by a median KL of 5e-5, about what rounding the official ones to fp16 does (4e-5; bf16's: 4.5e-4). The language model
  after a picture against llama.cpp fed the same embeddings: mean KL 0.001-0.0011 (two positions on a knife's edge
  left out: 0.1% noise on the embeddings moves llama.cpp itself as much there).

## How it gets there

**Weights** — `tools/quant-mix.txt`: IQ4_XS for the bulk (FFN, projections), Q6_K for the sensitive
`ssm_out` / `attn_output` / output head, Q8_0 for K/V projections and the MTP head; 15.9 GB. At load
the GGUF blocks are split into aligned arrays (same bits) so kernels use 16-byte loads. Token
embeddings stay in pinned host RAM and are read by the GPU over PCIe (saves 0.7 GB of VRAM).

**Loading** (`engine/load.cpp`) — from a stopped server to serving in ~1.5 s, overlapping everything that
can overlap: the GPU wakes from runtime suspend (~0.6 s, started when the process starts) while Python
starts, the tokenizer loads and a thread pulls any uncached part of the file into the page cache (the disk
streams ~4.9 GB/s). Writer threads then copy the tensors in file order straight into a VRAM staging ring
through the large BAR (non-temporal stores, ~21 GB/s, one pass over host RAM) and the GPU relayouts each
32 MB piece into place as it lands. VRAM is allocated meanwhile in ~1 GB pieces (an allocation holds a HIP
lock while the kernel clears it); the KV pool takes what is left, the staging ring included once the
weights are in. The server listens from the start (`/health` answers 503 until the model is in) and then
warms up under its GPU lock (~0.3 s): pinned host memory for the embeddings (relayouted on the CPU) and
the snapshot slots, and the first kernel launches, which load the code objects. Both stall other HIP calls,
so they stay out of the weight upload.

**Decode** — int8 dot-product GEMVs (activations quantized per 32 values) at ~800–920 GB/s; up to three
matrices per launch (q|k|v, qkv|z); SwiGLU fused with quantization of the down projection's input; one
fused kernel per recurrent layer (gates, conv, L2 norms, delta-rule scan with the 128×128 state in
registers, gated norm). MTP drafts on the GPU without host round trips, then one verify pass;
rejection sampling against greedy drafts keeps sampling exact; the recurrent state is committed by
replaying the accepted tokens. Where the reply's last 16 tokens appeared earlier in the conversation (a file being
rewritten, a passage quoted back), the step copies up to 7 drafts from there instead (prompt lookup): a step of 8 costs
1.4x one of 4, and every match that long had all 7 accepted, where shorter ones lost to the MTP head. Each reply seeds its sampler from the request's `seed` (or a random one) and the
response carries it back (`seed`, in the finish_reason chunk when streaming): the same prompt, settings and
seed give the same tokens, from a cold cache, a snapshot or the live conversation. A prompt read in different
chunks (an older conversation re-read after a restart) has slightly different logits and may drift after a
while. The presence penalty counts the current reply's tokens only.

The sampler (`engine/sampler.h`, shared by both engines) applies the presence penalty and DRY (`dry_multiplier`,
`dry_base`, `dry_allowed_length`; sequence breakers are tokens holding a newline, `:`, `"` or `*`; it looks back
32K tokens), then top-k, temperature, top-p and min-p in the order `samplers` names (llama.cpp's field; stages
before the temperature judge the untempered odds). The default order draws exactly as before (`tools/test_sampler.cpp`).
`logprobs` / `top_logprobs` (up to 64, all of them) return each piece of text's log probability and alternatives, from the 64
candidates it was drawn among (the reasoning's under `logprobs.reasoning_content`; alternatives carry their token
`id`). `stop` (a string or a list) ends the answer where the first stop string to be completed begins; the reasoning and
tool calls are not searched. `response_format` (`json_schema`, `json_object`; the Responses API's `text.format`) holds the
answer to a JSON schema, and tool calls can be held to their tools' schemas (a tool marked `strict`, or `tool_choice`
`"required"` or naming a tool; Anthropic's `any` and `tool`; `parallel_tool_calls`): the right tool, the parameters its
schema names in its order (the required ones always), one of an enum's values, any other type as its JSON
(`chrysalis/structured.py`: llguidance makes them grammars over the model's tokens). A strict call in a free answer is held from
its `<tool_call>` on; with `tool_choice` (or a schema) the whole answer after the reasoning is (and the step ends at
`</think>`). A held answer's steps verify drafts as free text's do, the tokens the grammar forces and then the MTP head's
guesses, with each position's logits masked on the GPU to what the grammar allows there, so the rejection sampling stays
exact for the masked distribution: 2.6 tokens a step where one at a time it was one (a JSON answer at 101 tok/s where
55), no logprobs. `thinking_budget` (also in `chat_template_kwargs`; Anthropic's `budget_tokens`, OpenRouter's
`reasoning.max_tokens`) ends the reasoning at that many tokens with the words Qwen3's technical report closes it with
("Considering the limited time by the user, I have to give the solution based on the thinking directly now." and
`</think>`), and the model answers from there. A reply never ends inside its reasoning: an end of turn drawn there (Flash-Next, now and
then, on a greeting with tools offered) is dropped, the reasoning closed with `</think>` alone, and the model answers. `continue_final_message` (as in vLLM) writes on from the end of a final assistant message: Chrysalis uses the
reply's own tokens when it generated it, restates the reply as it now stands (a `chrysalis_prefix` delta), and serves
the prompt from its cache. With `chrysalis_original` (the reply as generated) the final message may be a cut of it, and
`chrysalis_append_token` adds a token after the cut: that rewrites a reply from a token's alternative. `chrysalis_aside`
answers without thinking while rendering the conversation as it is cached, and keeps that cache as it was (the web
UI's drafts).

**Attention** — paged KV cache (64-token pages), K int8, V 5-bit Hadamard-rotated (as accurate as int8
here, 18% smaller: what makes 262K fit). Decode and verify attention: a workgroup of 8 waves per key split
and KV head walks 128-key stages; each wave scores 16 of the keys for all query rows (K rows straight from
memory into WMMA fragments), the waves agree on the softmax max through LDS, and each accumulates the
output for its own 32 of the 256 dims, so a wave holds 16 accumulator registers instead of 128 and every KV
byte is read once for all draft tokens. K and V decode to fp16 with packed arithmetic (bytes into the
mantissa of 1024 by `v_perm`, two 5-bit values per and-or). 0.67 ms per layer at 262K (~700 GB/s; the read
floor is 0.50 ms), 0.79 ms for a 4-token verify. Prefill attention: 16-wave workgroups share 32-key LDS
stages, and each stage's raw K/V is loaded into registers during the previous stage's softmax and P·V, so
DRAM latency hides behind WMMA work: ~80 TFLOPS at 260K.

**Prefill** — chunks of up to 384 tokens (agent turns) run a fused GEMM that dequantizes weights in
registers straight into WMMA fragments: the weights are read once at 4–6 bits instead of being expanded
to fp16 first (that copy costs ~140 ms per pass). WMMA wants each fragment in both half-waves; here each
half-wave dequantizes or loads half of it and they swap with `v_permlanex16` (the halves may hold K in
different orders as long as A and B agree), halving that work. Longer chunks use hipBLASLt fp16 GEMMs
(~95 TFLOPS on this card vs ~60 for rocBLAS) on weights dequantized per layer, each shape's kernel chosen once per
machine by timing hipBLASLt's suggestions back to back (`engine/lt_tune.h`: a long prefill holds the card at its power
cap, where the kernel fastest in a short burst can be the slower one): 1000-token chunks 13% faster than with its first
suggestion, 2048-token ones 4%. Token-parallel conv,
LDS-staged recurrent scan (16 tokens per stage, so every workgroup of a layer is resident at once, and
4-way split dot products for a short dependency chain per token). The recurrent layers' gate projections (alpha, beta)
are written in fp32: rounded to fp16 as the other projections are, they shift every token's decay and the error
compounds over a long prompt (llama.cpp KL on prefill 0.0050 -> 0.0035; after a picture 0.19 -> 0.06 at its most
sensitive position).

**Caching (TTFT)** — `chrysalis/chat.py` renders the chat template straight to token ids, byte-identical
to the official template, and replays assistant turns Chrysalis generated as their exact ids (looked up by
their text, tool calls and reasoning, so two turns that said the same "OK" stay apart), so each agent step
only prefills the new tool output. Snapshots (recurrent state in RAM, KV pages shared copy-on-write) at
the end of the tool block, of the static system prompt (cut before OpenCode's date and environment
lines, so it is shared across projects and days), of the system turn, and of every prompt: that last one
keeps the next token's distribution, so a resent prompt (a retry, a redraw) needs no prefill at all and an
agent step is one forward pass. A prompt is always cut at these points, so a seed draws the same reply
from any cached state. Beyond what RAM holds (40 snapshots on the 27B, 16 on Flash-Next), the ones a session has moved on
from go first (an agent's older steps), then the least recently used: an agent that waits while its sub-agents work keeps
its place however many steps they take, and reads only their results when they are done. Pinned buffers for all snapshot slots are made during warm-up (Flash-Next: one 1.9 GB
region whose pages fault in while the warm-up runs; made one at a time, each cost a request 0.1-0.4 s in the
minutes after a load). The ones every session shares (the tool block, the static system prompt, the side requests'
instructions) are also written to disk (`data/cache/<model>`, 0.2-0.4 GB each, 6 GB kept per model), by a thread, from a
copy taken in milliseconds: after a restart or a wake from sleep, OpenCode's first request reads its 8K-token prefix
back in 0.2 s instead of prefilling it for 6.4 s. A file holds its tokens and is read only by the same model file and
engine build. The rest live in RAM and VRAM only.
Tokenization is memoized per message, so re-rendering a 200K-token conversation each turn costs ~2 ms.
Side requests (OpenCode's session title, sent 8 ms before the session's first request, and its summaries)
go last: they start after 0.3 s without an agent request, waiting outside the three request lanes meanwhile (an agent
that starts three sub-agents needs all three), and their instructions are kept like the tool block. OpenCode specifics: the date in its environment block is frozen per conversation, plan-mode
reminders are kept where they first appeared, and its compaction request (normally a full re-read of the
older part of the conversation) is answered by appending the summary request to the cached conversation,
rendered with that conversation's own settings. Extra system messages from plugins are merged into the
first one. Special-token text inside files stays text. A request that offers no tools never draws `<tool_call>`: a model can call
only the tools it was given. `CHRYSALIS_TRACE=1` adds each request's time split
(cache lookup, prefill pieces, snapshots, first step) to its log line; `tools/bench_opencode.py` replays
OpenCode 2.0.25's real requests (`tools/opencode2.json`) against a running server.

## Pictures

OpenAI `image_url` parts (data URLs; Chrysalis fetches nothing from the network) in user and tool messages.
`chrysalis/vision.py` preprocesses as the official processor does: the picture resized so its sides are whole 32-pixel
blocks and its area between 65,536 and 16,777,216 pixels (Pillow's antialiased bicubic: within 2 steps of 255 on under
1% of the pixels of transformers' torchvision resize, identical when no resize is needed), normalized to [-1, 1], cut
into 16-pixel patches. It renders as `<|vision_start|>`, one image token per merged 2 x 2 patch block, `<|vision_end|>`;
the tokens name the picture (`IMG_BASE + slot * IMG_ROWS + row`), a slot per picture by its content, so the prompt cache
tells pictures apart and a conversation that resends its pictures is still a pure append. Positions are M-RoPE: a
picture of r x c tokens starting at position p puts its token (i, j) at (p, p + i, p + j), and the text after it goes on
from p + max(r, c); the attention prep turns each of the 32 frequency pairs with the position its section names
(interleaved [11, 11, 10]), so for text nothing changes. A picture can be read at fewer pixels, for fewer tokens:
`max_pixels` on the image part (as qwen-vl-utils takes it) or on the request, OpenAI's `detail: "low"` (512 x 512),
never more than `chrysalis serve --max-pixels` lets the server read.

Videos (`video_url` parts, as vLLM takes them; data URLs) are read as Qwen's own processor reads them: frames taken 2 a
second (4 to 768; `fps` on the part), sized within 24,576 tokens for the whole video and at most 768 a frame pair
(transformers' video processor with qwen-vl-utils' per-frame cap), each pair of frames a temporal patch the tower reads as
it reads a picture (it attends within a pair; the patch embedding takes the two frames with their own weights instead of
summed ones). In the prompt each pair follows its time ("<1.5 seconds>") between markers of its own, and sits in M-RoPE
as a picture would: transformers' Qwen3VLProcessor and Qwen3.5's get_rope_index, which `tools/test_vision.py` checks the
frames, pixels, layout, positions and tower against. ffmpeg decodes the frames Chrysalis takes.

**The vision tower** (`engine/vision.hip`, transformers' `Qwen3_5VisionModel`): 27 ViT blocks (1152 wide, 16 heads of
72, 2D rotary embedding), a learned 48 x 48 position table resampled to the picture's grid, and a merger to the language
model's width. It takes no VRAM of its own: its weights (0.92 GB, the mmproj GGUF's bf16 as fp16: no tensor moves by
more than fp32 rounding) stay in pinned RAM and each block's are copied in on a second stream while the one before runs;
everything else lives in the engine's activation arena, idle between calls, which holds even the largest picture
(65,536 patches in 0.92 GB). Its activations are fp16 between steps (GEMMs on hipBLASLt accumulate in fp32), and the
residual stream fp32; the GEMMs that add into it, which reaches ~7,800 in a few channels (Flash-Next's ~11,400), are
scaled by 1/16 on the way out and back. An fp32-grade tower (each activation split into two fp16 halves, emulated)
would take twice the time and moves the answers no less than rounding the official embeddings to fp16 does, which is
all fp16 gets them (above). Attention is a WMMA flash kernel over all patches (the head dim zero-padded to 80). A 640 x 480 picture (300 tokens) takes 34 ms, a 1080p screenshot (2,040 tokens) 0.25 s, 4K 2.6 s, the largest
(16,384 tokens) 10 s; the first picture after a load also reads the weights (0.35 s). The embeddings (fp16) wait in pinned RAM until the
prefill reads them over PCIe as it reads the token embeddings; 2 GB of them are kept, the least recently used dropped.

## Qwen3.8-Flash-Next

48 layers (36 gated delta net, 12 sparse "QSA" attention), hidden size 2560 widened to 4 hyper-connection
streams, 512 experts per layer (10 routed + a shared one), a 54 GB n-gram embedding table read at layer 1,
and an MTP block with its own 512 experts. Chrysalis runs one layout, sized so the model fits next to a desktop
on 64 GB of RAM with no swapping:

- **GPU (20.2 GB):** dense weights Q6_K / Q8_0, the LM head Q6_K, the MTP block (its experts Q3_K), and a pool of
  5,600 expert slots: at load the experts used most in coding (IQ4_XS gate/up, IQ4_NL down), then the ones the replies
  pick (below). 3.3 GB of KV covers the 262K context.
- **System RAM (42.6 GB):** the other 18,976 experts: Q3_K gate/up, the least used fifth of all experts Q2_K,
  IQ4_NL down. K-quants run at the CPU's full memory bandwidth (the IQ3 formats are compute-bound there).
  Flash-Next takes ~40 GiB of RAM in all, and more as the expert pool moves while the machine has RAM to spare (below).
- **Disk:** the n-gram table stays in the source file, memory-mapped; a token's 16 rows are requested
  together so the cold ones are read in parallel, and the next token's while the current step finishes.

Which experts sit in VRAM comes from routing statistics, blended: what the model picks while decoding in
OpenCode-style agent sessions (0.6; counted on the GPU with `CHRYSALIS_FN_DECODE_COUNTS=file`, 69K tokens of
`tools/agent_sim.py` sessions over a dozen coding tasks), what it picks while reading code (0.25,
`tools_fn/route_stats.py`) and Unsloth's imatrix counts (0.15). Decode routing correlates little with
either: tuned on code prefill alone, the VRAM tier caught 41-45% of decode picks; this one catches ~59% on
held-out sessions (cross-validated by task), so a third fewer experts come from RAM per token, and 59% of
code prefill's (64% before).

**The expert pool.** That tier fits coding: a story finds a fifth of its picks there (20%, measured), where the 5,600
experts it uses most would serve nine tenths, and every pick the CPU computes is a read of its weights from RAM. So
the VRAM tier is a pool that follows the passes (`engine_fn/experts.cpp`): picks are counted per (layer, expert),
decayed with a half-life of 256 tokens, and an expert the CPU would read more megabytes of than the pool's least (by a
quarter) takes that one's slot. A background thread copies it into a VRAM stage (by DMA from RAM, or read from the
model file), and one launch per pass writes the arrivals into their slots and patches the routing tables, in stream
order with the passes. An expert keeps its own format wherever it is (the file tier's IQ4_XS, RAM's Q3_K / Q2_K inside
the slot's share of the IQ4_XS arrays): moving it changes where it is computed, not its weights. One leaving the pool
goes back to its RAM copy. The file's own tier has none: one of those leaving gets a copy in RAM (the spill mirror),
read from the file before the expert leaves the pool and freed when it comes back, up to the whole tier (15 GB, or
`CHRYSALIS_FN_SPILL_GB`) and only while the machine has 4 GB of RAM available (past that, other programs would swap for
it); with the mirror full only RAM-tier experts leave. Computed from the model file's mapping instead, they leave
decode depending on the page cache to keep them, which is the memory every other program takes first, and a story alone
pushes 4,300 of them (11.6 GB) out of the pool: with Chrysalis using 50 of the 62 GB, decode fell from 85 to 15-40
tok/s reading experts from the disk. Through a day's mix (`tools_fn/bench_fn.py 700 --mix`: code, a story twice, a
three.js page, code again) decode ran 72 / 58 / 83 / 44 / 69 tok/s computing them from the file with the page cache to
spare, and 81 / 70 / 97 / 83 / 82 with the mirror unbounded (11 GB by the second story; held to 6 GB: 81-88 / 72 /
85-88 / 76-77 / 76-84, 3 GB: 90 / 66 / 70 / 65 / 95, none: 88 / 44 / 40 / 60 / 84); three stories at once wrote 144
tok/s together with it unbounded, 116 held to 6 GB, 137 from the file. Once it has followed a reply the pool serves
93-95% of a story's picks and 78-85% of code's (code spreads wider: even each window's own 5,600 most used serve 83%).
Where an expert is computed changes its result by rounding, so results depend on what the pool holds: checks that need
bit-identical repeats hold it still (`chrysalis_tune` "experts" 0), and `tools_fn/compare_fn.py` checks every path
against llama.cpp after the pool has moved both ways.

**Decode.** Per layer the GPU routes (one fused router + top-10 kernel) and hands the cold picks over
through pinned coherent memory: the CPU workers (one per physical core) compute most of them with AVX-512
VNNI row kernels (each 64-weight register decoded straight from the GGUF blocks, group scales as lane vectors,
one horizontal sum per row: ~52-56 GB/s from RAM, near the ~62 GB/s the CPU's fabric link allows). The shared expert, the pool's experts and the GPU's own share of the cold experts (read from
RAM over PCIe) run side by side in two launches (gate/up, then down), so the PCIe-bound picks hide the pool's
ones. That share follows the DMA rate the pool's copies measure: the card's firmware moves its PCIe link between speeds
with load, and here it has come up for whole sessions at 2.5 GT/s (3.5 GB/s) and at 8 GT/s (12.5 GB/s) within the hour.
One kernel waits for the CPU, sums and updates the residual streams. While the CPU works, 8
workgroups on a second stream read the next layer's dense weights into the 96 MB Infinity Cache, where the
GPU reads them ~2x faster (more workgroups would crowd out the expert kernels). This and the lookahead below run only
while the pool serves under three quarters of the picks: with few cold picks the CPU's turn is short and their work
crowds the expert kernels (measured: 3-10% faster with the file's tier alone, 5-10% slower at 80-95% hits). Wave reductions and the
router's top-10 use DPP / `v_permlanex16` lane exchanges and branch-free compares instead of LDS
round trips. CPU and PCIe share the RAM's bandwidth, which bounds the cold experts' part of a token. That
bandwidth idles while the GPU runs a layer's token mixer, so a fourth stream runs the layer's own MoE mixer
and router on a copy of its input residual meanwhile and publishes the predicted CPU picks (they catch ~70%
of the cold picks); the workers load those experts' weights into their caches until the real handoff
arrives (prefetch hints are dropped at this volume; plain loads are not). It changes no result and gains
1-2% at short context, ~4% at 64K.

**Prefill.** A prompt goes through in passes of 8,192 tokens. Within a pass, each layer runs its mixers,
attention and recurrent state over sub-chunks of 1,024 tokens, then its routed experts once for the whole
pass, so every expert in RAM is read once per pass rather than once per chunk. The pool's experts run as grouped
WMMA GEMMs (a launch per format) over work items of up to 256 of an expert's tokens: a workgroup dequantizes its 32 weight rows into
LDS once per half super-block and each of its waves runs 16 tokens against them (gate/up 8 waves, down 16:
up to ~80 TFLOPS). Each sub-chunk's picks and MoE inputs go to the host on the copy stream while the next
sub-chunks run, so planning a layer's experts waits only for the last one. The sparse attention gives
each (token, KV head) a workgroup of 8 waves over 128-key stages of the token's selected blocks, split by
output dims as in the 27B's decode kernel. Of the cold experts, the CPU computes the ones with few tokens in place (an
AVX-512 VNNI kernel with tokens as lanes) and the GPU DMAs the rest into a staging ring (the file tier's experts out of
the pool through pinned memory, copied from the mirror). The split follows a
cost model with rates measured at load (the DMA rate kept current, above) and a feedback term that keeps the CPU and GPU
finishing together. The
feedback runs for the first ten passes after a load and then stays put: the CPU and GPU round differently, so a
split that kept following timing made the same prompt prefill to different logits each time; the settled one
is as fast (frozen at any point between 1.5x and 2x the CPU estimate, within 2% of the adapting split, and up to
10% faster than its oscillation on long passes), so the split does not move results (the pool does, above).

**Memory.** The model is sized to fit, so there is nothing to guard: Chrysalis loads it and runs. Switching to
the 27B frees all of it (no experts are kept in RAM), and switching back reads the file again: 7 s once the 27B has
read 45 GB of it into the page cache (Notes), 17-19 s from the disk. Pages of the
model file are dropped from the page cache once copied, the mirror's copies too.

**Measured** (7800X3D, 64 GB DDR5, RX 7900 XTX):

| | |
|---|---|
| decode, no drafts / 2 MTP drafts | 67 / 93 tok/s (greedy 61 / 99; in a coding conversation: 58 / 85) |
| decode, a story, 2 drafts: the first reply / the next | 61 / 86 tok/s |
| prefill 16K tokens | 7.0 s |
| append to a 16K conversation: 256 / 1,024 / 4,096 tokens | 0.44 / 0.90 / 2.2 s to the first token |
| at 260K context: decode, no drafts / 2 drafts | 57.7 / 65 tok/s (27B: 30.6 / 69 with 3 drafts) |
| at 260K context: 1,024-token append | 0.91 s to the first token (27B: 2.38 s) |
| filling 260K tokens from empty | 106 s, 2,450 tok/s (27B: 377 s) |
| load / switch from the 27B | 17-18 s / 7.0 s (19 s with nothing of the file cached) |
| OpenCode (`tools/bench_opencode.py`): first request after a load / new session / new day or project | 5.9 / 0.14 / 0.5 s to the first token |

Those rows are `tools_fn/bench_fn.py` in a session whose PCIe link ran at 8 GT/s. In one at 2.5 GT/s (above) prefill
and appends take about twice as long (16K: 15.2 s) and decode in the coding conversation 61 tok/s.

Speed hardly changes with depth (`tools/bench_depth.py`, same code base as context for both models): only 12 of
the 48 layers attend, each to a fixed 2,048 selected cells plus the indexer's scan, where the 27B's full-attention
layers read the whole cache for every token.

**Quality.** Mean KL divergence against the Q8_0 original on llama.cpp (`tools_fn/compare_fn.py`): 0.011-0.016,
top-1 agreement 96-99%. A long text that is mostly code pasted into a user turn scores ~0.06: the model's
predictions inside user turns are chaotic (two Chrysalis runs that differ only in rounding disagree by 0.025 there).

**Pictures** go through the same tower with Flash-Next's own weights, to its 2,560-wide embeddings, which are read
again for each request with new pictures (~0.35 s; the n-gram table's pages need the RAM). Against transformers in
fp32 its median token is at cosine 0.99999 and its relative error 0.8-2.9%, but the tower is chaotic at a few tokens:
two of the photo's 300 come out at 0.89 and 0.93, and emulating Chrysalis's arithmetic one fp16 rounding at a time shows
any of them can move those that far (transformers' own bf16 takes one to 0.08). The answers do not follow: fed
Chrysalis's embeddings instead of the official ones, llama.cpp's next-token distributions move by a median KL of 0.003,
about what rounding the official ones to fp16 does (0.0015). After a picture, Chrysalis against llama.cpp (fed the same
embeddings) is at a mean KL of 0.04 where the same conversation with a description in the picture's place is at 0.015: a
picture makes this model's predictions about three times as sensitive (two Chrysalis runs that differ only in rounding:
0.0075 and 0.0025), and its requantized weights are a larger change than rounding.

## Layout

```
engine/   common.h kernels.h gguf.h      constants, kernel entry points, GGUF reader
          load.cpp relayout.hip          loading: GGUF -> VRAM pipeline, KV pool, warm-up
          quant.hip                      weight arrays, GEMVs, dequantization
          attn.hip gdn.hip ops.hip       attention, gated delta net, norms / top-k
          model.cpp model.h api.cpp      weights, sequences, snapshots, forward, MTP, C API
          vision.hip vision.h            the vision tower (both engines)
          sampler.h                      the host sampler (both engines): penalties, DRY, ordered stages
chrysalis/
          engine.py chat.py cache.py server.py apis.py structured.py vision.py documents.py sleep.py link.py warm.py
          metrics.py __main__.py
web/      server.ts app.ts files.ts sampler.ts wire.ts      Bun web UI; tools.ts tool-fetch.ts tool-agent.ts tool-mcp.ts its tools; tls.ts HTTPS (--lan)
          voice.ts voice-worklet.js                     voice mode's client: the microphone, playback, the protocol
          ocean.ts logo.svg logo-thin.svg               a new chat's sea of characters and its butterfly (WebGL2, else the CPU)
          setup.ts device.ts devices.ts                 the HTTPS chat on the network: a device's setup and pairing pages (on
                                                        the sea, under the aurora), the devices let in
          index.html style.css tokens.css               (make test_web: sampler.test.ts, server.test.ts with fake-mcp.ts
                                                        and fake-voice.ts, browser.test.ts: the page in headless Chromium,
                                                        and Firefox for the network's trust check;
                                                        bench-voice.ts: voice mode's latency end to end)
voice/    __main__.py talk.py parts.py                  the voice service: its WebSocket, a session's turns and replies,
                                                        the parts (make voice / tools/voice_setup.sh; make test_voice)
tools/    tests (make test: test_quant, test_attn, test_gdn, test_sampler, test_chat, test_cache, test_documents, test_snapshot,
          test_server, compare_ref.py: the next-token distributions against llama.cpp's; test_vision.py: preprocessing, the
          vision tower and prompts with pictures against transformers and llama.cpp, with vision_ref.py, ref_logits_mm.cpp;
          test_vision.hip runs the tower alone), benchmarks (bench_quant.hip, bench_opencode.py:
          OpenCode's real requests), agent_sim.py, lint_comments.py (make lint: the mechanical cases of AGENTS.md's comment rule)
engine_fn/                               the Flash-Next engine (same C API): cpu_moe.cpp cpu_vnni.cpp for the
                                         experts in RAM, moe.hip routing, ple.hip n-gram block, ...
tools_fn/ fn_quantize.py (builds chrysalis-fn.gguf), route_stats.py (routing counts for it), test_*_fn.cpp
          (make test_fn), compare_fn.py (make test_fn_model, with test_snapshot / test_server on Flash-Next),
          bench_fn.py, bench_dec_fn.cpp / bench_moeg_fn.cpp / bench_cpudec_fn.cpp / bench_cpujob_fn.cpp (decode / prefill
          MoE kernels / the CPU's decode handoff / the CPU's prefill job alone)
data/     (not code) rocm/ venv/ models/ llama.cpp/ (quantization tools + reference only), ref-venv/ (torch and transformers on
          the CPU: the vision references, Whisper's features), voice-venv/ voice/ (the voice service's venv and models)
```

## Rebuilding

`make` builds `build/libchrysalis.so` against the ROCm 10.1 tarball in `data/rocm`. `data/venv` is the server's Python:
numpy, tokenizers, Pillow and llguidance, and for the tests httpx, jsonschema, prometheus-client and the anthropic and openai
SDKs. The model file was made from bartowski's Q8_0 with llama.cpp's quantizer:
`llama-quantize --allow-requantize --imatrix Qwen3.8-27B-imatrix.gguf --tensor-type-file tools/quant-mix.txt
--output-tensor-type q6_k --token-embedding-type iq4_xs Qwen3.8-27B-Q8_0.gguf chrysalis-mix.gguf IQ4_XS`.

The vision tower is Unsloth's `mmproj-BF16.gguf` (unsloth/Qwen3.8-27B-GGUF) as
`data/models/vision/mmproj-27b-bf16.gguf`; its tensors are bit-identical to Qwen's own (`model.visual.*` in the first of
the 18 safetensors shards, `data/models/vision/qwen3.8-27b-shard1.safetensors`, which `tools/vision_ref.py` reads with
`preprocessor_config.json` beside it). Flash-Next's is unsloth/Qwen3.8-Flash-Next-GGUF's `mmproj-BF16.gguf` as
`mmproj-fn-bf16.gguf`, checked against Qwen/Qwen3.8-Flash-Next's first shard (`qwen3.8-flash-next-shard1.safetensors`,
its `config.json` as `flash-next-config.json`). `data/ref-venv` is a venv with CPU torch, torchvision, transformers, Pillow and
safetensors.

`make test_backends` runs what is in `data/backends`: Ollama's release tarball (`ollama-linux-amd64.tar.zst`) unpacked
in `ollama/`, its models in `ollama-models/` (`OLLAMA_MODELS=... ollama pull qwen3.5:2b`); KoboldCpp's
`koboldcpp-linux-x64-nocuda` as `koboldcpp/koboldcpp`; vLLM's CPU wheel (its GitHub release's `+cpu` one) in a venv at
`vllm/venv`, with torch from PyTorch's CPU index; and the model, lmstudio-community/Qwen3.5-2B-GGUF's `Q8_0` and
`mmproj-BF16` in `models/qwen3.5-2b-gguf/` and Qwen/Qwen3.5-2B in `models/qwen3.5-2b-hf/`. llama.cpp is `data/llama.cpp`'s
build. LM Studio is its headless install (`LMS_NO_MODIFY_PATH=1` with lmstudio.ai's `install.sh`, in `~/.lmstudio`), the
model added to it with `lms import -l`.

`make fn` builds `build/libchrysalis_fn.so`. Its model file comes from Unsloth's Q8_0 split (plus the MTP file and
imatrix) via `python tools_fn/fn_quantize.py SRC_DIR data/models/flash-next/chrysalis-fn.gguf --hot 5600 --q2 0.2
--route data/models/flash-next/route_code.npy:0.25 --route data/models/flash-next/route_decode.npy:0.6` (~22 min);
`tools_fn/route_stats.py` measured the code-prefill counts on local code, `CHRYSALIS_FN_DECODE_COUNTS` the decode counts
in `tools/agent_sim.py` sessions (`--model=qwen3.8-flash-next`).

## Notes

- Up to three requests run at once, one per sequence resident on the GPU (more are kept as snapshots in RAM): a prompt
  is read a prefill chunk at a time with the GPU to itself, and while replies are being written it gives the GPU back to
  them after each chunk for as long as the chunk took (an 8K-token prompt read beside a reply: the reply's longest pause
  1.4 s where it waited out the whole prompt, the prompt read in 9.5 s where 5.6 alone, in the same chunks, so its reply
  is the same). The replies being written share each pass
  (`Engine::generate_batch`: one read of the weights for all of them, each with its own attention, recurrent state and
  sampler; 8 rows shared out as drafts). Two at once write 141 tok/s together in the engine where 107 in turn (115 / 91
  through the server), and the second reply starts when its prompt is read instead of after the first. Sampling stays
  exact: two replies (3 drafts each, as alone) draw for a seed the very tokens each would alone, with the same log probs
  (the web UI asks for them with every reply: three Flash-Next replies wrote 75 tok/s together taken in turn, 125 sharing). A model switch or sleep
  waits for the replies in flight. Flash-Next's replies share passes too, with one draft each (two rows a reply, up to
  four replies; more take none): the projections, routers and experts once for all rows, the CPU's cold experts too.
  With the expert pool warm, two replies write 120 tok/s together where 103 in turn with two drafts each, three 127
  where 102 (`tools/bench_decode.py 512 2 --together N --use qwen3.8-flash-next`); a second draft each measured no
  faster, the CPU reading more distinct cold experts per pass. Each row is computed as alone (the hyper-connection
  mixers four rows at a time, the cold picks' GPU share counted per reply), so a reply draws for a seed what it draws
  alone with one draft.
- The GPU stays awake while the server runs (no wake-up latency) and suspends when it stops.
- Chrysalis's own memory never swaps (`MemorySwapMax=0` in chrysalis.service): with a zram setup's high swappiness
  (Omarchy's 150) the kernel moved Flash-Next's cold experts to swap under light pressure, before file cache, and every
  reply then waited on swap-ins (12 s to a first token, 9 tok/s). On an installed unit: `systemctl --user set-property
  chrysalis MemorySwapMax=0`.
- The PCIe link (`chrysalis/link.py`): the card trains it when it resumes from runtime suspend, and here it comes up at 2.5 or
  at 8 GT/s from one resume to the next (both x16; 2 of 4 resumes in a row at each). At 2.5 GT/s DMA runs at ~3 GB/s
  instead of ~12, prefill takes twice as long and Flash-Next's decode loses a quarter. Chrysalis reads the link beside
  the first load, once the GPU is awake (asleep it reads x1), and when it is slower than the fastest this machine has
  shown (`data/cache/link.json`) the server restarts itself (exec) after the GPU has suspended, to have it trained
  again, up to 4 times (`chrysalis serve --link-tries`): a start whose link came up at 2.5 GT/s was back at 8 GT/s after
  one restart, serving 23 s after `chrysalis restart` where 17 s without. A link still slow then, or after a wake from
  sleep (a process holding the GPU cannot let it suspend), is named in the log, `chrysalis status`, `/chrysalis/stats`
  and `/metrics`.
- Switching to Flash-Next reads all of its file (63 GB, more than the RAM) at the disk's ~4 GB/s. While the 27B runs, it
  leaves most of the RAM free, so 10 s after a load Chrysalis reads Flash-Next's file into the page cache
  (`chrysalis/warm.py`), pausing while requests run and stopping 10 GB short of filling the free memory; the 27B's own
  file leaves the cache for it (its weights are on the GPU: a restart or wake then reads it from the disk, 3.9 s instead
  of 1.5). Measured: 45 GB read in 9 s, then the switch took 7.0 s where 19 s from nothing cached (9.8 s with 29 GB, the
  27B's file still cached). Flash-Next reads nothing for the 27B: its experts take 43 GB and its own working set (the
  n-gram table) lives in the page cache. The switch back reads the 27B's 15 GB from the disk:
  4.3 s.
- Before the machine sleeps, Chrysalis frees the GPU (`chrysalis/sleep.py`: a logind delay lock), so the model's VRAM is
  never evicted into RAM: a reply being written ends at its next step (cut short, as at its length limit), a prompt
  being read stops at its next 8K-token piece, and requests wait. After the wake the model loads again, its cache empty
  but for the shared prefixes kept on disk.
