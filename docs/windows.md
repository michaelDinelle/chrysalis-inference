# Chrysalis on Windows, step by step

Windows 11 and a Radeon RX 7900 XTX with [AMD's driver](https://www.amd.com/en/support/download/drivers.html).
Every picture below is from a real run of these steps on that PC.

## 1. Get Chrysalis

Download [Chrysalis-Setup.exe](https://github.com/michaelDinelle/chrysalis-inference/releases/latest/download/Chrysalis-Setup.exe)
and open it from Downloads. If Windows says it protected your PC from an unrecognised app: **More info**, then
**Run anyway**.

<p align="center"><img src="../media/windows/01-downloads.png" width="620" alt="Chrysalis-Setup in the Downloads folder"></p>

## 2. Install it

| | |
|---|---|
| **I accept the agreement**, then **Next** | <img src="../media/windows/02-setup-licence.png" width="380" alt="The setup's licence page"> |
| **Next** | <img src="../media/windows/03-setup-folder.png" width="380" alt="The setup's folder page"> |
| Tick **Create a desktop shortcut** if you want one, then **Next** | <img src="../media/windows/04-setup-tasks.png" width="380" alt="The setup's shortcut page"> |
| **Install** | <img src="../media/windows/05-setup-ready.png" width="380" alt="The setup's ready page"> |
| **Finish**, with **Launch Chrysalis** ticked | <img src="../media/windows/07-setup-finished.png" width="380" alt="The setup's last page"> |

No administrator prompt: it installs for you alone.

## 3. First start

The terminal opens with the switches, and the chat opens in your browser. Nothing is downloaded yet.

<p align="center"><img src="../media/windows/08-first-start.png" width="620" alt="Chrysalis's terminal on its first start, with the butterfly and the switches, Model highlighted"></p>
<p align="center"><img src="../media/windows/09-page-first.webp" width="760" alt="The chat in the browser, its model button reading Download a model"></p>

## 4. Download a model

In the terminal, **Model** is already highlighted with **Qwen3.8-27B · download** picked: press **Enter**
(or **→**, then **Enter**, for Flash-Next). The chat's model button offers the same, and shows the download too.

<p align="center"><img src="../media/windows/10-downloading.png" width="620" alt="The 27B downloading, with its progress in the status line and a Download row to pause or cancel it"></p>

The **Download** row pauses or cancels it. When it finishes, the model loads:

<p align="center"><img src="../media/windows/11-ready.png" width="620" alt="The 27B downloaded and ready"></p>

## 5. Chat

<p align="center"><img src="../media/windows/12-chat.webp" width="760" alt="The chat answering why the sky is blue"></p>

## 6. Get Node.js (for OpenCode)

OpenCode gives the chat tools: files, a terminal, the web. It installs with npm, which comes with Node.js.

Open [nodejs.org/en/download](https://nodejs.org/en/download) and click **Windows Installer (.msi)**.

<p align="center"><img src="../media/windows/14-node-site.png" width="620" alt="nodejs.org's download page, with the Windows Installer button"></p>

Open it from the download list.

<p align="center"><img src="../media/windows/15-node-downloaded.png" width="560" alt="node-v26.13.1-x64.msi downloaded"></p>

| | |
|---|---|
| **Next** | <img src="../media/windows/16-node-welcome.png" width="330" alt="Node.js setup's welcome page"> |
| **I accept the terms**, then **Next** | <img src="../media/windows/17-node-licence.png" width="330" alt="Node.js setup's licence page"> |
| **Next** | <img src="../media/windows/18-node-folder.png" width="330" alt="Node.js setup's folder page"> |
| **Next** | <img src="../media/windows/19-node-features.png" width="330" alt="Node.js setup's features page"> |
| Leave it unticked, then **Next** | <img src="../media/windows/20-node-tools.png" width="330" alt="Node.js setup's tools for native modules page, unticked"> |
| **Install**. Windows asks *Do you want to allow this app to make changes to your device?*: **Yes** | <img src="../media/windows/21-node-ready.png" width="330" alt="Node.js setup's ready page"> |
| **Finish** | <img src="../media/windows/22-node-finished.png" width="330" alt="Node.js setup's last page"> |

## 7. Install OpenCode

Chrysalis's **opencode** button, top right, shows the line to run:

<p align="center"><img src="../media/windows/13-opencode-missing.png" width="520" alt="The OpenCode button's panel: not installed, with the line to copy"></p>

Open a new **Terminal** (Start menu, Terminal) and paste:

```powershell
npm.cmd install -g @opencode/cli
```

<p align="center"><img src="../media/windows/23-opencode-install.png" width="620" alt="npm installing OpenCode, with yellow warnings"></p>

The yellow warnings are normal. Check it:

```powershell
opencode.cmd --version
```

<p align="center"><img src="../media/windows/23b-opencode-check.png" width="620" alt="opencode v2.3.0"></p>

Type `npm.cmd` and `opencode.cmd`, not `npm` and `opencode`: Windows 11's PowerShell refuses to run their `.ps1` scripts
until its execution policy is changed, and the `.cmd` ones need no change.

## 8. Connect

Press the **opencode** button again. Its dot turns green, and the message box shows the computer and folder OpenCode
works in.

<p align="center"><img src="../media/windows/25-page-opencode.webp" width="760" alt="OpenCode connected: a green dot, and the computer and folder under the message box"></p>

Ask for something that needs a tool:

<p align="center"><img src="../media/windows/26-opencode-task.webp" width="760" alt="OpenCode running node --version and Chrysalis answering v26.13.1"></p>

## Removing it all

- Chrysalis, with its models and settings: the **Uninstall** row in its terminal, **Uninstall Chrysalis** in the Start
  menu, or the setup run again.
- OpenCode:

  ```powershell
  npm.cmd uninstall -g @opencode/cli
  ```

- Node.js: Settings, Apps, Installed apps, Node.js, Uninstall.
