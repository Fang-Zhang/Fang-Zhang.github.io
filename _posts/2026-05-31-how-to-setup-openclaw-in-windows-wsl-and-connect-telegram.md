---
layout: post
title: How to setup OpenClaw in Windows WSL and connect it with Telegram
date: 2026-05-31 23:45:00 +1200
categories: [Tech]
tags: [OpenClaw, WSL, Windows, Telegram, AI, automation]
---

OpenClaw is one of those tools that becomes much more useful once it is part of your daily environment. I wanted it running inside Windows WSL, but still reachable from Telegram, so I could talk to it from my phone and let it help with small tasks wherever I am.

This post is a practical setup note for getting that working.

## What this setup gives you

After this is done, you will have:

- OpenClaw running inside WSL on Windows
- your workspace and config living in Linux where it is easier to manage
- Telegram connected as a chat interface
- a simple way to message your agent from your phone

In short, WSL gives you a solid local runtime, and Telegram gives you a lightweight remote control surface.

## Why use WSL for OpenClaw

If you are on Windows, WSL is usually the cleanest place to run developer tools like OpenClaw.

A few reasons:

- package installation is easier in Linux
- shell tools behave more predictably
- file paths and scripts are more consistent with most docs
- it is easier to use git, node, ruby, python and other CLI tools together

You still get the convenience of Windows, but the agent runs in a more comfortable environment.

## Step 1: Install WSL

If you do not already have WSL installed, open PowerShell as Administrator and run:

```powershell
wsl --install
```

Then restart your machine if Windows asks you to.

After rebooting, open your Linux distro. Ubuntu is the easiest default for most people.

Check that WSL is working:

```bash
uname -a
```

If that prints Linux kernel information, you are good.

## Step 2: Install Node.js in WSL

OpenClaw runs in a Node.js environment, so install a recent Node version in WSL.

I prefer using `nvm` because it keeps versions manageable:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
```

Then reload your shell and install Node:

```bash
nvm install 24
nvm use 24
node -v
npm -v
```

If you already have a working Node setup, that is fine too. The important part is having a modern version available inside WSL.

## Step 3: Install OpenClaw

Once Node is ready, install OpenClaw in WSL.

Depending on how you prefer to manage global packages, one common pattern is:

```bash
npm install -g openclaw
```

Then verify it:

```bash
openclaw --version
```

If the command is found and prints a version, the core install is working.

## Step 4: Start OpenClaw and create your workspace

Launch OpenClaw and let it initialize its workspace and config.

A typical first check is:

```bash
openclaw status
```

You want to confirm the CLI is available and the runtime is healthy.

OpenClaw will usually create a workspace under your home directory, where files like these live:

- `AGENTS.md`
- `SOUL.md`
- `MEMORY.md`
- `HEARTBEAT.md`

This workspace becomes the agent's home.

## Step 5: Create a Telegram bot

To connect Telegram, you need a bot token.

In Telegram:

1. Open **@BotFather**
2. Run `/newbot`
3. Choose a bot name
4. Choose a unique bot username
5. Copy the bot token it gives you

Keep that token private. Anyone with it can control your bot integration.

## Step 6: Connect Telegram to OpenClaw

This is the part where you wire the Telegram channel into your OpenClaw runtime.

The exact config shape can vary by version, so the safest habit is:

- use the local OpenClaw docs first
- inspect gateway configuration fields before editing
- patch config instead of replacing it wholesale

In practice, that means checking the Telegram-related config path in OpenClaw, adding your bot token, and making sure the Telegram channel is enabled.

The pieces you generally need are:

- a Telegram bot token from BotFather
- the Telegram messaging channel enabled in OpenClaw
- the gateway running so incoming Telegram messages can reach your agent

Once configured, restart or reload OpenClaw if the config change requires it.

## Step 7: Test the Telegram connection

Now go to Telegram, open your bot chat, and send a simple message like:

- `hello`
- `status`
- `who are you`

If everything is working, OpenClaw should receive the message and reply in that Telegram conversation.

This is the nice moment of the setup: your local WSL agent suddenly feels portable.

## Step 8: Keep your workspace in git

Once the setup is stable, I recommend keeping your own workspace or related project files in git.

That gives you:

- a history of changes
- easy rollback when you experiment
- a cleaner way to move between machines

If your blog or notes live in another repository, WSL also makes those git workflows easier.

## A few practical tips

A few things that are worth remembering:

- run OpenClaw in WSL, not in a random Windows shell, if you want fewer environment surprises
- keep secrets like Telegram bot tokens out of public repos
- use git commits for workspace changes you want to keep
- check the current status before changing gateway config or restarting services
- if something feels off, verify the simplest things first: node version, CLI availability, config path, and whether the gateway is actually running

## Why this setup is useful

What I like about this setup is that it stays simple.

You do not need a huge cloud deployment just to talk to your own assistant. WSL gives you a local Linux environment on Windows, and Telegram gives you a familiar chat interface from anywhere.

That combination is enough for a surprisingly capable personal workflow.

## Final thoughts

If you are already comfortable with Windows but prefer Linux tooling, OpenClaw in WSL is a very natural fit. Adding Telegram on top makes it much easier to use in real life, not just when you are sitting in front of your computer.

For me, the interesting part is not only that it works, but that it lowers the friction between “I have an idea” and “I can ask my agent to help with it now”.

If you want, I can write a follow-up post next on one of these:

- how to structure `AGENTS.md`, `SOUL.md`, and `MEMORY.md`
- how to use OpenClaw with GitHub Pages and a personal blog
- how to make Telegram interactions feel more natural and useful
