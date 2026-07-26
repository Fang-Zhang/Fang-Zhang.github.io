## Introduction
This is my personal blog built with [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy), a Jekyll theme that I have been developing for years.

## Tech Stack
- [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) - A Jekyll theme for bloggers
- [GitHub Pages](https://pages.github.com/) - Hosting service for static sites
- [GitHub Actions](https://github.com/features/actions) - CI/CD service for automating workflows
- [VS Code Dev Containers](https://code.visualstudio.com/docs/remote/containers) - Development environment in a container

## How to write blog posts
- Create a new markdown file in the `_posts` directory with the format `YYYY-MM-DD-title.md`.
- Use the following front matter at the top of the file:  
```yaml
---
title: "Your Post Title"
date: YYYY-MM-DD HH:MM:SS +0000
categories: [category1, category2]
tags: [tag1, tag2]
---
```
- Write your content below the front matter using markdown syntax.  
- You can also include images by placing them in the `assets/images` directory and referencing them in your markdown like this:  
```markdown
![Alt text](/assets/images/your-image.jpg)
```

## Local Development
- Clone the repository and open it in VS Code.
- Use the Dev Containers extension to open the project in a containerized environment.
- Build the code and install dependencies by running the following command in the terminal:
```bash
bundle install
bundle exec jekyll build
```

- Run the following command in the terminal to start the Jekyll server:
```bash
bundle exec jekyll serve
```
- The site will be available at `http://localhost:4000`.
- Make changes to your blog posts or the site configuration, and the server will automatically reload to reflect your changes.
- When you're ready to publish your changes, commit and push them to the `main` branch, and the GitHub Actions workflow will handle the deployment to GitHub Pages.
- 

## Deployment
- Build and deploy the site to GitHub Pages using GitHub Actions on every push to the `main` branch.
- The workflow is defined in `.github/workflows/pages-deploy.yml`.

## Google Analytics (GA4)
- This site supports Google Analytics through Chirpy's built-in analytics integration.
- Set your GA4 Measurement ID in `_config.yml`:
```yaml
analytics:
  google:
    id: G-XXXXXXXXXX
```
- For production, you can keep `_config.yml` unchanged and set repository secret `GOOGLE_ANALYTICS_ID` in GitHub:
  - GitHub repo -> Settings -> Secrets and variables -> Actions -> New repository secret
  - Name: `GOOGLE_ANALYTICS_ID`
  - Value: your GA4 Measurement ID (for example `G-XXXXXXXXXX`)
- During GitHub Pages build, the workflow passes `GOOGLE_ANALYTICS_ID` to Jekyll and auto-populates `analytics.google.id` when config is empty.

## Email subscription (Kit / ConvertKit)
- A `Subscribe` tab is included and can post email signups to Kit (ConvertKit).
- Configure the following in `_config.yml`:
```yaml
subscriptions:
  provider: kit
  kit:
    formkit_id: 6fdc5182c0
    form_url: https://fang-zhang.kit.com/6fdc5182c0
    link_text: Subscribe
    script_url: https://f.convertkit.com/ckjs/ck.5.js
    form_action: # optional inline form fallback
    email_field_name: email_address # optional
    redirect_url: https://fang-zhang.com/subscribe-success/
```
- After setting `formkit_id` and `form_url`, the popup subscribe link is live at `/subscribe/`.
- A built-in success page is available at `/subscribe-success/`.

## Comments (Utterances)
- This site uses Chirpy's built-in comments integration with Utterances (GitHub Issues based).
- The comments provider is enabled in `_config.yml`:
```yaml
comments:
  provider: utterances
  utterances:
    repo: Fang-Zhang/Fang-Zhang.github.io
    issue_term: pathname
```
- One-time setup required:
  - Install the Utterances GitHub App for the repository: https://github.com/apps/utterances
  - Ensure GitHub Issues are enabled in the repository settings.
- After setup, each post page will automatically show a comment box and create/link issues by pathname.

## Comments troubleshooting
- Comment box not showing:
  - Confirm `comments.provider` is set to `utterances` in `_config.yml`.
  - Confirm each post allows comments (global default is `comments: true` under post defaults).
  - Rebuild/redeploy the site after config changes.
- `utterances` shows `Error: Bad credentials` or permission errors:
  - Reinstall or re-authorize the Utterances GitHub App for this repository.
  - Ensure the repository is selected during app installation.
- New comment cannot be submitted:
  - Verify the repository has Issues enabled.
  - Verify the visitor is signed in to GitHub.
- Duplicate issue threads for one post:
  - Keep `issue_term: pathname` stable and avoid changing post permalinks after comments exist.
- Private repository limitation:
  - Utterances works best with public repositories for broad reader participation.
