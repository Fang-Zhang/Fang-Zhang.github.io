## Introduction
This is my personal blog built with [**Chirpy**][chirpy], a Jekyll theme that I have been developing for years. 

## Tech Stack
- [**Chirpy**][chirpy] - A Jekyll theme for bloggers
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
