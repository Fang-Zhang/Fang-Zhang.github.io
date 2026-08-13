---
title: 中文
icon: fas fa-language
order: 6
permalink: /zh/
lang: zh
---

<div id="page-category">
  <ul class="content ps-0">
    {% assign zh_posts = site.posts | where: "lang", "zh" %}
    {% for post in zh_posts %}
      <li class="d-flex justify-content-between px-md-3">
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        <span class="dash flex-grow-1"></span>
        <span class="text-muted small text-nowrap">{{ post.date | date: "%Y-%m-%d" }}</span>
      </li>
    {% endfor %}
  </ul>
</div>
