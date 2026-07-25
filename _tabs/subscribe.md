---
icon: fas fa-envelope
order: 5
---

Get new posts in your inbox.

{% assign kit = site.subscriptions.kit %}
{% assign provider = site.subscriptions.provider | default: '' %}
{% assign email_field_name = kit.email_field_name | default: 'email_address' %}
{% assign kit_script_url = kit.script_url | default: 'https://f.convertkit.com/ckjs/ck.5.js' %}
{% assign kit_link_text = kit.link_text | default: 'Subscribe' %}

{% if provider == 'kit' and kit and kit.formkit_id and kit.form_url %}
<a data-formkit-toggle="{{ kit.formkit_id }}" href="{{ kit.form_url }}" class="btn btn-primary">
  {{ kit_link_text }}
</a>
<script async src="{{ kit_script_url }}"></script>
{% elsif provider == 'kit' and kit and kit.form_action %}
<form action="{{ kit.form_action }}" method="post" target="_blank">
  <div class="input-group mb-3">
    <input
      id="kit-email"
      type="email"
      name="{{ email_field_name }}"
      value=""
      class="form-control"
      placeholder="you@example.com"
      aria-label="Email address"
      required
    >
    <button class="btn btn-primary" type="submit">Subscribe</button>
  </div>
  {% if kit.redirect_url %}
  <input type="hidden" name="redirect_url" value="{{ kit.redirect_url }}">
  {% endif %}
</form>
{% else %}
> Subscription is not configured yet.
>
> Set either `subscriptions.kit.formkit_id` + `subscriptions.kit.form_url` or `subscriptions.kit.form_action` in `_config.yml`.
{: .prompt-warning }
{% endif %}
