# frozen_string_literal: true

# Populate Chirpy's Google Analytics ID from environment when not set in _config.yml.
Jekyll::Hooks.register :site, :after_init do |site|
  analytics = site.config["analytics"] ||= {}
  google = analytics["google"] ||= {}

  next unless google["id"].to_s.strip.empty?

  env_id = ENV["GOOGLE_ANALYTICS_ID"].to_s.strip
  google["id"] = env_id unless env_id.empty?
end
