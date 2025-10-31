#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/zhaokun/work/zhao-kun/vibevoice')

from backend.app import create_app

app = create_app()

print("=" * 60)
print("Flask App Configuration:")
print("=" * 60)
print(f"Static folder: {app.static_folder}")
print(f"Static URL path: {app.static_url_path}")
print()
print("Registered Routes:")
print("-" * 60)
for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
    methods = ','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
    print(f"{rule.rule:50s} {methods:20s} -> {rule.endpoint}")
print("=" * 60)

# Test the routing logic
print("\nTesting /voice-editor route:")
with app.test_client() as client:
    response = client.get('/voice-editor')
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.content_type}")
    # Check first 200 chars of HTML
    html = response.data.decode('utf-8')
    if '<h1' in html:
        import re
        h1 = re.search(r'<h1[^>]*>([^<]*)</h1>', html)
        if h1:
            print(f"H1 content: {h1.group(1)}")
