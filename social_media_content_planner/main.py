import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import requests
import time
import re
from datetime import datetime

load_dotenv()

def normalize_brand_key(s):
    return ''.join(c for c in s.lower() if c.isalnum())

def get_brand_from_wikipedia(brand_name):
    headers = {'User-Agent': 'ContentPlannerApp/1.0 (contact@example.com)'}
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": brand_name + " company",
            "format": "json",
            "srlimit": 1
        }
        for attempt in range(3):
            resp = requests.get(search_url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200 and resp.text.strip():
                break
            else:
                time.sleep(2 ** attempt)
        else:
            print("Failed to get Wikipedia search results after retries.")
            return None

        search_data = resp.json()
        results = search_data.get("query", {}).get("search", [])
        if results:
            page_title = results[0]["title"]
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title.replace(' ', '_')}"
            for attempt in range(3):
                sum_resp = requests.get(summary_url, headers=headers, timeout=10)
                if sum_resp.status_code == 200 and sum_resp.text.strip():
                    break
                else:
                    time.sleep(2 ** attempt)
            else:
                print("Failed to get Wikipedia summary after retries.")
                return None

            sum_data = sum_resp.json()
            desc = sum_data.get('description', '').lower()
            if 'company' in desc or 'brand' in desc or 'corporation' in desc:
                return {
                    'found': True,
                    'industry': sum_data.get('description', 'N/A'),
                    'goals': f"Boost awareness and drive engagement for {brand_name}",
                    'audience': "General audience",
                    'weeks': 2,
                    'platforms': ['instagram', 'twitter'],
                    'description': sum_data.get('extract', ''),
                    'brand_name': brand_name
                }
        return None
    except Exception as e:
        print(f"Wikipedia API error: {e}")
        return None

class GeminiSocialMediaPlanner:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            print("❌ Please add GEMINI_API_KEY to your .env file")
            self.gemini_available = False
            return

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini AI connected successfully!")
            self.gemini_available = True
        except Exception as e:
            print(f"⚠️ Gemini connection failed: {e}")
            self.gemini_available = False

        self.platforms = {
            'instagram': {'caption_limit': 2200, 'hashtag_limit': 30},
            'twitter': {'caption_limit': 280, 'hashtag_limit': 2},
            'linkedin': {'caption_limit': 3000, 'hashtag_limit': 5},
            'facebook': {'caption_limit': 63206, 'hashtag_limit': 5}
        }

    def generate_content_with_gemini(self, prompt):
        if not self.gemini_available:
            return "Gemini AI not available"
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Generation error: {e}")
            return "Error generating content"

    def create_content_plan(self, brand, industry, goals, audience, weeks=2, platforms=['instagram', 'twitter'], brand_info=None):
        print(f"\n🤖 Gemini AI is creating {weeks}-week content plan for {brand}")
        print(f"🎯 Campaign Goals: {goals}")
        print(f"👥 Target Audience: {audience}")
        print(f"📱 Platforms: {', '.join(platforms)}")
        print("-" * 60)

        content_plan = {
            'brand': brand,
            'industry': industry,
            'goals': goals,
            'audience': audience,
            'weeks': weeks,
            'platforms': platforms,
            'weekly_content': []
        }
        if brand_info:
            content_plan['brand_info'] = brand_info

        for week in range(1, weeks + 1):
            print(f"\n📅 Generating Week {week} content...")
            week_content = self.generate_weekly_content(
                brand, industry, goals, audience, week, platforms, brand_info
            )
            content_plan['weekly_content'].append(week_content)

        # Add color palette
        content_plan['color_palette'] = self.generate_color_palette(brand, industry, goals, audience)

        self.save_content_plan(content_plan)
        self.display_content_plan(content_plan)
        return content_plan

    def generate_weekly_content(self, brand, industry, goals, audience, week, platforms, brand_info=None):
        database_prompt = ""
        if brand_info:
            database_prompt = f"\nBrand Profile:\nDescription: {brand_info.get('description','')}\n"

        ideas_prompt = f"""Create 7 unique social media post ideas for {brand} (a {industry} company).
{database_prompt}
Campaign Goals: {goals}
Target Audience: {audience}
Week: {week}

Generate 7 different types of posts:
1. Educational/tip post
2. Behind-the-scenes content
3. User-generated content showcase
4. Product/service highlight
5. Industry news/trending topic
6. Interactive content (poll/question)
7. Inspirational/motivational post

Format as:
Day 1: [Post Type] - [Brief description]
Day 2: [Post Type] - [Brief description]
...continue for 7 days"""

        post_ideas = self.generate_content_with_gemini(ideas_prompt)
        ideas_list = self.parse_post_ideas(post_ideas)

        platform_content = {}
        for platform in platforms:
            platform_content[platform] = self.generate_platform_content(
                ideas_list, brand, platform, goals, audience
            )

        return {
            'week': week,
            'theme': f"Week {week}: {self.get_weekly_theme(goals, week)}",
            'post_ideas': ideas_list,
            'platform_content': platform_content
        }

    def generate_platform_content(self, ideas_list, brand, platform, goals, audience):
        platform_posts = []
        platform_specs = self.platforms.get(platform, self.platforms['instagram'])

        for i, idea in enumerate(ideas_list[:7], 1):
            caption_prompt = f"""Write an engaging {platform} caption for {brand}.
Post Idea: {idea}
Campaign Goals: {goals}
Target Audience: {audience}
Platform: {platform}
Character Limit: {platform_specs['caption_limit']}
Requirements:
- Hook the audience in the first line
- Match {platform}'s tone and style
- Include a clear call-to-action
- Encourage engagement (likes, comments, shares)
- Stay within {platform_specs['caption_limit']} characters
- Use appropriate emojis for {platform}
Caption:"""
            caption = self.generate_content_with_gemini(caption_prompt)

            hashtag_prompt = f"""Generate {platform_specs['hashtag_limit']} relevant hashtags for this {platform} post:
Brand: {brand}
Post Idea: {idea}
Campaign Goals: {goals}
Requirements:
- Mix of popular and niche hashtags
- Relevant to the post content
- Appropriate for {platform}
- Include brand-specific hashtags if relevant
- Maximum {platform_specs['hashtag_limit']} hashtags
Format: #hashtag1 #hashtag2 #hashtag3 (etc.)
Hashtags:"""
            hashtags = self.generate_content_with_gemini(hashtag_prompt)

            platform_posts.append({
                'day': i,
                'idea': idea,
                'caption': self.clean_caption(caption, platform_specs['caption_limit']),
                'hashtags': self.clean_hashtags(hashtags, platform_specs['hashtag_limit']),
                'post_type': self.suggest_post_type(idea)
            })

        return platform_posts

    def generate_color_palette(self, brand, industry, goals, audience):
        """Generate a 5-color palette with unique psychological meanings."""
        if not self.gemini_available:
            return []

        palette_prompt = f"""
Suggest a 5-color palette for {brand}, a company in the {industry} industry.
Campaign Goals: {goals}
Target Audience: {audience}

Requirements:
- Provide exactly 5 colors
- Each color must have:
  * a HEX code
  * a short color name
  * a description
  * a unique psychological meaning chosen from traits like:
    ["trust", "growth", "energy", "creativity", "calmness", "optimism", "confidence", "warmth", "professionalism"]
- No meaning should repeat across colors
- Respond ONLY in JSON (no extra text)
"""

        response = self.generate_content_with_gemini(palette_prompt)

        # Try direct JSON
        try:
            palette = json.loads(response)
            if isinstance(palette, list) and len(palette) >= 5:
                return palette[:5]
        except Exception:
            pass

        # Try extracting JSON substring
        try:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                palette = json.loads(match.group(0))
                if isinstance(palette, list) and len(palette) >= 5:
                    return palette[:5]
        except Exception:
            pass

        # Fallback static palette
        print("⚠️ Using fallback color palette.")
        return [
            {"color": "#0077CC", "name": "Ocean Blue", "description": "Reliable and stable", "meaning": "Trust"},
            {"color": "#FF9900", "name": "Sunset Orange", "description": "Energetic and enthusiastic", "meaning": "Energy"},
            {"color": "#66B3FF", "name": "Sky Blue", "description": "Imaginative and inspiring", "meaning": "Creativity"},
            {"color": "#80CBC4", "name": "Aqua Green", "description": "Balanced and refreshing", "meaning": "Growth"},
            {"color": "#FFD700", "name": "Golden Yellow", "description": "Positive and uplifting", "meaning": "Optimism"},
        ]

    def parse_post_ideas(self, ideas_text):
        ideas = []
        lines = ideas_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and ('Day' in line or line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.'))):
                if ':' in line:
                    idea = line.split(':', 1)[1].strip()
                else:
                    idea = line.strip()

                if len(idea) > 10:
                    ideas.append(idea)

        while len(ideas) < 7:
            ideas.append(f"Engaging content idea {len(ideas) + 1}")
        return ideas[:7]

    def clean_caption(self, caption, max_length):
        caption = caption.strip()
        if len(caption) > max_length:
            caption = caption[:max_length-3] + "..."
        return caption

    def clean_hashtags(self, hashtags, limit):
        hashtag_list = []
        words = hashtags.split()
        for word in words:
            word = word.strip()
            if word.startswith('#') and len(word) > 1:
                hashtag_list.append(word)
            elif word and not word.startswith('#') and word.isalnum():
                hashtag_list.append(f"#{word}")
        return ' '.join(hashtag_list[:limit])

    def suggest_post_type(self, idea):
        idea_lower = idea.lower()
        if any(word in idea_lower for word in ['video', 'reel', 'story']):
            return 'video'
        elif any(word in idea_lower for word in ['carousel', 'multiple', 'series']):
            return 'carousel'
        elif any(word in idea_lower for word in ['poll', 'question', 'quiz']):
            return 'interactive'
        else:
            return 'image'

    def get_weekly_theme(self, goals, week):
        themes = {
            1: "Brand Awareness & Introduction",
            2: "Product Showcase & Features",
            3: "Community Engagement & User Stories",
            4: "Conversion & Call-to-Action Focus"
        }
        return themes.get(week, f"Campaign Focus Week {week}")

    def save_content_plan(self, plan):
        filename = f"content_plan_{plan['brand']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(plan, f, indent=2, ensure_ascii=False)
            print(f"💾 Content plan saved as: {filename}")
        except Exception as e:
            print(f"⚠️ Could not save file: {e}")

    def display_content_plan(self, plan):
        print("\n" + "="*80)
        print(f"📱 GEMINI AI CONTENT PLAN - {plan['brand'].upper()}")
        print("="*80)
        for week_content in plan['weekly_content']:
            print(f"\n🗓️ {week_content['theme']}")
            print("-" * 60)
            print("\n📝 Weekly Post Ideas:")
            for i, idea in enumerate(week_content['post_ideas'], 1):
                print(f"   Day {i}: {idea}")
            for platform, posts in week_content['platform_content'].items():
                print(f"\n📊 {platform.upper()} CONTENT SAMPLES:")
                print("-" * 40)
                for post in posts[:2]:
                    print(f"\n   📅 Day {post['day']}: {post['post_type'].title()} Post")
                    print(f"   💭 Idea: {post['idea'][:50]}...")
                    print(f"   ✍️ Caption: {post['caption'][:80]}...")
                    print(f"   🏷️ Hashtags: {post['hashtags']}")
                if len(posts) > 2:
                    print(f"   ... and {len(posts)-2} more posts")

        if plan.get('color_palette'):
            print("\n🎨 Suggested Color Palette:")
            for c in plan['color_palette']:
                print(f"   {c['color']} - {c['name']}: {c['meaning']}")

        print("\n🎉 COMPLETE CONTENT PLAN GENERATED!")
        print("✨ Powered by Google Gemini AI")
        print("="*80)

def main():
    print("🤖 Gemini AI Social Media Content Planner")
    print("Generate Professional Content Plans with Google AI")
    print("-" * 60)

    planner = GeminiSocialMediaPlanner()
    if not getattr(planner, "gemini_available", True):
        print("\n❌ Gemini AI not available. Please check your API key.")
        return

    brand = input("🏢 Brand name: ").strip()
    wiki_info = get_brand_from_wikipedia(brand)

    if wiki_info:
        print(f"\n✅ Recognized brand '{brand}' from Wikipedia!")
        industry = wiki_info['industry']
        goals = input("🎯 Campaign goals: ")
        audience = input("👥 Target audience: ")
        weeks = int(input("📅 Weeks (1-4) [default 2]: ") or "2")
        weeks = min(max(weeks, 1), 4)
        platforms_input = input("📱 Platforms (comma separated) [default: instagram,twitter]: ")
        platforms = [p.strip().lower() for p in platforms_input.split(',')] if platforms_input else ['instagram', 'twitter']
        brand_info = wiki_info
    else:
        industry = input("🏭 Industry: ")
        goals = input("🎯 Campaign goals: ")
        audience = input("👥 Target audience: ")
        weeks = int(input("📅 Weeks (1-4) [default 2]: ") or "2")
        weeks = min(max(weeks, 1), 4)
        platforms_input = input("📱 Platforms (comma separated) [default: instagram,twitter]: ")
        platforms = [p.strip().lower() for p in platforms_input.split(',')] if platforms_input else ['instagram', 'twitter']
        brand_info = None

    content_plan = planner.create_content_plan(
        brand=brand, industry=industry, goals=goals, audience=audience,
        weeks=weeks, platforms=platforms, brand_info=brand_info
    )

if __name__ == "__main__":
    main()
