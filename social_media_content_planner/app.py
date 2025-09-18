import streamlit as st
from main import GeminiSocialMediaPlanner, get_brand_from_wikipedia

def main():
    st.title("🤖 Social Media Content Planner with Gemini AI")
    st.write("Plan and generate AI-powered social media content.")

    st.sidebar.header("Campaign Settings")
    brand = st.sidebar.text_input("🏢 Brand name")
    industry = st.sidebar.text_input("🏭 Industry (e.g., Tech, Fashion, Food)")
    goals = st.sidebar.text_area("🎯 Campaign goals")
    audience = st.sidebar.text_input("👥 Target audience")
    weeks = st.sidebar.slider("📅 Number of weeks", 1, 4, 2)
    platforms = st.sidebar.multiselect(
        "📱 Platforms",
        ["instagram", "twitter", "linkedin", "facebook"],
        default=["instagram", "twitter"]
    )

    if st.sidebar.button("Generate Content Plan"):
        if not brand:
            st.warning("Please enter a brand name.")
            return

        planner = GeminiSocialMediaPlanner()
        if not getattr(planner, "gemini_available", True):
            st.error("❌ Gemini AI not available. Please check your API key.")
            return

        st.info("🔄 Generating content plan... this may take 1-2 minutes.")
        wiki_info = get_brand_from_wikipedia(brand)
        brand_info = wiki_info if wiki_info else None

        content_plan = planner.create_content_plan(
            brand=brand,
            industry=industry or (wiki_info['industry'] if wiki_info else "General"),
            goals=goals,
            audience=audience,
            weeks=weeks,
            platforms=platforms,
            brand_info=brand_info
        )

        st.success("✅ Content plan generated!")
        st.subheader(f"📱 GEMINI AI CONTENT PLAN - {brand.upper()}")

        for week in content_plan['weekly_content']:
            with st.expander(f"🗓️ {week['theme']}"):
                st.write("### 📝 Weekly Post Ideas")
                for i, idea in enumerate(week['post_ideas'], 1):
                    st.write(f"**Day {i}:** {idea}")

                for platform, posts in week['platform_content'].items():
                    st.write(f"### 📊 {platform.capitalize()} Samples")
                    for post in posts[:2]:
                        st.write(f"📅 **Day {post['day']}** ({post['post_type'].title()} Post)")
                        st.write(f"💭 Idea: {post['idea']}")
                        st.write(f"✍️ Caption: {post['caption']}")
                        st.write(f"🏷️ Hashtags: {post['hashtags']}")
                    if len(posts) > 2:
                        st.write(f"... and {len(posts)-2} more posts")

        if "color_palette" in content_plan:
            st.subheader("🎨 Suggested Color Palette")
            for c in content_plan["color_palette"]:
                cols = st.columns([0.2, 0.8])
                with cols[0]:
                    st.markdown(
                        f"<div style='width:40px;height:40px;border-radius:8px;background-color:{c['color']}'></div>",
                        unsafe_allow_html=True
                    )
                with cols[1]:
                    st.write(f"**{c['name']}** ({c['color']})")
                    st.caption(f"✨ {c['description']}")
                    st.caption(f"💡 Meaning: {c['meaning']}")

if __name__ == "__main__":
    main()
