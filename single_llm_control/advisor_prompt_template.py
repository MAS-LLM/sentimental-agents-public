# ─── Step 1: Topic ────────────────────────────────────────────────────────────────
SINGLE_TOPIC = """\
What are the pros and cons of hiring this candidate: {candidate_name}

(with this bio:

-----START OF CANDIDATE RESUME/BIO-----
{candidate_bio}
-----END OF CANDIDATE RESUME/BIO-----
)

for this role: {role_to_fill},
given this role description:

-----START OF JOB DESCRIPTION-----
{role_description}
-----END OF JOB DESCRIPTION-----

Be very upfront with how you feel about this candidate. Express your sentiments strongly and clearly.
"""

# ─── Step 2: Unified “Description” ───────────────────────────────────────────────
SINGLE_DESCRIPTION = """\
Please reply with a **description** of the UnifiedAdvisor, in {word_limit} words or less.
Speak in the first person, stick to professional experience and expertise,
tone it down (no superlatives), and give a clear point of view.
"""

# ─── Step 3: Unified “Priorities” ────────────────────────────────────────────────
SINGLE_PRIORITIES = """\
Please reply with the main **objectives and priorities** of the UnifiedAdvisor, in {word_limit} words or less.
Be concise, avoid corporate jargon, and give a focused point of view.
"""

# ─── Step 4: Unified “Criteria” ─────────────────────────────────────────────────
SINGLE_CRITERIA = """\
Please reply with a bullet list of the **evaluation criteria** the UnifiedAdvisor
should use to judge if a candidate is a good match for this role: {role_to_fill}.
Limit to 5 bullets or {word_limit} words. Speak directly and originally—no generic points.
"""

# ─── Step 5: Final Ask ────────────────────────────────────────────────────────────
SINGLE_ADVISOR_PROMPT = """{topic}

{description}

{priorities}

{criteria}

Now, as the UnifiedAdvisor, **synthesize** your pros vs. cons in one strong, emotionally upfront paragraph.
Base your reasoning strictly on your description, objectives, and criteria above.
Give a clear “overall_sentiment” in [–1, +1] at the end.
"""
